from typing import TypedDict, Literal, Optional
from datetime import datetime
from pydantic import BaseModel, Field
import uuid
import os

from langchain_core.messages import SystemMessage, HumanMessage, merge_message_runs
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from trustcall import create_extractor
from langgraph.store.base import BaseStore
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.graph import StateGraph, MessagesState, START, END
    
# Update memory tool
class UpdateMemory(TypedDict):
    """ Decision on what memory type to update"""
    update_type: Literal['user', 'todo', 'instructions']

# User profile schema
class Profile(BaseModel):
    """This is the profile of the user you are chatting with"""
    name: Optional[str] = Field(description="O nome do usuário", default=None)
    location: Optional[str] = Field(description="A localização do usuário", default=None)
    job: Optional[str] = Field(description="O cargo do usuário", default=None)
    connections: list[str] = Field(
        description="Conexão pessoal do usuário, como familiares, amigos ou colegas de trabalho",
        default_factory=list
    )
    interests: list[str] = Field(
        description="Interesses que o usuário tem", 
        default_factory=list
    )

# ToDo schema
class ToDo(BaseModel):
    task: str = Field(description="A tarefa a ser completada.")
    time_to_complete: Optional[int] = Field(description="Tempo estimado para completar a tarefa (minutos).")
    deadline: Optional[datetime] = Field(
        description="Quando a tarefa precisa ser completada (se aplicável)",
        default=None
    )
    solutions: list[str] = Field(
        description="Lista de soluções específicas, ações (e.g., ideias específicas, provedores de serviços ou opções concretas relevantes para completar a tarefa)",
        min_items=1,
        default_factory=list
    )
    status: Literal["not started", "in progress", "done", "archived"] = Field(
        description="Status atual da tarefa",
        default="not started"
    )

model = ChatOpenAI(model="deepseek-chat",
                   base_url="https://api.deepseek.com/beta",
                   temperature=0,
                   api_key=os.getenv("DEEP_SEEK_API_KEY"))

# Chatbot instruction for choosing what to update and what tools to call 
MODEL_SYSTEM_MESSAGE = """
Você é um chatbot útil.

Você foi criado para ser um companheiro para um usuário, 
ajudando-o a manter o controle de sua lista de tarefas.

Você tem uma memória de longo prazo que mantém o controle de três coisas:
1. O perfil do usuário (informações gerais sobre ele)
2. A lista de tarefas do usuário
3. Instruções gerais para atualizar a lista de tarefas

Aqui está o perfil do usuário atual (pode estar vazio se nenhuma informação tiver sido coletada ainda):
<user_profile>
{user_profile}
</user_profile>

Aqui está a lista de tarefas atual (pode estar vazia se nenhuma tarefa tiver sido adicionada ainda):
<todo>
{todo}
</todo>

Aqui estão as preferências atuais especificadas pelo usuário para atualizar a lista de tarefas 
(pode estar vazia se nenhuma preferência tiver sido especificada ainda):
<instructions>
{instructions}
</instructions>

Aqui estão suas instruções para raciocinar sobre as mensagens do usuário:

1. Raciocine cuidadosamente sobre as mensagens do usuário conforme apresentado abaixo.

2. Decida se alguma de suas memórias de longo prazo deve ser atualizada:
- Se informações pessoais foram fornecidas sobre o usuário, atualize o perfil do usuário 
chamando a ferramenta UpdateMemory com o tipo `user`
- Se tarefas forem mencionadas, atualize a lista de tarefas chamando a ferramenta UpdateMemory 
com o tipo `todo`
- Se o usuário tiver especificado preferências sobre como atualizar a lista de tarefas, 
atualize as instruções chamando a ferramenta UpdateMemory com o tipo `instructions`

3. Diga ao usuário que você atualizou sua memória, se apropriado:
- Não diga ao usuário que você atualizou o perfil do usuário
- Diga ao usuário quando você atualizar a lista de tarefas
- Não diga ao usuário que você atualizou as instruções

4. Erre do lado da atualização da lista de tarefas. Não há necessidade de pedir permissão explícita.

5. Responda naturalmente ao usuário user após uma chamada de ferramenta ter sido feita para salvar memórias, 
ou se nenhuma chamada de ferramenta tiver sido feita.
"""

# Trustcall instruction
TRUSTCALL_INSTRUCTION = """Reflita sobre a interação a seguir.

Use as ferramentas fornecidas para reter quaisquer memórias necessárias sobre o usuário.

Use chamadas de ferramentas paralelas para manipular atualizações e inserções simultaneamente.

Hora do sistema: {time}"""

# Instructions for updating the ToDo list
CREATE_INSTRUCTIONS = """Reflita sobre a interação a seguir.

Com base nesta interação, atualize suas instruções para atualizar a lista de tarefas.

Use qualquer feedback do usuário para atualizar como ele gostaria de ter itens adicionados, etc.

Suas instruções atuais são:

<current_instructions>
{current_instructions}
</current_instructions>"""

def extract_tool_info(tool_calls, schema_name="Memory"):
    """Extract information from tool calls for both patches and new memories.
    
    Args:
        tool_calls: List of tool calls from the model
        schema_name: Name of the schema tool (e.g., "Memory", "ToDo", "Profile")
    """

    # Initialize list of changes
    changes = []
    
    for call_group in tool_calls:
        for call in call_group:
            if call['name'] == 'PatchDoc':
                changes.append({
                    'type': 'update',
                    'doc_id': call['args']['json_doc_id'],
                    'planned_edits': call['args']['planned_edits'],
                    'value': call['args']['patches'][0]['value']
                })
            elif call['name'] == schema_name:
                changes.append({
                    'type': 'new',
                    'value': call['args']
                })

    # Format results as a single string
    result_parts = []
    for change in changes:
        if change['type'] == 'update':
            result_parts.append(
                f"Document {change['doc_id']} updated:\n"
                f"Plan: {change['planned_edits']}\n"
                f"Added content: {change['value']}"
            )
        else:
            result_parts.append(
                f"New {schema_name} created:\n"
                f"Content: {change['value']}"
            )
    
    return "\n\n".join(result_parts)

# Conditional edge
def route_message(state: MessagesState) -> Literal[END, "update_todos", "update_instructions", "update_profile"]:

    """Reflect on the memories and chat history to decide whether to update the memory collection."""
    message = state['messages'][-1]
    if len(message.tool_calls) ==0:
        return END
    else:
        tool_call = message.tool_calls[0]
        if tool_call['args']['update_type'] == "user":
            return "update_profile"
        elif tool_call['args']['update_type'] == "todo":
            return "update_todos"
        elif tool_call['args']['update_type'] == "instructions":
            return "update_instructions"
        else:
            raise ValueError

class Spy:
    def __init__(self):
        self.called_tools = []

    def __call__(self, run):
        # Collect information about the tool calls made by the extractor.
        q = [run]
        while q:
            r = q.pop()
            if r.child_runs:
                q.extend(r.child_runs)
            if r.run_type == "chat_model":
                self.called_tools.append(
                    r.outputs["generations"][0][0]["message"]["kwargs"]["tool_calls"]
                )

# Create the Trustcall extractor for updating the user profile 
profile_extractor = create_extractor(
    model,
    tools=[Profile],
    tool_choice="Profile",
)

# Node definitions
def task_mAIstro(state: MessagesState, config: RunnableConfig, store: BaseStore):

    """Load memories from the store and use them to personalize the chatbot's response."""
    
    # Get the user ID from the config
    user_id = config["configurable"]["user_id"]

    # Retrieve profile memory from the store
    namespace = ("profile", user_id)
    memories = store.search(namespace)
    if memories:
        user_profile = memories[0].value
    else:
        user_profile = None

    # Retrieve task memory from the store
    namespace = ("todo", user_id)
    memories = store.search(namespace)
    todo = "\n".join(f"{mem.value}" for mem in memories)

    # Retrieve custom instructions
    namespace = ("instructions", user_id)
    memories = store.search(namespace)
    if memories:
        instructions = memories[0].value
    else:
        instructions = ""
    
    system_msg = MODEL_SYSTEM_MESSAGE.format(user_profile=user_profile, todo=todo, instructions=instructions)

    # Respond using memory as well as the chat history
    response = model.bind_tools([UpdateMemory], parallel_tool_calls=False).invoke([SystemMessage(content=system_msg)]+state["messages"])

    return {"messages": [response]}

def update_profile(state: MessagesState, config: RunnableConfig, store: BaseStore):

    """Reflect on the chat history and update the memory collection."""
    
    # Get the user ID from the config
    user_id = config["configurable"]["user_id"]

    # Define the namespace for the memories
    namespace = ("profile", user_id)

    # Retrieve the most recent memories for context
    existing_items = store.search(namespace)

    # Format the existing memories for the Trustcall extractor
    tool_name = "Profile"
    existing_memories = ([(existing_item.key, tool_name, existing_item.value)
                          for existing_item in existing_items]
                          if existing_items
                          else None
                        )

    # Merge the chat history and the instruction
    TRUSTCALL_INSTRUCTION_FORMATTED=TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat())
    updated_messages=list(merge_message_runs(messages=[SystemMessage(content=TRUSTCALL_INSTRUCTION_FORMATTED)] + state["messages"][:-1]))

    # Invoke the extractor
    result = profile_extractor.invoke({"messages": updated_messages, 
                                         "existing": existing_memories})

    # Save the memories from Trustcall to the store
    for r, rmeta in zip(result["responses"], result["response_metadata"]):
        store.put(namespace,
                  rmeta.get("json_doc_id", str(uuid.uuid4())),
                  r.model_dump(mode="json"),
            )
    tool_calls = state['messages'][-1].tool_calls
    return {"messages": [{"role": "tool", "content": "updated profile", "tool_call_id":tool_calls[0]['id']}]}

def update_todos(state: MessagesState, config: RunnableConfig, store: BaseStore):

    """Reflect on the chat history and update the memory collection."""
    
    # Get the user ID from the config
    user_id = config["configurable"]["user_id"]

    # Define the namespace for the memories
    namespace = ("todo", user_id)

    # Retrieve the most recent memories for context
    existing_items = store.search(namespace)

    # Format the existing memories for the Trustcall extractor
    tool_name = "ToDo"
    existing_memories = ([(existing_item.key, tool_name, existing_item.value)
                          for existing_item in existing_items]
                          if existing_items
                          else None
                        )

    # Merge the chat history and the instruction
    TRUSTCALL_INSTRUCTION_FORMATTED=TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat())
    updated_messages=list(merge_message_runs(messages=[SystemMessage(content=TRUSTCALL_INSTRUCTION_FORMATTED)] + state["messages"][:-1]))

    # Initialize the spy for visibility into the tool calls made by Trustcall
    spy = Spy()
    
    # Create the Trustcall extractor for updating the ToDo list 
    todo_extractor = create_extractor(
    model,
    tools=[ToDo],
    tool_choice=tool_name,
    enable_inserts=True
    ).with_listeners(on_end=spy)

    # Invoke the extractor
    result = todo_extractor.invoke({"messages": updated_messages, 
                                    "existing": existing_memories})

    # Save the memories from Trustcall to the store
    for r, rmeta in zip(result["responses"], result["response_metadata"]):
        store.put(namespace,
                  rmeta.get("json_doc_id", str(uuid.uuid4())),
                  r.model_dump(mode="json"),
            )
        
    # Respond to the tool call made in task_mAIstro, confirming the update
    tool_calls = state['messages'][-1].tool_calls

    # Extract the changes made by Trustcall and add the the ToolMessage returned to task_mAIstro
    todo_update_msg = extract_tool_info(spy.called_tools, tool_name)
    return {"messages": [{"role": "tool", "content": todo_update_msg, "tool_call_id":tool_calls[0]['id']}]}

def update_instructions(state: MessagesState, config: RunnableConfig, store: BaseStore):

    """Reflect on the chat history and update the memory collection."""
    
    # Get the user ID from the config
    user_id = config["configurable"]["user_id"]
    
    namespace = ("instructions", user_id)

    existing_memory = store.get(namespace, "user_instructions")
        
    # Format the memory in the system prompt
    system_msg = CREATE_INSTRUCTIONS.format(current_instructions=existing_memory.value if existing_memory else None)
    new_memory = model.invoke([SystemMessage(content=system_msg)]+state['messages'][:-1] + [HumanMessage(content="Please update the instructions based on the conversation")])

    # Overwrite the existing memory in the store 
    key = "user_instructions"
    store.put(namespace, key, {"memory": new_memory.content})
    tool_calls = state['messages'][-1].tool_calls
    return {"messages": [{"role": "tool", "content": "updated instructions", "tool_call_id":tool_calls[0]['id']}]}

def graph():
    # Create the graph + all nodes
    builder = StateGraph(MessagesState)

    # Define the flow of the memory extraction process
    builder.add_node(task_mAIstro)
    builder.add_node(update_todos)
    builder.add_node(update_profile)
    builder.add_node(update_instructions)
    builder.add_edge(START, "task_mAIstro")
    builder.add_conditional_edges("task_mAIstro", route_message)
    builder.add_edge("update_todos", "task_mAIstro")
    builder.add_edge("update_profile", "task_mAIstro")
    builder.add_edge("update_instructions", "task_mAIstro")

    # Store for long-term (across-thread) memory
    # Checkpointer for short-term (within-thread) memory
    # We compile the graph with the checkpointer and store
    return builder.compile(checkpointer=MemorySaver(), store=InMemoryStore())