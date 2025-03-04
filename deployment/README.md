# Task Maistro

Um assistente inteligente de gerenciamento de tarefas construído com LangGraph.

## Descrição

Task Maistro é um assistente de gerenciamento de tarefas baseado em IA que ajuda os usuários a criar, organizar e gerenciar suas listas de tarefas. O sistema utiliza LangGraph para orquestrar fluxos de conversação inteligentes e manter o contexto das interações do usuário.

## Funcionalidades

- Criação e gerenciamento de tarefas (ToDos)
- Perfil de usuário personalizado
- Categorização de tarefas
- Estimativas de tempo para completar tarefas
- Acompanhamento de prazos
- Sugestão de soluções para tarefas

## Tecnologias Utilizadas

- Python 3.11
- LangGraph
- Docker
- Redis
- PostgreSQL
- LangSmith

## Estrutura do Projeto

O projeto é organizado em torno de um grafo LangGraph que gerencia o fluxo de conversação e as atualizações de estado:

- **Perfil do Usuário**: Armazena informações sobre o usuário como nome, localização, cargo e interesses
- **Tarefas (ToDos)**: Gerencia tarefas com campos como descrição, tempo estimado, prazo e status
- **Instruções**: Permite personalizar o comportamento do assistente

## Configuração e Instalação

### Pré-requisitos

- Docker e Docker Compose
- Python 3.11

### Instalação

1. Clone o repositório
2. Configure as variáveis de ambiente necessárias
3. Execute o Docker Compose:

```bash
cd app/deployment
docker-compose up -d
```

## Variáveis de Ambiente

O sistema utiliza as seguintes variáveis de ambiente:

- `REDIS_URI`: URI para conexão com Redis
- `POSTGRES_URI`: URI para conexão com PostgreSQL
- `DEEP_SEEK_API_KEY`: Chave de API para o modelo DeepSeek
- `LANGSMITH_API_KEY`: Chave de API para LangSmith

## Uso

Após a inicialização, o assistente estará disponível para interação através da API na porta 8123.

## Desenvolvimento

Para contribuir com o desenvolvimento:

1. Configure um ambiente virtual Python
2. Instale as dependências do projeto
3. Execute os testes para garantir que tudo está funcionando corretamente
