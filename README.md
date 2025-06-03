# Tech Challenge - Fase 4: API de Predição de Preços de Ações com LSTM

Este projeto implementa uma solução completa de Machine Learning para predição de preços de ações utilizando redes neurais LSTM (Long Short-Term Memory) através de uma API RESTful desenvolvida com FastAPI e deployada na AWS Lambda.

## Índice

- [Visão Geral](#visão-geral)
- [Arquitetura](#arquitetura)
- [Execução Local](#execução-local)
- [Deploy na AWS](#deploy-na-aws)
- [Uso da API](#uso-da-api)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Modelo LSTM](#modelo-lstm)
- [Tecnologias Utilizadas](#tecnologias-utilizadas)

## Visão Geral

O projeto atende aos requisitos do Tech Challenge da Fase 4 da Pós-Tech FIAP, implementando uma pipeline completa de Machine Learning para predição de preços de ações, composta por:

1. **Coleta e Pré-processamento de Dados**: Utiliza a biblioteca `yfinance` para obter dados históricos de ações, realizando limpeza, normalização e preparação dos dados para o modelo.
2. **Desenvolvimento e Treinamento do Modelo LSTM**: O modelo LSTM é desenvolvido com PyTorch Lightning, aproveitando recursos como Early Stopping, divisão automática de conjuntos de treino/teste e suporte a diferentes tamanhos de sequência. O pipeline de treinamento inclui validação contínua e cálculo de métricas (MAE, MAPE, RMSE, R²).
3. **Salvamento e Exportação**: Os modelos treinados, junto com seus metadados, são automaticamente armazenados no AWS S3, facilitando versionamento e reuso.
4. **Deploy da API**: A API RESTful, construída com FastAPI, é empacotada em container Docker e deployada na AWS Lambda, permitindo acesso escalável e serverless às funcionalidades de predição e treinamento.
5. **Escalabilidade e Monitoramento**: A solução é otimizada para produção em nuvem, com monitoramento de saúde, gerenciamento de variáveis de ambiente e arquitetura stateless para alta disponibilidade.

## Arquitetura

![Diagrama da Arquitetura](docs/arquitetura.drawio.svg)

### Componentes Principais

- **FastAPI**: Framework web para criação da API RESTful
- **PyTorch Lightning**: Framework para treinamento de modelos LSTM
- **AWS Lambda**: Servidor serverless para execução da API
- **AWS S3**: Armazenamento de modelos treinados e metadados
- **Docker**: Containerização para deploy consistente
- **Mangum**: Adaptador ASGI para AWS Lambda

## Execução Local

### Pré-requisitos

- Docker e Docker Compose instalados
- Arquivo `.env` configurado com credenciais AWS

### Configuração

1. Clone o repositório e acesse o diretório:
```bash
git clone git@github.com:gabrielguizardi/mlet3-tech-challenge-4-deep-learning.git lstm-api
cd lstm-api
```

2. Configure as variáveis de ambiente:
```bash
# Copie o arquivo de exemplo
copy .env.example .env

# Edite o arquivo .env com suas credenciais AWS
code .env
```

Exemplo do arquivo `.env`:
```
AWS_ACCESS_KEY_ID=sua_access_key
AWS_SECRET_ACCESS_KEY=sua_secret_key
AWS_SESSION_TOKEN=seu_session_token
S3_BUCKET_NAME=seu_bucket_name
```

### Executar com Docker Compose

```bash
# Subir o serviço
docker-compose up --build

# Para parar o serviço
docker-compose down
```

### Acessar a Aplicação

Após executar o comando acima, a aplicação estará disponível em:

- **API**: http://localhost:8000
- **Documentação Interativa (Swagger)**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/up

## Deploy na AWS

### Pré-requisitos para Deploy

- AWS CLI configurado
- Conta AWS com permissões para ECR, Lambda e S3
- AWS Academy Lab ativo (se aplicável)

### Passo 1: Criar Repositório no ECR

```bash
# Criar repositório ECR
aws ecr create-repository --repository-name lstm-api --region us-east-1

# Obter URI do repositório (anote o valor retornado)
aws ecr describe-repositories --repository-names lstm-api --region us-east-1
```

### Passo 2: Build e Push da Imagem

```bash
# Login no ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin {account-id}.dkr.ecr.us-east-1.amazonaws.com

# Build da imagem para produção
docker build -t lstm-api .

# Tag para ECR
docker tag lstm-api:latest {account-id}.dkr.ecr.us-east-1.amazonaws.com/lstm-api:latest

# Push para ECR
docker push {account-id}.dkr.ecr.us-east-1.amazonaws.com/lstm-api:latest
```

### Passo 3: Criar Bucket S3

```bash
# Criar bucket S3 para armazenar modelos
aws s3 mb s3://seu-bucket-name --region us-east-1
```

### Passo 4: Criar Função Lambda

- Acesse o AWS Lambda Console
- Clique em "Create function"
- Selecione "Container image"
- Configure:
    - Function name: `lstm-api`
    - Container image URI: `{account-id}.dkr.ecr.us-east-1.amazonaws.com/lstm-api:latest`

### Passo 5: Configurar Variáveis de Ambiente na AWS Console

- No Console AWS, acesse o serviço **Lambda**.
- Clique na função `lstm-api`.
- No menu lateral, selecione **Configuração** > **Variáveis de ambiente**.
- Clique em **Editar** e adicione a variável: `S3_BUCKET_NAME` = seu-bucket-name
- Clique em **Salvar**.

### Passo 6: Criar URL Pública para a Função Lambda


-  Ainda na função Lambda, vá até o menu **Função Lambda**.
-  No painel lateral, clique em **URL da função**.
-  Clique em **Criar URL da função**.
-  Em **Tipo de autenticação**, selecione **Nenhuma** (deixa a URL pública).
-  Clique em **Salvar**.
-  Copie a URL gerada para acessar sua API publicamente.


## Uso da API

### Funcionalidades Disponíveis

1. **Treinamento de Modelos** - `POST /models/train`
   - Coleta automática de dados históricos via Yahoo Finance
   - Pré-processamento e normalização dos dados
   - Treinamento de modelo LSTM com Early Stopping
   - Avaliação com múltiplas métricas (MAE, MAPE, RMSE, R²)
   - Armazenamento automático no S3

2. **Predição de Preços** - `POST /models/{model_id}/predict`
   - Carregamento automático do modelo do S3
   - Coleta de dados recentes para predição
   - Retorno de previsões de preços de fechamento

3. **Consulta de Dados** - `POST /models/fetch-data`
   - Acesso direto aos dados históricos de ações
   - Suporte a períodos personalizados ou últimos N dias

4. **Health Check** - `GET /up`
   - Verificação de status da API

## Estrutura do Projeto

```
api/
├── app.py                          # Aplicação principal FastAPI
├── Dockerfile                      # Container para AWS Lambda
├── Dockerfile.local                # Container para desenvolvimento local
├── docker-compose.yml              # Configuração Docker Compose
├── requirements.txt                # Dependências Python
├── error_handlers.py               # Handlers de exceções customizados
├── models/
│   └── lightning_lstm_model.py     # Implementação do modelo LSTM
├── schemas/
│   ├── fetch_data.py               # Schema para consulta de dados
│   └── train.py                    # Schema para treinamento
└── services/
    ├── yfinance_service.py         # Serviço de coleta de dados
    ├── preprocess_data_service.py  # Pré-processamento
    ├── train/
    │   ├── prepare_data_service.py # Preparação para treinamento
    │   ├── train_service.py        # Serviço de treinamento
    │   └── evaluate_service.py     # Avaliação de modelos
    ├── predict/
    │   ├── prepare_data_service.py # Preparação para predição
    │   └── predict_service.py      # Serviço de predição
    └── s3/
        ├── base_service.py         # Cliente S3 base
        ├── upload_service.py       # Upload de modelos
        └── download_service.py     # Download de modelos
```

## Modelo LSTM

### Características Técnicas
- **Framework**: PyTorch Lightning para facilitar treinamento
- **Early Stopping**: Previne overfitting
- **Batch First**: Configuração otimizada para dados sequenciais
- **Escalabilidade**: Suporte a diferentes tamanhos de sequência

### Preparação dos Dados
1. **Normalização**: MinMaxScaler para features numéricas
2. **Sequências**: Janelas deslizantes de tamanho configurável
3. **Divisão**: Train/Test split configurável (padrão 80/20)
4. **Target**: Preço de fechamento do próximo dia

### Métricas de Avaliação

O sistema calcula automaticamente as seguintes métricas:

- **MAE (Mean Absolute Error)**: Erro médio absoluto
- **MAPE (Mean Absolute Percentage Error)**: Erro percentual médio absoluto
- **RMSE (Root Mean Square Error)**: Raiz do erro quadrático médio
- **R² (Coefficient of Determination)**: Coeficiente de determinação

## Tecnologias Utilizadas

### Backend e API
- **FastAPI**: Framework web moderno e rápido
- **Mangum**: Adaptador ASGI para AWS Lambda
- **Pydantic**: Validação de dados e schemas
- **Uvicorn**: Servidor ASGI para desenvolvimento

### Machine Learning
- **PyTorch**: Framework de deep learning
- **PyTorch Lightning**: Simplificação do treinamento
- **scikit-learn**: Pré-processamento e métricas
- **NumPy**: Computação numérica

### Dados
- **yfinance**: Coleta de dados financeiros
- **pandas**: Manipulação de dados

### Cloud e Infrastructure
- **AWS Lambda**: Computação serverless
- **AWS S3**: Armazenamento de modelos
- **AWS ECR**: Registry de containers
- **Docker**: Containerização

### DevOps
- **python-dotenv**: Gerenciamento de variáveis de ambiente
- **boto3**: SDK da AWS para Python

## Considerações de Produção

### Performance
- **Cold Start**: Otimizado para AWS Lambda
- **Caching**: Modelos mantidos em memória durante execução
- **Timeout**: Configurado para 15 minutos para treinamento

### Escalabilidade
- **Serverless**: Escalabilidade automática do Lambda
- **Storage**: S3 para armazenamento durável de modelos
- **Stateless**: API sem estado para alta disponibilidade

### Limitações Conhecidas
- **Timeout Lambda**: Máximo 15 minutos por execução
- **Memory**: Limitado a 10GB no Lambda
- **Cold Start**: Latência inicial para inicialização
