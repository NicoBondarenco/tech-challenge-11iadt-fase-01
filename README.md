# Tech Challenge - Fase 1 | Pós-graduação IA para Desenvolvedores (FIAP - 11IADT)

Projeto desenvolvido como parte do Tech Challenge da Fase 1 da Pós-graduação em Inteligência Artificial para Desenvolvedores da FIAP (Turma 11IADT). O objetivo do projeto é construir uma aplicação completa de Machine Learning para a predição de Câncer de Mama (Base de Dados Wisconsin Breast Cancer), incluindo preparação de dados, treinamento de modelos (com e sem PCA), um notebook exploratório, uma API robusta em **FastAPI** e um aplicativo cliente em **Swift / SwiftUI** (`diagnostic`).

---

## 📁 Estrutura do Projeto

```text
├── LICENSE
├── README.md
├── requirements.txt
├── app/
│   └── diagnostic/         # Aplicativo SwiftUI (iOS/macOS)
├── data/
│   └── breast-cancer-wisconsin-data.csv
├── models/                 # Modelos treinados (.pkl)
├── notebooks/
│   └── tech_challenge_fase1.ipynb
├── src/
│   ├── __init__.py
│   ├── data_prep.py
│   ├── main.py             # API FastAPI
│   └── model.py
└── tests/
    └── test_main.py
```

---

## 🛠️ Como Montar o Ambiente e Executar a API

### 1. Pré-requisitos
- Python 3.10 ou superior instalado.
- Gerenciador de pacotes `pip`.
- (Opcional) Xcode 14+ para rodar o aplicativo SwiftUI.

### 2. Configuração do Ambiente Virtual

Abra o terminal na raiz do projeto e crie/ative um ambiente virtual:

```bash
# Criar o ambiente virtual
python -m venv venv

# Ativar o ambiente virtual
# No macOS/Linux:
source venv/bin/activate

# No Windows (PowerShell):
# .\venv\Scripts\Activate
```

### 3. Instalação das Dependências

Instale as bibliotecas necessárias listadas em `requirements.txt`:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Executando a API FastAPI

A API foi desenvolvida utilizando **FastAPI** e utiliza modelos de Machine Learning treinados salvos na pasta `models/`.

Para iniciar o servidor de desenvolvimento:

```bash
fastapi dev src/main.py
```

Ou utilizando o uvicorn diretamente:

```bash
uvicorn src/main.py:app --reload --port 8000
```

A API estará rodando em: **`http://127.0.0.1:8000`**

### 📄 Documentação Interativa
Você pode acessar a documentação interativa gerada automaticamente pelo FastAPI:
- **Swagger UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 🔍 Endpoints Principais

- `GET /`: Retorna a mensagem de boas-vindas da API.
- `GET /models`: Lista os modelos de Machine Learning disponíveis.
- `GET /predict`: Endpoint genérico para predição informando o parâmetro `model_name` e as 30 features do dataset de câncer de mama.
- `GET /predict/{model_name}`: Endpoint específico para cada modelo treinado (ex: `/predict/random_forest`, `/predict/knn`, `/predict/svm_pca`, etc.).

Exemplo de chamada via `curl`:
```bash
curl -X 'GET' \
  'http://127.0.0.1:8000/predict/random_forest?radius_mean=14.22&texture_mean=23.12&perimeter_mean=92.87&area_mean=620.5&smoothness_mean=0.1039&compactness_mean=0.1101&concavity_mean=0.07215&concave%20points_mean=0.04879&symmetry_mean=0.1794&fractal_dimension_mean=0.05964&radius_se=0.3456&texture_se=1.234&perimeter_se=2.456&area_se=28.5&smoothness_se=0.0065&compactness_se=0.025&concavity_se=0.035&concave%20points_se=0.012&symmetry_se=0.018&fractal_dimension_se=0.0035&radius_worst=15.89&texture_worst=30.45&perimeter_worst=105.2&area_worst=780.1&smoothness_worst=0.141&compactness_worst=0.28&concavity_worst=0.25&concave%20points_worst=0.11&symmetry_worst=0.31&fractal_dimension_worst=0.089' \
  -H 'accept: application/json'
```

---

## 🧪 Executando os Testes

Para rodar os testes automatizados da aplicação:

```bash
pytest
```

---

## 📱 Aplicativo iOS / SwiftUI (`diagnostic`)

O projeto inclui um aplicativo SwiftUI na pasta `app/diagnostic`. 
1. Abra o arquivo `app/diagnostic/diagnostic.xcodeproj` no **Xcode**.
2. Certifique-se de que a `baseURL` no arquivo `NetworkManager.swift` aponta para o endereço correto da API (por padrão `http://127.0.0.1:8000`).
3. Execute o projeto no simulador ou em seu dispositivo.
