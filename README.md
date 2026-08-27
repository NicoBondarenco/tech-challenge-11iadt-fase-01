# 🧬 Tech Challenge — Fase 1
## Análise e Classificação de Câncer de Mama

Projeto desenvolvido como parte do **Tech Challenge** da Pós-Graduação em IA para Devs da FIAP, turma 11IADT, com foco em **Análise Exploratória de Dados (EDA)**, **Machine Learning** e entrega de uma solução completa para predição de câncer de mama.

O projeto utiliza o dataset **Wisconsin Diagnostic Breast Cancer (WDBC)** para explorar, preparar e modelar dados relacionados à classificação de tumores como **benignos** ou **malignos**. A solução inclui notebook de análise, modelos treinados, uma API em **FastAPI** e um cliente em **Swift / SwiftUI**.

> ⚠️ **Aviso académico:** este projeto tem finalidade educacional e experimental. Os modelos desenvolvidos não substituem diagnóstico médico profissional.

---

## 🎯 Objetivo

O objetivo do projeto é desenvolver e comparar modelos de Machine Learning capazes de classificar amostras em duas categorias:

- **B — Benigno**
- **M — Maligno**

As etapas principais incluem:

- exploração dos dados;
- análise da qualidade dos dados;
- tratamento e limpeza;
- engenharia de atributos;
- pré-processamento;
- treinamento dos modelos;
- avaliação das métricas;
- comparação dos resultados;
- análise da matriz de confusão;
- experimentação com redução de dimensionalidade usando PCA.

---

## 📊 Dataset

O projeto utiliza o dataset **Wisconsin Diagnostic Breast Cancer (WDBC)**.

Arquivo utilizado:

```text
data/breast-cancer-wisconsin-data.csv
```

Fonte: [https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data)

Durante o pré-processamento, colunas como `id` e `Unnamed: 32` são removidas. A variável `diagnosis` é convertida para representação binária:

```text
B → 0
M → 1
```

---

## 🔎 Análise Exploratória

A análise contempla:

- dimensões do dataset;
- tipos das variáveis;
- estatísticas descritivas;
- valores ausentes;
- registros duplicados;
- distribuição das classes;
- análise das variáveis numéricas;
- visualizações e correlações.

A divisão dos dados foi feita com:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

Assim, 80% dos dados são usados para treino e 20% para teste, preservando a proporção das classes.

---

## 🧹 Pré-processamento

As principais etapas de preparação dos dados são:

1. remoção de colunas desnecessárias;
2. codificação da variável alvo;
3. separação entre `X` e `y`;
4. divisão entre treino e teste;
5. padronização das features;
6. experimentação com diferentes estratégias de escalonamento.

A solução compara principalmente:

- `StandardScaler`
- `RobustScaler`
- `PCA` após escalonamento

---

## 🤖 Modelos Avaliados

Foram avaliados três algoritmos de classificação:

### Logistic Regression
Modelo de classificação linear utilizado como baseline.

### K-Nearest Neighbors (KNN)
Algoritmo baseado na proximidade entre as observações.

### Random Forest
Modelo baseado em um conjunto de árvores de decisão.

Os modelos são implementados com `scikit-learn`.

---

## 📏 Métricas

Os modelos foram avaliados por:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Matriz de Confusão
- Classification Report

---

## 📈 Resultados

Os resultados do notebook indicam desempenho muito alto para os modelos avaliados, com destaque para o **Random Forest sem PCA**.

| Métrica | Resultado |
|---|---:|
| Accuracy | 97,37% |
| Precision | 100,00% |
| Recall | 92,86% |
| F1 Score | 96,30% |
| ROC-AUC | 99,29% |

Matriz de confusão do modelo principal:

|  | Predito B | Predito M |
|---|---:|---:|
| **Real B** | 72 | 0 |
| **Real M** | 3 | 39 |

Isso corresponde a:

- **TP:** 39
- **TN:** 72
- **FP:** 0
- **FN:** 3

---

## 🛠️ Tecnologias

O projeto foi desenvolvido em **Python** e inclui:

- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Uvicorn](https://www.uvicorn.org/)
- [Swift / SwiftUI](https://developer.apple.com/xcode/swiftui/)

---

## 📁 Estrutura do Projeto

```text
tech-challenge-11iadt-fase-01/
├── LICENSE
├── README.md
├── requirements.txt
├── app/
│   └── diagnostic/
│       └── diagnostic.xcodeproj/
├── data/
│   └── breast-cancer-wisconsin-data.csv
├── models/                       # modelos treinados (.pkl)
├── notebooks/
│   └── tech_challenge_fase1.ipynb
├── src/
│   ├── __init__.py
│   ├── data_prep.py
│   ├── main.py                   # API FastAPI
│   └── model.py
├── tests/
│   └── test_main.py
└── .venv/                       # ambiente virtual local (opcional)
```

---

## 🧰 Como configurar o ambiente

### 1. Pré-requisitos

- Python 3.10 ou superior
- `pip` instalado
- Git
- Opcional: Xcode 14+ para executar o app SwiftUI

### 2. Clone o repositório

```bash
git clone https://github.com/NicoBondarenco/tech-challenge-11iadt-fase-01.git
cd tech-challenge-11iadt-fase-01
```

### 3. Crie e ative um ambiente virtual

No macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

No Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

No Windows (CMD):

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

### 4. Instale as dependências

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Executando a API FastAPI

A API está localizada em `src/main.py` e utiliza modelos salvos na pasta `models/`.

### Opção 1: FastAPI em modo de desenvolvimento

```bash
fastapi dev src/main.py
```

### Opção 2: Uvicorn

```bash
uvicorn src.main:app --reload --port 8000
```

A aplicação ficará disponível em:

- http://127.0.0.1:8000

### Documentação interativa

- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

---

## 🔍 Endpoints principais

- `GET /` — mensagem de boas-vindas
- `GET /models` — lista os modelos disponíveis
- `GET /predict` — predição genérica, com `model_name` e as 30 features do dataset
- `GET /predict/{model_name}` — rota específica por modelo

Exemplos de modelos disponíveis:

- `random_forest`
- `random_forest_pca`
- `knn`
- `knn_pca`
- `logistic_regression`
- `logistic_regression_pca`

### Exemplo de chamada via curl

```bash
curl -X 'GET' \
  'http://127.0.0.1:8000/predict/random_forest?radius_mean=14.22&texture_mean=23.12&perimeter_mean=92.87&area_mean=620.5&smoothness_mean=0.1039&compactness_mean=0.1101&concavity_mean=0.07215&concave%20points_mean=0.04879&symmetry_mean=0.1794&fractal_dimension_mean=0.05964&radius_se=0.3456&texture_se=1.234&perimeter_se=2.456&area_se=28.5&smoothness_se=0.0065&compactness_se=0.025&concavity_se=0.035&concave%20points_se=0.012&symmetry_se=0.018&fractal_dimension_se=0.0035&radius_worst=15.89&texture_worst=30.45&perimeter_worst=105.2&area_worst=780.1&smoothness_worst=0.141&compactness_worst=0.28&concavity_worst=0.25&concave%20points_worst=0.11&symmetry_worst=0.31&fractal_dimension_worst=0.089' \
  -H 'accept: application/json'
```

---

## 🧪 Executando os testes

Para rodar a suíte de testes automatizados do projeto:

```bash
pytest
```

---

## 📓 Notebook

O notebook principal está em:

```text
notebooks/tech_challenge_fase1.ipynb
```

Ele reúne o processo completo de análise exploratória, processamento, treino, comparação de modelos e avaliação das métricas.

---

## 📱 Aplicativo SwiftUI (`diagnostic`)

O projeto também inclui um cliente em SwiftUI em:

```text
app/diagnostic/
```

### Como rodar o app

1. Abra o arquivo `app/diagnostic/diagnostic.xcodeproj` no Xcode.
2. Verifique se a `baseURL` em `NetworkManager.swift` aponta para a API correta (normalmente `http://127.0.0.1:8000`).
3. Execute o projeto no simulador ou em um dispositivo.

---

## ⚠️ Limitações e observações

- Os resultados refletem o dataset e a metodologia utilizada neste projeto.
- O objetivo é acadêmico e experimental.
- Os modelos não devem ser usados como diagnóstico clínico real.
- O app cliente depende da API estar em execução localmente para funcionar corretamente.

---

## 🖥️ Executando o Streamlit

A aplicação visual está localizada em `src/app.py` e utiliza o modelo treinado em `models/`.

Com o ambiente virtual ativado e as dependências instaladas, execute:

```bash
python -m streamlit run src/app.py
```

A aplicação ficará disponível em:

- http://localhost:8501

O arquivo do modelo esperado é:

```text
models/mobilenetv2_cbis_ddsm_best.keras
```

---

## 👥 Equipe

- [Adolfho Athyla](https://github.com/adolfhoathyla)
- [Andre Pasquetti](https://github.com/pasquettiandrepaulo)
- [Heloysa Arruda](https://github.com/heloysasa)
- [Kauanny Felix](https://github.com/KakauFelix)
- [Nicanor Bondarenco](https://github.com/NicoBondarenco)

---

## ✅ Resumo rápido de execução

```bash
git clone https://github.com/NicoBondarenco/tech-challenge-11iadt-fase-01.git
cd tech-challenge-11iadt-fase-01
python -m venv .venv
source .venv/bin/activate  # ou .\.venv\Scripts\Activate.ps1 no Windows
pip install -r requirements.txt
fastapi dev src/main.py
```

Em seguida, acesse:

- http://127.0.0.1:8000/docs
- http://127.0.0.1:8000/redoc


