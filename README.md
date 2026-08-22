# 🧬 Tech Challenge — Fase 1
## Análise e Classificação de Câncer de Mama

Projeto desenvolvido como parte do **Tech Challenge** da Pós-Graduação em IA para Devs da FIAP, com foco em **Análise Exploratória de Dados (EDA)** e **Machine Learning para classificação de câncer de mama**.

O projeto utiliza o dataset **Wisconsin Diagnostic Breast Cancer (WDBC)** para explorar, preparar e modelar dados relacionados à classificação de tumores como **benignos** ou **malignos**.

> ⚠️ **Aviso:** este projeto possui finalidade acadêmica e experimental. Os modelos desenvolvidos não constituem uma ferramenta de diagnóstico médico.

---

## 🎯 Objetivo

O objetivo deste projeto é desenvolver e comparar modelos de Machine Learning capazes de classificar amostras em duas categorias:

- **B — Benigno**
- **M — Maligno**

O notebook percorre as principais etapas de um projeto de Machine Learning, incluindo:

- exploração dos dados;
- análise da qualidade dos dados;
- tratamento e limpeza;
- engenharia de atributos;
- pré-processamento;
- treinamento dos modelos;
- avaliação das métricas;
- comparação dos resultados;
- análise da matriz de confusão;
- experimentação com redução de dimensionalidade utilizando PCA.


---

## 📊 Dataset

O projeto utiliza o dataset **Wisconsin Diagnostic Breast Cancer (WDBC)**.

Arquivo utilizado:

```text
breast-cancer-wisconsin-data.csv
```

Fonte: [https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data)

O notebook realiza o carregamento do dataset utilizando o Pandas e conduz a análise exploratória antes das etapas de modelagem.

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

Durante o pré-processamento, as colunas `id` e `Unnamed: 32` são removidas.

A variável `diagnosis` é convertida para representação binária:

```text
B → 0
M → 1
```

O notebook utiliza `LabelEncoder` para realizar essa transformação. :contentReference[oaicite:3]{index=3}

---

## 🧹 Pré-processamento

As principais etapas de preparação dos dados são:

1. Remoção de colunas desnecessárias;
2. Codificação da variável alvo;
3. Separação entre features (`X`) e variável alvo (`y`);
4. Divisão entre treino e teste;
5. Padronização das features;
6. Experimentação com diferentes estratégias de escalonamento.

A divisão dos dados utiliza:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

Assim, são utilizados 80% dos dados para treinamento e 20% para teste, preservando a proporção das classes.

---

## 🤖 Modelos

Foram avaliados três algoritmos de classificação:

### Logistic Regression

Modelo de classificação linear utilizado como baseline para comparação.

### K-Nearest Neighbors (KNN)

Algoritmo baseado na proximidade entre as observações.

### Random Forest

Modelo baseado em um conjunto de árvores de decisão.

Os três modelos são implementados utilizando o `scikit-learn`. 

---

## 📏 Métricas

Os modelos foram avaliados utilizando:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Matriz de Confusão
- Classification Report

As métricas são calculadas utilizando ferramentas do módulo `sklearn.metrics`.

---

## 🧪 StandardScaler, RobustScaler e PCA

O projeto também compara diferentes estratégias de preparação dos dados.

A abordagem principal utiliza `StandardScaler`.

Também foi testada uma abordagem utilizando:

```text
RobustScaler
      ↓
PCA
      ↓
Modelo de classificação
```

O objetivo é analisar o impacto do escalonamento robusto e da redução de dimensionalidade sobre o desempenho dos modelos.

---

## 📈 Resultados

Os resultados apresentados no notebook indicam um desempenho elevado dos modelos avaliados. Entre eles, o Random Forest sem PCA apresentou os melhores resultados e, por isso, foi selecionado como o modelo principal para dar continuidade às análises e aos desenvolvimentos.

O **Random Forest sem PCA** apresentou:

| Métrica | Resultado |
|---|---:|
| Accuracy | 97,37% |
| Precision | 100,00% |
| Recall | 92,86% |
| F1 Score | 96,30% |
| ROC-AUC | 99,29% |

A matriz de confusão do Random Forest apresentou:

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

O projeto foi desenvolvido em **Python** utilizando:

- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [Scikit-learn](https://scikit-learn.org/)

---

## 📁 Estrutura do Projeto

```text
tech-challenge-11iadt-fase-01/
│
├── notebooks/
│   └── tech_challenge_fase1.ipynb
├── datasets/
│   └── breast-cancer-wisconsin-data.csv
│
├── README.md
│
└── requirements.txt
...
```

---

## 🚀 Como executar

### 1. Clone o repositório

```bash
git clone https://github.com/NicoBondarenco/tech-challenge-11iadt-fase-01.git
```

### 2. Entre na pasta

```bash
cd tech-challenge-11iadt-fase-01
```

### 3. Crie um ambiente virtual

No Windows:

```bash
python -m venv .venv
```

### 4. Ative o ambiente virtual

PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Caso esteja utilizando o CMD:

```cmd
.venv\Scripts\activate
```

### 5. Instale as dependências

```bash
pip install -r requirements.txt
```

### 6. Execute o notebook

Abra:

```text
notebooks/tech_challenge_fase1.ipynb
```

O notebook pode ser executado utilizando Jupyter Notebook, JupyterLab ou Google Colab.

---

## 📓 Notebook

O notebook principal está disponível em:

```text
notebooks/tech_challenge_fase1.ipynb
```

Ele contém todo o processo de exploração, preparação dos dados, treinamento, avaliação e comparação dos modelos.

---

## ⚠️ Limitações

Os resultados apresentados são referentes ao dataset e à metodologia de divisão de treino e teste utilizada no projeto.

Este trabalho possui **finalidade acadêmica** e não representa um modelo clínico validado.

Os resultados não devem ser utilizados para diagnóstico ou tomada de decisão médica.

---

## 👥 Equipe

- [Adolfho Athyla](https://github.com/adolfhoathyla)
- [Andre Pasquetti](https://github.com/pasquettiandrepaulo)
- [Heloysa Arruda](https://github.com/heloysasa)
- [Kauanny Felix](https://github.com/KakauFelix)
- [Nicanor Bondarenco](https://github.com/NicoBondarenco)
