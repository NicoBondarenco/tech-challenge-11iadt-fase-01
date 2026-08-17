# tech-challenge-11iadt-fase-01
Pós IA para Devs FIAP - 11IADT - Tech Challenge - Fase 01

## EXTRA - Visão Computacional

Implementação de classificação de alterações mamográficas recortadas do CBIS-DDSM
com pré-processamento, transfer learning via MobileNetV2, avaliação e inferência.

### Inspeção do Repositório e Dataset

- Código existente: `src/data_prep.py`, `src/model.py` e `src/__init__.py` estavam vazios; o notebook principal está em `notebooks/tech_challenge_fase1.ipynb`.
- CSV tabular original: `data/breast-cancer-wisconsin-data.csv`.
- CSVs CBIS-DDSM: `data/mass_case_description_train_set.csv`, `data/mass_case_description_test_set.csv`, `data/calc_case_description_train_set.csv`, `data/calc_case_description_test_set.csv`, `data/dicom_info.csv` e `data/meta.csv`.
- Imagens: `data/img`, com 10.237 arquivos `.jpg`; não foram encontrados `.png`, `.dcm`, `.jpeg`, `.tif` ou `.tiff` no repositório.
- `dicom_info.csv` identifica 2.857 `full mammogram images`, 3.567 `cropped images`, 3.247 `ROI mask images` e 566 linhas sem `SeriesDescription`.
- Os nomes dos JPGs foram achatados em `data/img` com sufixos como `_1`, `_2`; por isso a associação física não usa apenas basename, mas basename original + dimensões registradas no `dicom_info.csv`.
- Casos oficiais: 3.568 linhas, 1.566 pacientes, 2.864 casos no train oficial e 704 no test oficial.
- Distribuição de diagnósticos nos casos: 1.457 `MALIGNANT`, 1.429 `BENIGN` e 682 `BENIGN_WITHOUT_CALLBACK`.
- Distribuição por anormalidade: 1.696 `mass` e 1.872 `calcification`.
- Há pacientes presentes tanto no train oficial quanto no test oficial. Para evitar data leakage, o pipeline preserva o test oficial e marca as linhas conflitantes do train como `excluded_official_overlap`.

### Arquitetura Implementada

- `src/dataset.py`: inventário, associação CSVs + imagens, criação de `data/cv_master_dataset.csv`, split por paciente e validação de leakage.
- `src/preprocessing.py`: leitura grayscale, conversão para 3 canais, resize com padding e normalização MobileNetV2.
- `src/model.py`: MobileNetV2 congelada com saída `Dense(1, activation="sigmoid")`.
- `src/inference.py`: serviço compartilhado de inferência para CLI, futura API e Streamlit.
- `src/gradcam.py`: utilitários reutilizáveis para Grad-CAM.
- `src/train.py`: treinamento com `EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau` e pesos de classe.
- `src/evaluate.py`: métricas, matriz de confusão, classification report e CSV com score por imagem.
- `src/predict.py`: CLI fina para inferência de uma alteração recortada.
- `src/app.py`: interface Streamlit para upload, predição, Grad-CAM, métricas e explicação do pipeline.

### Preparação do Dataset

Use um Python estável com suporte a TensorFlow, preferencialmente 3.11 ou 3.12.
No Windows/Git Bash, este projeto foi ajustado para rodar com o Python 3.11
instalado na máquina:

```bash
py -3.11 -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Execução Completa no Git Bash

Para uma pessoa rodar o projeto do zero no Git Bash:

```bash
cd ~/techChallenge/tech-challenge-11iadt-fase-01
py -3.11 -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m src.data_prep --inventory --image-type cropped --output data/cv_master_dataset.csv
python -m src.train --dataset data/cv_master_dataset.csv --epochs 5 --fine-tune-epochs 2 --batch-size 8
python -m src.evaluate --dataset data/cv_master_dataset.csv --model models/mobilenetv2_cbis_ddsm_best.keras --split test
streamlit run src/app.py
```

Se a `.venv`, o dataset mestre, o modelo e os relatórios já existirem, para reabrir
a demonstração normalmente basta:

```bash
cd ~/techChallenge/tech-challenge-11iadt-fase-01
source .venv/Scripts/activate
streamlit run src/app.py
```

```bash
python -m src.data_prep --inventory --image-type cropped --output data/cv_master_dataset.csv
```

O dataset mestre usa `cropped images` por padrão, porque elas mapeiam caso-a-caso
com os diagnósticos. Portanto, o modelo atual classifica uma alteração mamográfica
recortada como benigna ou maligna. Ele não classifica uma mamografia completa como
normal/anormal. ROI masks são identificadas, mas não entram como imagens de
classificação.

Resultado da preparação atual:

- 3.565 imagens cropped associadas a metadados.
- Split final: 2.191 treino, 588 validação, 704 teste e 82 excluídas por sobreposição oficial.
- Pacientes por split: 974 treino, 243 validação e 349 teste.
- Vazamento entre treino/validação/teste: `false`.

### Treinamento

Smoke test rápido:

```bash
python -m src.train --dataset data/cv_master_dataset.csv --epochs 1 --batch-size 8
```

Treinamento mais adequado para avaliação do projeto:

```bash
python -m src.train --dataset data/cv_master_dataset.csv --epochs 15 --fine-tune-epochs 5 --batch-size 8
```

O melhor modelo global é salvo em `models/mobilenetv2_cbis_ddsm_best.keras`. O
treino monitora `val_roc_auc` por padrão, mantendo também precision, recall e accuracy.
Se houver fine-tuning, a Fase 2 salva temporariamente em
`models/mobilenetv2_cbis_ddsm_fine_tune_best.keras` e só substitui o melhor global
se superar a métrica da Fase 1.
Uma época é apenas um teste de execução; resultados como ROC-AUC próximo de 0,54
não devem ser tratados como conclusão do modelo.

### Avaliação

```bash
python -m src.evaluate --dataset data/cv_master_dataset.csv --model models/mobilenetv2_cbis_ddsm_best.keras --split test
```

Artefatos gerados em `reports/`:

- `metrics_test.json`
- `predictions_test.csv`
- `confusion_matrix_test.png`

O CSV de predições contém imagem, classe real, classe prevista, score para
malignidade, indicador de acerto e tipo de erro (`false_positive`,
`false_negative` ou `correct`).

### Predição de Nova Alteração Recortada

```bash
python -m src.predict data/img/alguma_imagem.jpg --model models/mobilenetv2_cbis_ddsm_best.keras
```

A saída informa se a alteração recortada foi classificada como `BENIGNO` ou
`MALIGNO` e o score produzido pelo classificador para a classe maligna. Esse
score não deve ser interpretado como chance clínica da paciente ter câncer.

### Interface Streamlit

```bash
streamlit run src/app.py
```

A interface permite upload de uma alteração mamográfica recortada, exibe a imagem,
carrega o modelo `.keras`, mostra a classificação `BENIGNO`/`MALIGNO`, o score
para malignidade e um Grad-CAM sobreposto. O Grad-CAM é uma visualização de
influência da CNN e não representa localização clínica de tumor.

A segunda aba mostra métricas geradas por `src.evaluate`, matriz de confusão,
curvas de treinamento e uma explicação visual do pipeline.

### Separação de Responsabilidades

Fluxo arquitetural planejado:

```text
dataset -> preprocessing -> modelo -> inference -> API -> Streamlit
```

- Machine Learning: `src/dataset.py`, `src/preprocessing.py`, `src/model.py`, `src/train.py`, `src/evaluate.py` e `src/gradcam.py`.
- Serviço de inferência/API: `src/inference.py` concentra carregamento do modelo e predição; uma futura API deve chamar esse serviço.
- Interface: `src/app.py` é cliente/apresentação Streamlit. Ela não deve ser a fonte final da regra de negócio.
