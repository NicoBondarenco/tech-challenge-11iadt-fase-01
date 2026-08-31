
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

try:
    from .images.config import DEFAULT_CLASSIFICATION_THRESHOLD
    from .images.dataset import ROOT
    from .images.gradcam import compute_grad_cam, overlay_heatmap
    from .images.inference import load_model, predict_with_model
except ImportError:
    from images.config import DEFAULT_CLASSIFICATION_THRESHOLD
    from images.dataset import ROOT
    from images.gradcam import compute_grad_cam, overlay_heatmap
    from images.inference import load_model, predict_with_model


DEFAULT_MODEL_PATH = ROOT / "models" / "mobilenetv2_cbis_ddsm_best.keras"
DEFAULT_HISTORY_PATH = ROOT / "models" / "training_history.json"
DEFAULT_REPORTS_DIR = ROOT / "reports"


@st.cache_resource
def cached_model(model_path: str):
    return load_model(model_path)


@st.cache_data
def read_json(path: str) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def save_uploaded_file(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(uploaded_file.getbuffer())
        return Path(handle.name)


def show_classification_tab() -> None:
    st.header("Classificacao de Alteracoes Mamograficas")
    st.write(
        "A entrada esperada e uma regiao mamografica recortada, associada a uma alteracao ja identificada no dataset."
    )
    uploaded_file = st.file_uploader("Upload da alteracao recortada em JPG/JPEG", type=["jpg", "jpeg"])

    if not uploaded_file:
        st.info("Envie uma imagem recortada para visualizar a predicao e o Grad-CAM.")
        return

    model_path = DEFAULT_MODEL_PATH
    if not model_path.exists():
        st.error("Modelo treinado ainda nao encontrado. Execute o treinamento antes de usar a demonstracao.")
        return

    image_path = save_uploaded_file(uploaded_file)
    model = cached_model(str(model_path))
    result = predict_with_model(model, str(image_path), threshold=DEFAULT_CLASSIFICATION_THRESHOLD)
    color = "#9b1c1c" if result.label == "MALIGNO" else "#166534"
    st.markdown(
        f"""
        <div style="padding: 1rem; border-radius: 0.5rem; border: 1px solid #ddd;">
          <h2 style="margin: 0; color: {color};">Alteracao classificada como: {result.label}</h2>
          <p style="font-size: 1.25rem; margin-bottom: 0;">
            Score para classe maligna produzido pelo classificador:
            <b>{result.malignant_score * 100:.1f}%</b>
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "O Grad-CAM destaca regioes do recorte que mais influenciaram a CNN. "
        "Este mapa nao representa localizacao clinica do tumor e nao substitui avaliacao medica."
    )
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Imagem analisada")
        st.image(str(image_path), caption="Alteracao recortada enviada", width=380)
    with col2:
        st.subheader("Grad-CAM")
        try:
            heatmap = compute_grad_cam(model, str(image_path))
            st.image(overlay_heatmap(str(image_path), heatmap), caption="Grad-CAM sobreposto", width=380)
        except Exception as exc:
            st.warning(f"Nao foi possivel gerar Grad-CAM para esta imagem: {exc}")

    st.info("Aplicacao com finalidade academica e demonstrativa. Nao utilizar para diagnostico medico.")


def show_history_figure(title: str, keys: list[str], history: dict[str, list[float]]) -> None:
    available = [key for key in keys if key in history]
    if not available:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for key in available:
        ax.plot(history[key], label=key)
    ax.set_title(title)
    ax.set_xlabel("Epoca")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)


def render_history_grid(metric_groups: list[tuple[str, list[str]]], history: dict[str, list[float]]) -> None:
    for start in range(0, len(metric_groups), 2):
        left, right = st.columns(2)
        with left:
            title, keys = metric_groups[start]
            show_history_figure(title, keys, history)
        if start + 1 < len(metric_groups):
            with right:
                title, keys = metric_groups[start + 1]
                show_history_figure(title, keys, history)


def plot_training_history(history: dict[str, list[float]]) -> None:
    metric_groups = [
        ("Loss", ["loss", "val_loss"]),
        ("ROC-AUC", ["roc_auc", "val_roc_auc"]),
        ("Recall", ["recall", "val_recall"]),
        ("Precision", ["precision", "val_precision"]),
    ]
    max_epochs = max((len(values) for values in history.values() if isinstance(values, list)), default=0)
    if max_epochs < 2:
        st.info("As curvas de treinamento serao exibidas apos um treinamento com multiplas epocas.")
        return

    render_history_grid(metric_groups, history)


def show_metrics_tab() -> None:
    st.header("Metricas e funcionamento")
    split = st.selectbox("Split avaliado", ["validation", "test"], index=1)
    metrics_path = DEFAULT_REPORTS_DIR / f"metrics_{split}.json"
    confusion_matrix_path = DEFAULT_REPORTS_DIR / f"confusion_matrix_{split}.png"

    if metrics_path.exists():
        metrics = read_json(str(metrics_path))
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
        col2.metric("Precision", f"{metrics.get('precision', 0):.3f}")
        col3.metric("Recall maligno", f"{metrics.get('recall_sensitivity_malignant', 0):.3f}")
        col4.metric("F1", f"{metrics.get('f1_score', 0):.3f}")
        roc_auc = metrics.get("roc_auc")
        col5.metric("ROC-AUC", "N/A" if roc_auc is None else f"{roc_auc:.3f}")
        with st.expander("Detalhes tecnicos do Classification Report"):
            st.json(metrics.get("classification_report", {}))
    else:
        st.info(f"Metricas ainda nao encontradas em `{metrics_path}`. Execute `python -m src.images.evaluate --split {split}`.")

    if confusion_matrix_path.exists():
        left, center, right = st.columns([1, 2, 1])
        with center:
            st.image(str(confusion_matrix_path), caption=f"Matriz de confusao - {split}", width=360)
    else:
        st.info(f"Matriz de confusao ainda nao encontrada em `{confusion_matrix_path}`.")

    st.subheader("Curvas de Treinamento")
    if DEFAULT_HISTORY_PATH.exists():
        plot_training_history(read_json(str(DEFAULT_HISTORY_PATH)))
    else:
        st.info(f"Historico ainda nao encontrado em `{DEFAULT_HISTORY_PATH}`.")

    st.subheader("Funcionamento do pipeline")
    st.markdown(
        """
        **Imagem recortada -> Pre-processamento -> CNN MobileNetV2 -> Score do modelo -> BENIGNO/MALIGNO**

        A imagem e preparada para o formato esperado pela rede neural. A MobileNetV2 analisa os padroes visuais
        e gera um score para a classe maligna. A partir desse score, a alteracao e classificada como benigna ou maligna.

        O recall da classe maligna e uma metrica importante, pois um falso negativo ocorre quando uma alteracao
        realmente maligna e classificada como benigna.
        """
    )


def main() -> None:
    st.set_page_config(page_title="Classificador de Alteracoes Mamograficas", layout="wide")
    st.title("Classificador de Alteracoes Mamograficas Recortadas - CBIS-DDSM")
    st.caption("Demonstracao educacional de visao computacional. Nao utilizar como ferramenta diagnostica.")

    classification_tab, metrics_tab = st.tabs(["Classificacao", "Metricas e funcionamento"])
    with classification_tab:
        show_classification_tab()
    with metrics_tab:
        show_metrics_tab()


if __name__ == "__main__":
    main()
