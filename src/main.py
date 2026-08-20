# fastapi dev src/main.py

import pickle
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

app = FastAPI()

MODEL_FILES = {
    "logistic_regression": "model_logistic_regression.pkl",
    "logistic_regression_pca": "model_logistic_regression_pca.pkl",
    "random_forest": "model_random_forest.pkl",
    "random_forest_pca": "model_random_forest_pca.pkl",
    "knn": "model_knn.pkl",
    "knn_pca": "model_knn_pca.pkl"
}

FEATURE_COLUMNS = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave points_se",
    "symmetry_se",
    "fractal_dimension_se",
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave points_worst",
    "symmetry_worst",
    "fractal_dimension_worst",
]


def get_model_path(model_name: str) -> Path:
    if model_name not in MODEL_FILES:
        valid_models = ", ".join(MODEL_FILES.keys())
        raise HTTPException(
            status_code=404,
            detail=f"Modelo '{model_name}' não encontrado. Modelos disponíveis: {valid_models}",
        )

    base_dir = Path(__file__).resolve().parent.parent
    return base_dir / "models" / MODEL_FILES[model_name]


def build_feature_frame(**kwargs) -> pd.DataFrame:
    values = [kwargs[column] for column in FEATURE_COLUMNS]
    return pd.DataFrame([values], columns=FEATURE_COLUMNS)


def get_pca_preprocessor() -> tuple[RobustScaler, PCA]:
    dataset_path = Path(__file__).resolve().parent.parent / "data" / "breast-cancer-wisconsin-data.csv"
    dataset = pd.read_csv(dataset_path)
    feature_frame = dataset.drop(columns=["id", "diagnosis", "Unnamed: 32"], errors="ignore")[FEATURE_COLUMNS]

    scaler = RobustScaler()
    scaled = scaler.fit_transform(feature_frame)

    pca = PCA(n_components=0.95, random_state=42)
    pca.fit(scaled)

    return scaler, pca


def prepare_features_for_model(model_name: str, features: pd.DataFrame) -> pd.DataFrame:
    if not model_name.endswith("_pca"):
        return features

    expected_features = getattr(__import__("pickle").loads, "__call__", None)
    if expected_features is None:
        return features

    with get_model_path(model_name).open("rb") as model_file:
        model = pickle.load(model_file)

    if getattr(model, "n_features_in_", None) != 10:
        return features

    if features.shape[1] == 10:
        return features

    scaler, pca = get_pca_preprocessor()
    scaled = scaler.transform(features[FEATURE_COLUMNS])
    components = pca.transform(scaled)
    component_names = [f"PC{i + 1}" for i in range(components.shape[1])]
    return pd.DataFrame(components, columns=component_names)


def predict_with_model(model_name: str, **kwargs) -> dict:
    with get_model_path(model_name).open("rb") as model_file:
        model = pickle.load(model_file)

    features = build_feature_frame(**kwargs)
    features = prepare_features_for_model(model_name, features)
    prediction = model.predict(features)

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(features)[:, 1][0] * 100
    else:
        probability = None

    diagnostico = "Maligno" if int(prediction[0]) == 1 else "Benigno"

    return {
        "model": model_name,
        "prediction": diagnostico,
        "probability": probability,
    }


@app.get("/")
def read_root():
    return {"message": "Tech Challenge 11iADT - Fase 1 - API de Previsão de Câncer de Mama"}


@app.get("/models")
def list_models():
    return {"models": list(MODEL_FILES.keys())}


@app.get("/predict")
def predict(
    model_name: str = Query("random_forest", description="Nome do modelo a ser usado."),
    radius_mean: float = Query(..., description="Média do raio"),
    texture_mean: float = Query(..., description="Média da textura"),
    perimeter_mean: float = Query(..., description="Média do perímetro"),
    area_mean: float = Query(..., description="Média da área"),
    smoothness_mean: float = Query(..., description="Média da suavidade"),
    compactness_mean: float = Query(..., description="Média da compacidade"),
    concavity_mean: float = Query(..., description="Média da concavidade"),
    concave_points_mean: float = Query(..., description="Média dos pontos côncavos"),
    symmetry_mean: float = Query(..., description="Média da simetria"),
    fractal_dimension_mean: float = Query(..., description="Média da dimensão fractal"),
    radius_se: float = Query(..., description="Erro padrão do raio"),
    texture_se: float = Query(..., description="Erro padrão da textura"),
    perimeter_se: float = Query(..., description="Erro padrão do perímetro"),
    area_se: float = Query(..., description="Erro padrão da área"),
    smoothness_se: float = Query(..., description="Erro padrão da suavidade"),
    compactness_se: float = Query(..., description="Erro padrão da compacidade"),
    concavity_se: float = Query(..., description="Erro padrão da concavidade"),
    concave_points_se: float = Query(..., description="Erro padrão dos pontos côncavos"),
    symmetry_se: float = Query(..., description="Erro padrão da simetria"),
    fractal_dimension_se: float = Query(..., description="Erro padrão da dimensão fractal"),
    radius_worst: float = Query(..., description="Pior raio"),
    texture_worst: float = Query(..., description="Pior textura"),
    perimeter_worst: float = Query(..., description="Pior perímetro"),
    area_worst: float = Query(..., description="Pior área"),
    smoothness_worst: float = Query(..., description="Pior suavidade"),
    compactness_worst: float = Query(..., description="Pior compacidade"),
    concavity_worst: float = Query(..., description="Pior concavidade"),
    concave_points_worst: float = Query(..., description="Pior pontos côncavos"),
    symmetry_worst: float = Query(..., description="Pior simetria"),
    fractal_dimension_worst: float = Query(..., description="Pior dimensão fractal"),
):
    feature_values = {
        "radius_mean": radius_mean,
        "texture_mean": texture_mean,
        "perimeter_mean": perimeter_mean,
        "area_mean": area_mean,
        "smoothness_mean": smoothness_mean,
        "compactness_mean": compactness_mean,
        "concavity_mean": concavity_mean,
        "concave points_mean": concave_points_mean,
        "symmetry_mean": symmetry_mean,
        "fractal_dimension_mean": fractal_dimension_mean,
        "radius_se": radius_se,
        "texture_se": texture_se,
        "perimeter_se": perimeter_se,
        "area_se": area_se,
        "smoothness_se": smoothness_se,
        "compactness_se": compactness_se,
        "concavity_se": concavity_se,
        "concave points_se": concave_points_se,
        "symmetry_se": symmetry_se,
        "fractal_dimension_se": fractal_dimension_se,
        "radius_worst": radius_worst,
        "texture_worst": texture_worst,
        "perimeter_worst": perimeter_worst,
        "area_worst": area_worst,
        "smoothness_worst": smoothness_worst,
        "compactness_worst": compactness_worst,
        "concavity_worst": concavity_worst,
        "concave points_worst": concave_points_worst,
        "symmetry_worst": symmetry_worst,
        "fractal_dimension_worst": fractal_dimension_worst,
    }
    return predict_with_model(model_name, **feature_values)


def _create_model_endpoint(model_name: str):
    def endpoint(
        radius_mean: float = Query(..., description="Média do raio"),
        texture_mean: float = Query(..., description="Média da textura"),
        perimeter_mean: float = Query(..., description="Média do perímetro"),
        area_mean: float = Query(..., description="Média da área"),
        smoothness_mean: float = Query(..., description="Média da suavidade"),
        compactness_mean: float = Query(..., description="Média da compacidade"),
        concavity_mean: float = Query(..., description="Média da concavidade"),
        concave_points_mean: float = Query(..., description="Média dos pontos côncavos"),
        symmetry_mean: float = Query(..., description="Média da simetria"),
        fractal_dimension_mean: float = Query(..., description="Média da dimensão fractal"),
        radius_se: float = Query(..., description="Erro padrão do raio"),
        texture_se: float = Query(..., description="Erro padrão da textura"),
        perimeter_se: float = Query(..., description="Erro padrão do perímetro"),
        area_se: float = Query(..., description="Erro padrão da área"),
        smoothness_se: float = Query(..., description="Erro padrão da suavidade"),
        compactness_se: float = Query(..., description="Erro padrão da compacidade"),
        concavity_se: float = Query(..., description="Erro padrão da concavidade"),
        concave_points_se: float = Query(..., description="Erro padrão dos pontos côncavos"),
        symmetry_se: float = Query(..., description="Erro padrão da simetria"),
        fractal_dimension_se: float = Query(..., description="Erro padrão da dimensão fractal"),
        radius_worst: float = Query(..., description="Pior raio"),
        texture_worst: float = Query(..., description="Pior textura"),
        perimeter_worst: float = Query(..., description="Pior perímetro"),
        area_worst: float = Query(..., description="Pior área"),
        smoothness_worst: float = Query(..., description="Pior suavidade"),
        compactness_worst: float = Query(..., description="Pior compacidade"),
        concavity_worst: float = Query(..., description="Pior concavidade"),
        concave_points_worst: float = Query(..., description="Pior pontos côncavos"),
        symmetry_worst: float = Query(..., description="Pior simetria"),
        fractal_dimension_worst: float = Query(..., description="Pior dimensão fractal"),
    ):
        feature_values = {
            "radius_mean": radius_mean,
            "texture_mean": texture_mean,
            "perimeter_mean": perimeter_mean,
            "area_mean": area_mean,
            "smoothness_mean": smoothness_mean,
            "compactness_mean": compactness_mean,
            "concavity_mean": concavity_mean,
            "concave points_mean": concave_points_mean,
            "symmetry_mean": symmetry_mean,
            "fractal_dimension_mean": fractal_dimension_mean,
            "radius_se": radius_se,
            "texture_se": texture_se,
            "perimeter_se": perimeter_se,
            "area_se": area_se,
            "smoothness_se": smoothness_se,
            "compactness_se": compactness_se,
            "concavity_se": concavity_se,
            "concave points_se": concave_points_se,
            "symmetry_se": symmetry_se,
            "fractal_dimension_se": fractal_dimension_se,
            "radius_worst": radius_worst,
            "texture_worst": texture_worst,
            "perimeter_worst": perimeter_worst,
            "area_worst": area_worst,
            "smoothness_worst": smoothness_worst,
            "compactness_worst": compactness_worst,
            "concavity_worst": concavity_worst,
            "concave points_worst": concave_points_worst,
            "symmetry_worst": symmetry_worst,
            "fractal_dimension_worst": fractal_dimension_worst,
        }
        return predict_with_model(model_name, **feature_values)

    endpoint.__name__ = f"predict_{model_name}"
    return endpoint


for model_name in MODEL_FILES:
    app.add_api_route(f"/predict/{model_name}", _create_model_endpoint(model_name), methods=["GET"])