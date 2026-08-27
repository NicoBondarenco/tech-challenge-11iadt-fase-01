from fastapi import APIRouter, Query

from src.service.model_service import MODEL_FILES, predict_with_model

router = APIRouter()


@router.get("/")
def read_root():
    return {"message": "Tech Challenge 11iADT - Fase 1 - API de Previsão de Câncer de Mama"}


@router.get("/models")
def list_models():
    return {"models": list(MODEL_FILES.keys())}


def _predict_feature_values(model_name: str, **feature_values):
    for suffix in ("mean", "se", "worst"):
        feature_name = f"concave_points_{suffix}"
        if feature_name in feature_values:
            feature_values[f"concave points_{suffix}"] = feature_values.pop(feature_name)
    return predict_with_model(model_name, **feature_values)


@router.get("/predict")
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
    feature_values = locals().copy()
    feature_values.pop("model_name")
    return _predict_feature_values(model_name, **feature_values)


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
        return _predict_feature_values(model_name, **locals())

    endpoint.__name__ = f"predict_{model_name}"
    return endpoint


for model_name in MODEL_FILES:
    router.add_api_route(f"/predict/{model_name}", _create_model_endpoint(model_name), methods=["GET"])
