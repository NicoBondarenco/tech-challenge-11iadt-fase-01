# fastapi dev src/main.py

import pickle
import pandas as pd
from fastapi import FastAPI, Query

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Tech Challenge 11iADT - Fase 1 - API de Previsão de Câncer de Mama"}


@app.get("/predict")
def predict(
    radius_mean: float = Query(..., description="Média do raio"),
    texture_mean: float = Query(..., description="Média da textura"),
    perimeter_mean: float = Query(..., description="Média do perímetro"),
    area_mean: float = Query(..., description="Média da área"),
    smoothness_mean: float = Query(..., description="Média da suavidade"),
    compactness_mean: float = Query(..., description="Média da compacidade"),
    concavity_mean: float = Query(..., description="Média da concavidade"),
    concave_points_mean: float = Query(..., alias="concave points_mean", description="Média dos pontos côncavos"),
    symmetry_mean: float = Query(..., description="Média da simetria"),
    fractal_dimension_mean: float = Query(..., description="Média da dimensão fractal"),
    radius_se: float = Query(..., description="Erro padrão do raio"),
    texture_se: float = Query(..., description="Erro padrão da textura"),
    perimeter_se: float = Query(..., description="Erro padrão do perímetro"),
    area_se: float = Query(..., description="Erro padrão da área"),
    smoothness_se: float = Query(..., description="Erro padrão da suavidade"),
    compactness_se: float = Query(..., description="Erro padrão da compacidade"),
    concavity_se: float = Query(..., description="Erro padrão da concavidade"),
    concave_points_se: float = Query(..., alias="concave points_se", description="Erro padrão dos pontos côncavos"),
    symmetry_se: float = Query(..., description="Erro padrão da simetria"),
    fractal_dimension_se: float = Query(..., description="Erro padrão da dimensão fractal"),
    radius_worst: float = Query(..., description="Pior raio"),
    texture_worst: float = Query(..., description="Pior textura"),
    perimeter_worst: float = Query(..., description="Pior perímetro"),
    area_worst: float = Query(..., description="Pior área"),
    smoothness_worst: float = Query(..., description="Pior suavidade"),
    compactness_worst: float = Query(..., description="Pior compacidade"),
    concavity_worst: float = Query(..., description="Pior concavidade"),
    concave_points_worst: float = Query(..., alias="concave points_worst", description="Pior pontos côncavos"),
    symmetry_worst: float = Query(..., description="Pior simetria"),
    fractal_dimension_worst: float = Query(..., description="Pior dimensão fractal"),
):
    with open("models/model_random_forest.pkl", "rb") as model_file:
        model = pickle.load(model_file)

    feature_columns = [
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

    values = [
        radius_mean,
        texture_mean,
        perimeter_mean,
        area_mean,
        smoothness_mean,
        compactness_mean,
        concavity_mean,
        concave_points_mean,
        symmetry_mean,
        fractal_dimension_mean,
        radius_se,
        texture_se,
        perimeter_se,
        area_se,
        smoothness_se,
        compactness_se,
        concavity_se,
        concave_points_se,
        symmetry_se,
        fractal_dimension_se,
        radius_worst,
        texture_worst,
        perimeter_worst,
        area_worst,
        smoothness_worst,
        compactness_worst,
        concavity_worst,
        concave_points_worst,
        symmetry_worst,
        fractal_dimension_worst,
    ]

    features = pd.DataFrame([values], columns=feature_columns)
    prediction = model.predict(features)
    probability = model.predict_proba(features)[:, 1]

    diagnostico = 'Maligno' if prediction[0] == 1 else 'Benigno'
    proba = probability[0] * 100


    return {
        "prediction": diagnostico,
        "probability": proba,
    }