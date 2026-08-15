from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_root_route():
    response = client.get("/")
    assert response.status_code == 200


def test_available_models_route():
    response = client.get("/models")
    assert response.status_code == 200
    payload = response.json()
    assert "models" in payload
    assert "random_forest" in payload["models"]
    assert "logistic_regression" in payload["models"]
    assert "knn" in payload["models"]


def test_predict_random_forest_route():
    params = {
        "radius_mean": 12.0,
        "texture_mean": 10.0,
        "perimeter_mean": 80.0,
        "area_mean": 500.0,
        "smoothness_mean": 0.08,
        "compactness_mean": 0.08,
        "concavity_mean": 0.05,
        "concave_points_mean": 0.04,
        "symmetry_mean": 0.2,
        "fractal_dimension_mean": 0.06,
        "radius_se": 1.0,
        "texture_se": 1.5,
        "perimeter_se": 6.0,
        "area_se": 30.0,
        "smoothness_se": 0.01,
        "compactness_se": 0.02,
        "concavity_se": 0.03,
        "concave_points_se": 0.012,
        "symmetry_se": 0.02,
        "fractal_dimension_se": 0.01,
        "radius_worst": 15.0,
        "texture_worst": 20.0,
        "perimeter_worst": 90.0,
        "area_worst": 700.0,
        "smoothness_worst": 0.1,
        "compactness_worst": 0.1,
        "concavity_worst": 0.08,
        "concave_points_worst": 0.05,
        "symmetry_worst": 0.25,
        "fractal_dimension_worst": 0.07,
    }
    response = client.get("/predict/random_forest", params=params)
    assert response.status_code == 200
    payload = response.json()
    assert "prediction" in payload
    assert "probability" in payload
    assert "model" in payload
