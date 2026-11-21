from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

def get_models():

    models = {
        "Linear Regression with Lags" : Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),

        "Polynomial Regression" : Pipeline([
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("model", LinearRegression())
        ]),

        "Ridge Regression" : Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0))
        ]),

        "KNN Regressor" : Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsRegressor(n_neighbors=5))
        ]),

        "MLP Regressor" : Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPRegressor(hidden_layer_sizes=(100, 50),
                                   max_iter=2000, random_state= 42))
        ])
    }

    return models