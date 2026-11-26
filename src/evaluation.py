from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def evaluate_model(model, X_train, X_test, y_train, y_test):

    # train model
    model.fit(X_train, y_train)

    # predicting groundwater levels
    predictions = model.predict(X_test)

    # calculating evaluation metrics
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    return{
        "MAE" : mae,
        "RMSE" : rmse,
        "R2" : r2
    }