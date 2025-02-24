import pandas as pd
from scipy import stats
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df=pd.read_csv("E:\Project1\car_price_ai.csv")

from sklearn.preprocessing import OrdinalEncoder
encoder=OrdinalEncoder()
for i in df.select_dtypes(include=["object"]).columns:
    df[i]=encoder.fit_transform(df[[i]])

df["km_driven"]= np.cbrt(df["km_driven"])
df["price"]=stats.boxcox(df["price"],lmbda=0)
df["power"]=stats.boxcox(df["power"],lmbda=-0.5)

y=df["price"]
x=df.drop("price",axis=1)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


mlflow.set_experiment("Car_Price_Prediction")
#mlflow.set_tracking_uri("http://127.0.0.1:8501")
# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "XGBoost": xgb.XGBRegressor(),
    "SVR": SVR()
}
#run_name=model_name
for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        # Calculate Metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Log parameters and metrics
        mlflow.log_param("Model", model_name)
        mlflow.log_metric("Mean Squared Error", mse)
        mlflow.log_metric("Mean Absolute Error", mae)
        mlflow.log_metric("R² Score", r2)

        # Log the model
        mlflow.sklearn.log_model(model, model_name)

        print(f"{model_name}:")
        print(f"  Mean Squared Error: {mse:.2f}")
        print(f"  Mean Absolute Error: {mae:.2f}")
        print(f"  R² Score: {r2:.2f}\n")

print("All models trained and logged in MLflow.")