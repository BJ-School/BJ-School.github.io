# model_evaluation.py
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

def evaluateModel(X_train, y_train, X_test, y_test, modelName="XGBoost"):
    # Initialize the model based on modelName
    if modelName == "LightGBM":
        model = LGBMClassifier()
    elif modelName == "XGBoost":
        model = XGBClassifier()
    elif modelName == "HistGBM":
        model = HistGradientBoostingClassifier()
    elif modelName == "LogisticRegression":
        model = LogisticRegression()
    else:
        raise ValueError("Invalid modelName. Choose from 'LightGBM', 'XGBoost', 'HistGBM', or 'LogisticRegression'.")
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Predict using the model
    y_pred = model.predict(X_test)
    
    # Calculate and print various metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n{modelName} Accuracy: {accuracy}")
    
    f1 = f1_score(y_test, y_pred)
    print(f"{modelName} F1 Score: {f1}")
    
    precision = f1_score(y_test, y_pred)
    print(f"{modelName} Precision Score: {precision}")
    
    recall = f1_score(y_test, y_pred)
    print(f"{modelName} Recall Score: {recall}")
    
    r2 = f1_score(y_test, y_pred)
    print(f"{modelName} R2 Score: {r2}")
    
    root_mean2_error = f1_score(y_test, y_pred)
    print(f"{modelName} Root Mean2 Error: {root_mean2_error}")

    # Return the trained model
    return model