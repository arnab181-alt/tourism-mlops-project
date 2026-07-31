"""Model Training, Experiment Tracking (MLflow) and model export.

Reads the train/test splits produced by prep.py, builds a preprocessing +
XGBoost pipeline, tunes it with GridSearchCV, logs everything to MLflow, and
saves the best model into tourism_project/deployment/ so the workflow can
commit it back to the repository for the Streamlit app to use.
"""
import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import xgboost as xgb

DEPLOY_DIR = "tourism_project/deployment"
MODEL_PATH = os.path.join(DEPLOY_DIR, "best_tourism_model_v1.joblib")

CATEGORICAL = ["TypeofContact", "Occupation", "Gender",
               "ProductPitched", "MaritalStatus", "Designation"]


def main():
    Xtrain = pd.read_csv("Xtrain.csv")
    Xtest = pd.read_csv("Xtest.csv")
    ytrain = pd.read_csv("ytrain.csv").squeeze("columns")
    ytest = pd.read_csv("ytest.csv").squeeze("columns")

    numeric = [c for c in Xtrain.columns if c not in CATEGORICAL]

    numeric_pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    categorical_pipe = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore"),
    )
    preprocessor = make_column_transformer(
        (numeric_pipe, numeric),
        (categorical_pipe, CATEGORICAL),
    )

    # Handle class imbalance (~19% positives)
    scale_pos_weight = (ytrain == 0).sum() / (ytrain == 1).sum()
    xgb_clf = xgb.XGBClassifier(
        eval_metric="logloss",
        random_state=42,
        scale_pos_weight=scale_pos_weight,
    )
    model = make_pipeline(preprocessor, xgb_clf)

    param_grid = {
        "xgbclassifier__n_estimators": [150, 300],
        "xgbclassifier__max_depth": [3, 5],
        "xgbclassifier__learning_rate": [0.05, 0.1],
    }

    mlflow.set_experiment("tourism-wellness-package")
    with mlflow.start_run():
        grid = GridSearchCV(model, param_grid, cv=3, scoring="f1", n_jobs=-1)
        grid.fit(Xtrain, ytrain)
        best = grid.best_estimator_

        preds = best.predict(Xtest)
        proba = best.predict_proba(Xtest)[:, 1]
        report = classification_report(ytest, preds, output_dict=True)
        auc = roc_auc_score(ytest, proba)

        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("test_accuracy", report["accuracy"])
        mlflow.log_metric("test_precision_class1", report["1"]["precision"])
        mlflow.log_metric("test_recall_class1", report["1"]["recall"])
        mlflow.log_metric("test_f1_class1", report["1"]["f1-score"])
        mlflow.log_metric("test_roc_auc", auc)

        # Save the deployable model with joblib and track it as an MLflow artifact.
        # (Logging the raw file is portable across all MLflow versions.)
        os.makedirs(DEPLOY_DIR, exist_ok=True)
        joblib.dump(best, MODEL_PATH)
        mlflow.log_artifact(MODEL_PATH)

        print("Best params:", grid.best_params_)
        print(classification_report(ytest, preds, digits=3))
        print("ROC-AUC:", round(auc, 3))
        print("Confusion matrix:")
        print(confusion_matrix(ytest, preds))
        print("Saved best model to", MODEL_PATH)


if __name__ == "__main__":
    main()
