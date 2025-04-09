import argparse
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def train_model(data_path, target_col, model_path):
    df = pd.read_csv(data_path)
    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")

    # Split features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("Model trained. Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save model
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an XGBoost model and save it.")
    parser.add_argument('--data', required=True, help='Path to training data CSV')
    parser.add_argument('--target', required=True, help='Target column name')
    parser.add_argument('--output', required=True, help='Path to save model file (.pkl)')

    args = parser.parse_args()
    train_model(args.data, args.target, args.output)
