"""
Training script for PaySim prototype.

Usage:
python -m src.training.train --input data/raw/paysim.csv --out-dir src/app/models --sample 200000
"""

import argparse
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE

from src.app.preprocessing.pipeline import PreprocessingPipeline


def load_csv(path: str) -> pd.DataFrame:
    print(f"[INFO] Loading dataset from: {path}")
    df = pd.read_csv(path)
    print(f"[INFO] Dataset loaded successfully with shape: {df.shape}")
    return df


def train(args):
    print("\n🚀 Starting PaySim model training...\n")

    df = load_csv(args.input)

    # Optional sampling
    if args.sample and args.sample > 0:
        df = df.sample(n=args.sample, random_state=42).reset_index(drop=True)
        print(f"[INFO] Using random sample of {len(df)} records for training.\n")

    # Target column check
    if "isFraud" not in df.columns:
        raise RuntimeError("❌ Input CSV must contain 'isFraud' column.")

    # Drop identifiers
    drop_cols = [c for c in ["nameOrig", "nameDest"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        print(f"[INFO] Dropped identifier columns: {drop_cols}\n")

    target = "isFraud"
    y = df[target]
    X = df.drop(columns=[target])

    print("[STEP 1] Starting preprocessing pipeline...")
    pipeline = PreprocessingPipeline()
    X_proc = pipeline.fit_transform(X)
    print(f"[INFO] Preprocessing complete. Feature matrix shape: {X_proc.shape}\n")

    print("[STEP 2] Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_proc, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[INFO] Train size: {X_train.shape}, Test size: {X_test.shape}")
    print(f"[INFO] Positive class ratio before SMOTE: {y_train.mean():.4f}\n")

    print("[STEP 3] Applying SMOTE (balancing classes)...")
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    print(f"[INFO] Class ratio after SMOTE: {y_train_res.mean():.4f}")
    print(f"[INFO] Resampled train size: {X_train_res.shape}\n")

    print("[STEP 4] Training RandomForestClassifier...")
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train_res, y_train_res)
    print("[INFO] Model training completed successfully!\n")

    print("[STEP 5] Evaluating model performance...")
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    print("\n📊 Classification Report (Test Set):")
    print(classification_report(y_test, y_pred, digits=4))
    auc = roc_auc_score(y_test, y_proba)
    print(f"[INFO] ROC AUC Score: {auc:.4f}\n")

    print("[STEP 6] Saving model artifacts...")
    os.makedirs(args.out_dir, exist_ok=True)
    model_path = os.path.join(args.out_dir, "model.joblib")
    pipeline_path = os.path.join(args.out_dir, "preprocessing.joblib")
    joblib.dump(clf, model_path)
    joblib.dump(pipeline, pipeline_path)
    print(f"[✅ SAVED] Model → {model_path}")
    print(f"[✅ SAVED] Preprocessing Pipeline → {pipeline_path}\n")

    print("🎉 Training completed successfully! Ready for API integration.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to raw PaySim CSV file")
    parser.add_argument("--out-dir", type=str, default="src/app/models", help="Output directory for model")
    parser.add_argument("--sample", type=int, default=50000, help="Optional sample size for faster training")

    args = parser.parse_args()
    train(args)
