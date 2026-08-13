from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs" / "historical_features.csv"
MODEL = ROOT / "outputs" / "skill_gap_model.joblib"
METRICS = ROOT / "outputs" / "model_metrics.json"

FEATURE_COLUMNS = [
    "curriculum_coverage",
    "industry_demand",
    "job_posting_frequency",
    "practical_exposure",
    "skill_gap_score",
    "industry_pressure",
    "readiness_score",
]


def main():
    df = pd.read_csv(DATA)
    X = df[FEATURE_COLUMNS]
    y = df["high_gap"].astype(int)

    if y.nunique() < 2:
        raise ValueError("The dataset needs both high-gap and low-gap examples.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)

    report = classification_report(y_test, pred, output_dict=True, zero_division=0)
    metrics = {"accuracy": float(accuracy), "classification_report": report}

    MODEL.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL)
    METRICS.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Model accuracy: {accuracy:.4f}")
    print(json.dumps(report, indent=2))
    print(f"Model saved to {MODEL}")


if __name__ == "__main__":
    main()
