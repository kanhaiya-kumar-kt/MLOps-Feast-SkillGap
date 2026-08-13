from pathlib import Path
import json
import joblib
import pandas as pd
from feast import FeatureStore

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "feature_repo"
MODEL = ROOT / "outputs" / "skill_gap_model.joblib"
OUT = ROOT / "outputs" / "online_prediction.json"

FEATURE_NAMES = [
    "curriculum_coverage",
    "industry_demand",
    "job_posting_frequency",
    "practical_exposure",
    "skill_gap_score",
    "industry_pressure",
    "readiness_score",
]
FEATURE_REFS = [f"skill_gap_features:{name}" for name in FEATURE_NAMES]


def main():
    model = joblib.load(MODEL)
    source = pd.read_parquet(ROOT / "data" / "skill_gap_features.parquet")
    skill_id = int(source.iloc[0]["skill_id"])

    store = FeatureStore(repo_path=str(STORE))
    response = store.get_online_features(
        features=FEATURE_REFS,
        entity_rows=[{"skill_id": skill_id}],
    ).to_dict()

    values = {name: response.get(name, [None])[0] for name in FEATURE_NAMES}
    feature_vector = pd.DataFrame([values])
    prediction = int(model.predict(feature_vector)[0])
    probability = float(model.predict_proba(feature_vector)[0][1])

    result = {
        "skill_id": skill_id,
        "online_features": values,
        "predicted_high_gap": prediction,
        "high_gap_probability": probability,
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("Online feature output:")
    print(json.dumps(result, indent=2))
    print(f"Final prediction: high_gap={prediction}")


if __name__ == "__main__":
    main()
