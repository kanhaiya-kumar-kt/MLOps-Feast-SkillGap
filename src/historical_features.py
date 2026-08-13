from pathlib import Path
import pandas as pd
from feast import FeatureStore

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "historical_features.csv"
STORE = ROOT / "feature_repo"

FEATURES = [
    "skill_gap_features:curriculum_coverage",
    "skill_gap_features:industry_demand",
    "skill_gap_features:job_posting_frequency",
    "skill_gap_features:practical_exposure",
    "skill_gap_features:skill_gap_score",
    "skill_gap_features:industry_pressure",
    "skill_gap_features:readiness_score",
    "skill_gap_features:high_gap",
]


def main():
    source = pd.read_parquet(ROOT / "data" / "skill_gap_features.parquet")
    entity_df = source[["skill_id", "event_timestamp", "high_gap"]].copy()
    entity_df["event_timestamp"] = pd.to_datetime(entity_df["event_timestamp"], utc=True)

    store = FeatureStore(repo_path=str(STORE))
    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=FEATURES,
    ).to_df()

    OUT.parent.mkdir(exist_ok=True)
    training_df.to_csv(OUT, index=False)
    print("Historical feature output:")
    print(training_df.head(10).to_string(index=False))
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
