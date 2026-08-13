from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "skill_gap_raw.csv"
OUT = ROOT / "data" / "skill_gap_features.parquet"


def main():
    df = pd.read_csv(RAW)
    numeric = [
        "curriculum_coverage",
        "industry_demand",
        "job_posting_frequency",
        "practical_exposure",
    ]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="raise").clip(0, 100)

    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], utc=True)
    df["skill_gap_score"] = df["industry_demand"] - df["curriculum_coverage"]
    df["industry_pressure"] = (
        df["industry_demand"] + df["job_posting_frequency"]
    ) / 2
    df["readiness_score"] = (
        df["curriculum_coverage"] + df["practical_exposure"]
    ) / 2
    df["high_gap"] = (df["skill_gap_score"] >= 20).astype("int64")

    columns = [
        "skill_id", "skill_name", "curriculum_coverage", "industry_demand",
        "job_posting_frequency", "practical_exposure", "event_timestamp",
        "skill_gap_score", "industry_pressure", "readiness_score", "high_gap",
    ]
    df[columns].to_parquet(OUT, index=False)
    print(f"Created {OUT}")
    print(df[columns].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
