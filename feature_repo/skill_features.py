from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float64, Int64

ROOT = Path(__file__).resolve().parents[1]
PARQUET_PATH = ROOT / "data" / "skill_gap_features.parquet"

skill = Entity(
    name="skill",
    join_keys=["skill_id"],
    description="A curriculum-industry technical skill.",
)

skill_gap_source = FileSource(
    name="skill_gap_features_source",
    path=str(PARQUET_PATH),
    timestamp_field="event_timestamp",
)

skill_gap_features = FeatureView(
    name="skill_gap_features",
    entities=[skill],
    ttl=timedelta(days=365),
    schema=[
        Field(name="curriculum_coverage", dtype=Float64),
        Field(name="industry_demand", dtype=Float64),
        Field(name="job_posting_frequency", dtype=Float64),
        Field(name="practical_exposure", dtype=Float64),
        Field(name="skill_gap_score", dtype=Float64),
        Field(name="industry_pressure", dtype=Float64),
        Field(name="readiness_score", dtype=Float64),
        Field(name="high_gap", dtype=Int64),
    ],
    source=skill_gap_source,
    online=True,
)
