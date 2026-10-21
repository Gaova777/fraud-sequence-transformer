"""What the API needs to know about a customer to score a new transaction: their usual spending and where they are."""
from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float64, Int64, String

customer = Entity(name="customer", join_keys=["user"])

profiles = FileSource(path="data/customer_profile.parquet", timestamp_field="event_timestamp")

customer_profile = FeatureView(
    name="customer_profile",
    entities=[customer],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="mean_amount", dtype=Float64),
        Field(name="last_ts", dtype=Int64),
        Field(name="home_state", dtype=String),
        Field(name="tx_count", dtype=Int64),
    ],
    source=profiles,
    online=True,
)
