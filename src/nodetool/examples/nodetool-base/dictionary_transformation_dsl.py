# pyright: reportMissingImports=false

"""
Dictionary Transformation DSL Example

Normalize nested API payloads into a clean profile dictionary, enrich with derived
fields, and validate the result against a JSON schema.

Workflow:
1. **Source Payload** – Parse raw JSON into a dictionary object
2. **Selection & Extraction** – Keep relevant sections and pull nested fields
3. **Derived Fields** – Compute full name, uppercase tier, lifecycle category
4. **Dictionary Assembly** – Merge segments, metrics, and lifecycle details
5. **Schema Validation** – Ensure final profile matches required structure
6. **Outputs** – Emit normalized dictionary, JSON preview, validation flag, summary

Streaming behavior:
- All dictionary operations run in batch mode; no streaming values are emitted.
- Validation waits for the complete dictionary before producing a result.

ASCII workflow outline:

        [JSON Payload]
              |
         [ParseJSON]
              |
        [Filter Keys]
              |
    +---- extract handles ----+
    |      |       |         |
 [Get]  [Get]   [Get]    [Get]
    \\      |       |         /
     \\  derived FormatText  /
      \\        |           /
       [MakeDictionary nodes]
              |
        [Combine/Update]
              |
        [ValidateJSON]---\\
              |          |
        [StringifyJSON]  |
              |          |
    [Outputs: dict/json/summary/bool]
"""

from nodetool.dsl.graph import create_graph, run_graph
from nodetool.dsl.nodetool.dictionary import (
    ParseJSON,
    Filter,
    GetValue,
    MakeDictionary,
    Combine,
)
from nodetool.dsl.lib.json import ValidateJSON, StringifyJSON
from nodetool.dsl.nodetool.text import FormatText
from nodetool.dsl.nodetool.output import (
    DictionaryOutput,
    StringOutput,
    BooleanOutput,
)


# --- Source payload ---------------------------------------------------------
raw_payload = ParseJSON(
    json_string="""{
  "user": {
    "id": "cust_8241",
    "first_name": "Leah",
    "last_name": "Jimenez",
    "email": "leah.jimenez@example.com"
  },
  "account": {
    "tier": "growth",
    "seats": 28,
    "renewal_date": "2025-02-01"
  },
  "metrics": {
    "mrr": 1825.75,
    "currency": "USD",
    "churn_risk": 0.18
  },
  "metadata": {
    "signup_source": "webinar",
    "last_login": "2025-01-15T09:20:00Z"
  },
  "tags": ["b2b", "priority", "expansion"]
}""",
)

# Keep only the sections we care about.
selected_sections = Filter(
    dictionary=raw_payload.output,
    keys=["user", "account", "metrics", "metadata", "tags"],
)

# --- Nested extraction ------------------------------------------------------
user_info = GetValue(dictionary=selected_sections.output, key="user", default={})
account_info = GetValue(dictionary=selected_sections.output, key="account", default={})
metrics_info = GetValue(dictionary=selected_sections.output, key="metrics", default={})
metadata_info = GetValue(dictionary=selected_sections.output, key="metadata", default={})
segments_list = GetValue(
    dictionary=selected_sections.output,
    key="tags",
    default=[],
)

first_name = GetValue(dictionary=user_info.output, key="first_name", default="")
last_name = GetValue(dictionary=user_info.output, key="last_name", default="")
email = GetValue(dictionary=user_info.output, key="email", default="")
customer_id = GetValue(dictionary=user_info.output, key="id", default="")

tier = GetValue(dictionary=account_info.output, key="tier", default="")
seat_count = GetValue(dictionary=account_info.output, key="seats", default=0)
renewal_date = GetValue(dictionary=account_info.output, key="renewal_date", default="")

mrr_value = GetValue(dictionary=metrics_info.output, key="mrr", default=0.0)
currency = GetValue(dictionary=metrics_info.output, key="currency", default="USD")
churn_risk = GetValue(dictionary=metrics_info.output, key="churn_risk", default=0.0)

signup_source = GetValue(
    dictionary=metadata_info.output,
    key="signup_source",
    default="unknown",
)
last_login = GetValue(
    dictionary=metadata_info.output,
    key="last_login",
    default="1970-01-01T00:00:00Z",
)

# --- Derived fields ---------------------------------------------------------
full_name = FormatText(
    template="{{ first }} {{ last }}",
    first=first_name.output,
    last=last_name.output,
)

tier_label = FormatText(
    template="{{ tier | upper }}",
    tier=tier.output,
)

churn_category = FormatText(
    template="""{% if risk <= 0.1 %}healthy{% elif risk <= 0.25 %}watchlist{% else %}at-risk{% endif %}""",
    risk=churn_risk.output,
)

renewal_outlook = FormatText(
    template="Renews on {{ date }} via {{ channel }} channel.",
    date=renewal_date.output,
    channel=signup_source.output,
)

# --- Dictionary assembly ----------------------------------------------------
core_profile = MakeDictionary(
    customer_id=customer_id.output,
    full_name=full_name.output,
    email=email.output,
    signup_channel=signup_source.output,
)

plan_profile = MakeDictionary(
    plan_tier=tier_label.output,
    seat_count=seat_count.output,
    renewal_date=renewal_date.output,
)

metrics_profile = MakeDictionary(
    mrr=mrr_value.output,
    currency=currency.output,
    churn_risk=churn_risk.output,
)

lifecycle_details = MakeDictionary(
    churn_category=churn_category.output,
    renewal_date=renewal_date.output,
    last_login=last_login.output,
)

lifecycle_wrapper = MakeDictionary(
    lifecycle=lifecycle_details.output,
    renewal_overview=renewal_outlook.output,
)

segments_profile = MakeDictionary(
    segments=segments_list.output,
)

profile_with_plan = Combine(dict_a=core_profile.output, dict_b=plan_profile.output)
profile_with_metrics = Combine(
    dict_a=profile_with_plan.output,
    dict_b=metrics_profile.output,
)
profile_with_lifecycle = Combine(
    dict_a=profile_with_metrics.output,
    dict_b=lifecycle_wrapper.output,
)
normalized_profile = Combine(
    dict_a=profile_with_lifecycle.output,
    dict_b=segments_profile.output,
)

# --- Schema validation ------------------------------------------------------
profile_schema = ParseJSON(
    json_string="""{
  "type": "object",
  "required": [
    "customer_id",
    "full_name",
    "email",
    "plan_tier",
    "mrr",
    "currency",
    "segments",
    "lifecycle"
  ],
  "properties": {
    "customer_id": { "type": "string" },
    "full_name": { "type": "string" },
    "email": { "type": "string" },
    "plan_tier": { "type": "string" },
    "seat_count": { "type": "integer" },
    "renewal_date": { "type": "string" },
    "mrr": { "type": "number" },
    "currency": { "type": "string" },
    "churn_risk": { "type": "number" },
    "renewal_overview": { "type": "string" },
    "segments": {
      "type": "array",
      "items": { "type": "string" }
    },
    "lifecycle": {
      "type": "object",
      "required": ["churn_category", "renewal_date", "last_login"],
      "properties": {
        "churn_category": { "type": "string" },
        "renewal_date": { "type": "string" },
        "last_login": { "type": "string" }
      }
    }
  }
}""",
)

validation = ValidateJSON(
    data=normalized_profile.output,
    json_schema=profile_schema.output,
)

profile_json = StringifyJSON(
    data=normalized_profile.output,
    indent=2,
)

# --- Reporting --------------------------------------------------------------
summary_text = FormatText(
    template="""# Normalized Customer Profile

- **Customer ID:** {{ customer_id }}
- **Name:** {{ full_name }}
- **Plan:** {{ plan_tier }} ({{ seat_count }} seats)
- **Monthly Recurring Revenue:** ${{ mrr | round(2) }} {{ currency }}
- **Churn Risk:** {{ churn_risk | round(2) }} ({{ churn_category }})
- **Lifecycle:** {{ renewal_overview }}
- **Segments:** {{ segments | join(', ') }}

Schema validation passed? {{ validation }}
""",
    customer_id=customer_id.output,
    full_name=full_name.output,
    plan_tier=tier_label.output,
    seat_count=seat_count.output,
    mrr=mrr_value.output,
    currency=currency.output,
    churn_risk=churn_risk.output,
    churn_category=churn_category.output,
    renewal_overview=renewal_outlook.output,
    segments=segments_list.output,
    validation=validation.output,
)

# --- Outputs ----------------------------------------------------------------
profile_output = DictionaryOutput(
    name="normalized_profile",
    value=normalized_profile.output,
)

json_preview_output = StringOutput(
    name="normalized_profile_json",
    value=profile_json.output,
)

summary_output = StringOutput(
    name="profile_summary",
    value=summary_text.output,
)

validation_output = BooleanOutput(
    name="schema_valid",
    value=validation.output,
)

graph = create_graph(
    profile_output,
    json_preview_output,
    summary_output,
    validation_output,
)


if __name__ == "__main__":
    result = run_graph(graph)
    print("✅ Dictionary transformation complete!")
    print("Schema valid:", result["schema_valid"])
