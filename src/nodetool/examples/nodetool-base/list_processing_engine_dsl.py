"""
List Processing Engine DSL Example

Batch-transform customer records with map/filter/reduce style operations to produce
regional summaries and marketing insights.

Workflow:
1. **Customer Dataset** – Provide sample subscription records (list constant)
2. **Filter Stages** – Keep only active customers above a revenue threshold
3. **Feature Mapping** – Extract regions and emails for downstream use
4. **Batch Analytics** – Convert to DataFrame and aggregate counts & revenue by region
5. **Summaries** – Produce structured table plus narrative insight
6. **Outputs** – Expose tabular metrics, narrative summary, and campaign email list

Streaming behavior:
- All list and dataframe nodes operate on the entire collection at once (batch mode).
- No streaming branches are used; outputs accumulate only after full computation.

ASCII workflow outline:

    [List Constant]
           |
    [FilterDicts]  -- removes inactive records
           |
    [FilterDictsByNumber] -- keeps high-value customers
           |
      ______________________________
     /                              \
 [MapField] -> [Dedupe] -> [Sort]     \
     |                                 \
 [ListOutput]                          [FromList]
                                         |
                          ______________________________
                         /                              \
                [SelectColumn(region,id)]     [SelectColumn(region,mrr)]
                         |                              |
                     [Aggregate count]            [Aggregate sum]
                         |                              |
                     [Rename -> value]           [Rename -> value]
                         \                              /
                          \_____[Join on region]_______/
                                    |
                             [Rename metrics]
                                    |
             ______________________/ \____________________
            /                                             \
 [DataframeOutput]                             [FormatText -> StringOutput]
                                                     |
                                             narrative summary
"""

from nodetool.dsl.graph import create_graph, run_graph
from nodetool.dsl.nodetool.constant import List as ListConstant
from nodetool.dsl.nodetool.list import (
    FilterDicts,
    FilterDictsByNumber,
    MapField,
    Dedupe,
    Sort,
    Length,
)
from nodetool.dsl.nodetool.data import (
    FromList,
    SelectColumn,
    Aggregate,
    Rename,
    Join,
    ToList,
    SortByColumn,
)
from nodetool.dsl.nodetool.output import DataframeOutput, StringOutput, ListOutput
from nodetool.dsl.nodetool.text import FormatText


# --- Customer dataset -------------------------------------------------------
customer_records = ListConstant(
    value=[
        {
            "id": 101,
            "name": "Alice Gomez",
            "email": "alice@example.com",
            "plan": "Scale",
            "region": "North America",
            "status": "active",
            "mrr": 2200,
        },
        {
            "id": 102,
            "name": "Ben Yoon",
            "email": "ben@example.com",
            "plan": "Growth",
            "region": "Asia Pacific",
            "status": "trial",
            "mrr": 300,
        },
        {
            "id": 103,
            "name": "Chloe Martin",
            "email": "chloe@example.com",
            "plan": "Scale",
            "region": "EMEA",
            "status": "active",
            "mrr": 1850,
        },
        {
            "id": 104,
            "name": "Deepak Verma",
            "email": "deepak@example.com",
            "plan": "Enterprise",
            "region": "Asia Pacific",
            "status": "active",
            "mrr": 4200,
        },
        {
            "id": 105,
            "name": "Ella Chen",
            "email": "ella@example.com",
            "plan": "Growth",
            "region": "North America",
            "status": "active",
            "mrr": 980,
        },
        {
            "id": 106,
            "name": "Farah Hussein",
            "email": "farah@example.com",
            "plan": "Enterprise",
            "region": "EMEA",
            "status": "active",
            "mrr": 3100,
        },
        {
            "id": 107,
            "name": "Gabe Silva",
            "email": "gabe@example.com",
            "plan": "Scale",
            "region": "Latin America",
            "status": "active",
            "mrr": 1560,
        },
        {
            "id": 108,
            "name": "Hana Ito",
            "email": "hana@example.com",
            "plan": "Growth",
            "region": "Asia Pacific",
            "status": "active",
            "mrr": 1210,
        },
        {
            "id": 109,
            "name": "Ivan Petrov",
            "email": "ivan@example.com",
            "plan": "Scale",
            "region": "EMEA",
            "status": "churned",
            "mrr": 0,
        },
        {
            "id": 110,
            "name": "Jana Lopez",
            "email": "jana@example.com",
            "plan": "Enterprise",
            "region": "North America",
            "status": "active",
            "mrr": 5100,
        },
    ]
)


# --- List transformations ---------------------------------------------------
active_customers = FilterDicts(
    values=customer_records.output,
    condition="status == 'active'",
)

high_value_customers = FilterDictsByNumber(
    values=active_customers.output,
    key="mrr",
    filter_type=FilterDictsByNumber.FilterDictNumberType.GREATER_THAN,
    value=1200,
)

regions = MapField(values=high_value_customers.output, field="region")
unique_regions = Dedupe(values=regions.output)
sorted_regions = Sort(
    values=unique_regions.output,
    order=Sort.SortOrder.ASCENDING,
)

marketing_emails = MapField(values=high_value_customers.output, field="email")

high_value_count = Length(values=high_value_customers.output)


# --- Dataframe analytics ----------------------------------------------------
customer_dataframe = FromList(values=high_value_customers.output)

detailed_columns = SelectColumn(
    dataframe=customer_dataframe.output,
    columns="id,name,email,plan,region,mrr",
)

detailed_sorted = SortByColumn(df=detailed_columns.output, column="region")

region_id = SelectColumn(
    dataframe=customer_dataframe.output,
    columns="region,id",
)
region_counts = Aggregate(
    dataframe=region_id.output,
    columns="region",
    aggregation="count",
)
count_metric = Rename(
    dataframe=region_counts.output,
    rename_map="id:value",
)

region_mrr = SelectColumn(
    dataframe=customer_dataframe.output,
    columns="region,mrr",
)
region_totals = Aggregate(
    dataframe=region_mrr.output,
    columns="region",
    aggregation="sum",
)
revenue_metric = Rename(
    dataframe=region_totals.output,
    rename_map="mrr:value",
)

regional_combined = Join(
    dataframe_a=count_metric.output,
    dataframe_b=revenue_metric.output,
    join_on="region",
)
regional_summary = Rename(
    dataframe=regional_combined.output,
    rename_map="value_x:customer_count,value_y:total_mrr",
)

regional_records = ToList(dataframe=regional_summary.output)


# --- Narrative summary ------------------------------------------------------
summary_text = FormatText(
    template="""# High-Value Customer Summary

- Total high-value accounts: {{ hv_count }}
- Active regions (sorted): {{ regions | join(', ') }}
- Campaign emails: {{ emails | join(', ') }}

## Regional Breakdown
{% for row in summary %}
- {{ row.region }} → {{ row.customer_count }} customers • Total MRR: ${{ row.total_mrr | round(0) }}
{% endfor %}
""",
    hv_count=high_value_count.output,
    regions=sorted_regions.output,
    emails=marketing_emails.output,
    summary=regional_records.output,
)


# --- Outputs ----------------------------------------------------------------
regional_output = DataframeOutput(
    name="regional_summary",
    value=regional_summary.output,
)

customers_output = DataframeOutput(
    name="high_value_customers",
    value=detailed_sorted.output,
)

emails_output = ListOutput(
    name="campaign_emails",
    value=marketing_emails.output,
)

summary_output = StringOutput(
    name="summary_report",
    value=summary_text.output,
)

graph = create_graph(
    regional_output,
    customers_output,
    emails_output,
    summary_output,
)


if __name__ == "__main__":
    result = run_graph(graph)
    print("✅ List processing engine completed!")
    print("Summary preview:")
    print(result["summary_report"])
