from __future__ import annotations

import pandas as pd

from services.dataset_profiler import profile_dataset
from services.query_guide import (
    build_dataset_query_guide,
    build_hint_chips,
    build_query_examples,
    suggest_columns,
)


def test_query_guide_builds_schema_examples_and_suggestions() -> None:
    dataframe = pd.DataFrame({
        "Order Date": ["2017-01-01", "2017-02-01"],
        "Ship Date": ["2017-01-03", "2017-02-03"],
        "Ship Mode": ["Standard Class", "Second Class"],
        "Region": ["West", "East"],
        "Sales": [100.0, 200.0],
        "Profit": [10.0, 20.0],
    })

    guide = build_dataset_query_guide(dataframe, profile_dataset(dataframe))
    examples = build_query_examples(guide)
    chips = build_hint_chips(guide)

    assert [column.column_name for column in guide.date_columns] == ["Order Date", "Ship Date"]
    assert "Using Order Date, show monthly Profit for 2017." in examples
    assert "Show value counts of Ship Mode." in examples
    assert "Order Date" in chips["Date field"]
    assert "Sales" in chips["Metric"]
    assert "Ship Mode" in suggest_columns("ship", guide)
    assert "Sales" in suggest_columns("revenue", guide)
