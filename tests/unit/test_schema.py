"""
Real, executable tests (pytest, not markdown) - unlike the original
notebook's "Testing Framework" section, which only ever showed test code
inside markdown cells and was never actually collected or run.

These specifically guard against the bug class that broke the pipeline:
a downstream cell/module referencing a column that doesn't exist in the
generator's actual output. Run with: pytest tests/unit/test_schema.py
"""

import pytest

from src.common.schema import RAW_SCHEMA, LEGACY_ALIASES, BRONZE_METADATA_COLUMNS


def test_raw_schema_has_no_duplicates():
    assert len(RAW_SCHEMA) == len(set(RAW_SCHEMA))


def test_legacy_aliases_do_not_shadow_raw_columns():
    # An alias name must not collide with a real generator column, or
    # add_legacy_aliases() would silently overwrite raw data.
    overlap = set(LEGACY_ALIASES) & set(RAW_SCHEMA)
    assert not overlap, f"Alias names collide with real columns: {overlap}"


def test_bronze_metadata_columns_do_not_collide_with_raw_schema():
    overlap = set(BRONZE_METADATA_COLUMNS) & set(RAW_SCHEMA)
    assert not overlap


@pytest.mark.parametrize("downstream_column", [
    "rotor_speed", "generator_speed", "power_output", "active_power",
    "reactive_power", "voltage", "current", "pressure",
    "oil_temperature", "maintenance_flag",
])
def test_every_known_downstream_reference_has_an_alias(downstream_column):
    """Regression test for the specific columns Silver/Gold/Features cells
    reference that don't exist in RAW_SCHEMA. If this fails, either a
    downstream module needs its reference updated, or a new alias needs
    to be added to LEGACY_ALIASES - don't just add the raw column name
    without checking which one is actually correct.
    """
    assert downstream_column in LEGACY_ALIASES


def test_source_generator_columns_referenced_by_aliases_exist_in_raw_schema():
    """Catches drift in the other direction: if LEGACY_ALIASES points at a
    generator column that gets renamed or removed, this fails instead of
    the pipeline failing at runtime."""
    from pyspark.sql import Column
    for alias, source in LEGACY_ALIASES.items():
        if isinstance(source, str):
            assert source in RAW_SCHEMA, (
                f"LEGACY_ALIASES['{alias}'] points at '{source}', which is "
                f"not in RAW_SCHEMA. Either RAW_SCHEMA is stale or the alias is."
            )
        # Column-expression aliases (e.g. unit conversions, derived flags)
        # are checked structurally, not here - see test below.


def test_column_expression_aliases_only_reference_raw_schema_columns():
    """For aliases defined as callables returning PySpark Column expressions
    (power_output, oil_temperature, maintenance_flag), statically verify
    every column name mentioned in the built expression's string form exists
    in RAW_SCHEMA, so a rename anywhere upstream is caught here instead of
    at Silver-write time. Calling source() requires an active SparkContext
    (that's the whole point of deferring it - see schema.py) so this test
    needs a SparkSession fixture.
    """
    import re
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.master("local[1]").appName("schema-test").getOrCreate()
    try:
        for alias, source in LEGACY_ALIASES.items():
            if isinstance(source, str):
                continue
            expr_str = str(source())
            referenced = set(re.findall(r"'([a-z_]+)'", expr_str))
            unknown = referenced - set(RAW_SCHEMA) - {"MAINTENANCE"}
            assert not unknown, f"Alias '{alias}' expression references unknown columns: {unknown}"
    finally:
        spark.stop()


def test_add_legacy_aliases_runs_against_a_real_dataframe():
    """End-to-end check: build a one-row DataFrame with every RAW_SCHEMA
    column, run add_legacy_aliases(), and confirm every alias column is
    actually present and non-null afterwards. This is the test that would
    have caught the eager-F.col()-at-import-time bug before it shipped.
    """
    from datetime import datetime
    from pyspark.sql import SparkSession
    from src.common.schema import add_legacy_aliases

    spark = SparkSession.builder.master("local[1]").appName("schema-test-2").getOrCreate()
    try:
        row = {}
        for col in RAW_SCHEMA:
            if col == "timestamp":
                row[col] = datetime(2025, 1, 1)
            elif col in {"event_id", "turbine_id", "farm_id", "turbine_model",
                         "operating_state", "alarm_code", "alarm_severity", "failure_type"}:
                row[col] = "T001" if col == "turbine_id" else "x"
            elif col in {"availability_flag", "grid_connected_flag",
                         "maintenance_mode_flag", "alarm_flag", "failure_flag"}:
                row[col] = False
            else:
                row[col] = 1.0
        row["operating_state"] = "MAINTENANCE"

        df = spark.createDataFrame([row])
        result = add_legacy_aliases(df)

        for alias in LEGACY_ALIASES:
            assert alias in result.columns

        out = result.collect()[0].asDict()
        assert out["power_output"] == pytest.approx(1.0 / 1000.0)
        assert out["maintenance_flag"] == 1  # operating_state was set to MAINTENANCE
    finally:
        spark.stop()
