"""
Single source of truth for the SCADA sensor schema.

WHY THIS FILE EXISTS
---------------------
The original notebook generated data with one column schema (e.g.
`rotor_speed_rpm`, `active_power_kw`, `hydraulic_oil_temperature`) but the
Bronze DDL, Silver cleaning, Gold aggregation, and feature-engineering cells
were written against an earlier, simpler schema (`rotor_speed`, `power_output`,
`oil_temperature`, `voltage`, `current`, `maintenance_flag`...) that doesn't
exist in the generator's actual output. That mismatch is what broke the
pipeline (NameError / AnalysisException on missing columns) — five separate
cells (Bronze, Silver, Gold-KPI, Gold-training, Features) each hard-coded
their own copy of "what the columns are called."

Fixing this once, here, is the actual fix — not patching five call sites and
hoping a sixth doesn't drift the same way later.

RAW_SCHEMA is the literal column list the synthetic generator produces.
LEGACY_ALIASES maps names the rest of the pipeline was written to expect onto
the real generator columns. `add_legacy_aliases()` is called once, immediately
after Bronze ingestion, so every downstream module (Silver/Gold/Features/
Models) can keep using the column names it already references without each
of them needing to know about the generator's real schema.

If you rename a generator column, this is the ONLY file that needs to change.
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

# Exact columns produced by src/data_generation/synthetic_generator.py
RAW_SCHEMA = [
    "event_id", "turbine_id", "farm_id", "timestamp",
    "turbine_model", "turbine_age_years",
    "wind_speed", "wind_direction", "ambient_temperature", "humidity",
    "atmospheric_pressure", "air_density", "precipitation",
    "rotor_speed_rpm", "generator_speed_rpm",
    "blade_pitch_angle", "yaw_angle", "yaw_error",
    "gearbox_temperature", "generator_temperature", "bearing_temperature",
    "main_bearing_temperature", "nacelle_temperature", "brake_temperature",
    "hydraulic_pressure", "hydraulic_oil_temperature",
    "gearbox_oil_temperature", "gearbox_oil_pressure",
    "vibration_x", "vibration_y", "vibration_z",
    "vibration_rms", "vibration_peak", "vibration_kurtosis",
    "active_power_kw", "reactive_power_kvar", "apparent_power_kva",
    "voltage_v", "current_a", "power_factor", "grid_frequency_hz",
    "generator_load_pct", "electrical_efficiency",
    "operating_state", "availability_flag", "grid_connected_flag",
    "maintenance_mode_flag", "alarm_flag", "alarm_code", "alarm_severity",
    "failure_flag", "failure_type",
    "operating_hours", "health_status",
]

# name the rest of the pipeline expects -> real generator column (plain
# rename) OR a zero-arg callable returning a Column expression (derived
# value). Callables, not bare Column objects: building a Column calls
# pyspark's F.col(), which requires an active SparkContext - evaluating
# these eagerly at import time breaks any non-Databricks import (e.g.
# `pytest` without a live Spark session already running). Deferring
# construction until add_legacy_aliases() actually runs against a
# DataFrame is what fixes that.
LEGACY_ALIASES = {
    "rotor_speed": "rotor_speed_rpm",
    "generator_speed": "generator_speed_rpm",
    "active_power": "active_power_kw",
    "reactive_power": "reactive_power_kvar",
    "voltage": "voltage_v",
    "current": "current_a",
    "pressure": "atmospheric_pressure",
    # power_output (MW-scale, 0-3.0 range used by Silver's range check) is a
    # unit conversion, not a rename: active_power_kw / 1000
    "power_output": lambda: F.col("active_power_kw") / 1000.0,
    # oil_temperature never existed as a single sensor in the real spec —
    # generator produces hydraulic_oil_temperature AND gearbox_oil_temperature
    # separately. Silver/Gold referenced a single "oil_temperature"; this
    # derivation is an explicit, flagged approximation (avg of the two),
    # not a rediscovered ground truth. Replace with the specific sensor you
    # actually mean if precision matters downstream.
    "oil_temperature": lambda: (F.col("hydraulic_oil_temperature") + F.col("gearbox_oil_temperature")) / 2.0,
    # the generator never produces a maintenance_flag column at all (it
    # produces separate maintenance_events records). This derivation is a
    # placeholder — TRUE while operating_state == 'MAINTENANCE' — and should
    # be replaced by joining maintenance_events on turbine_id once that
    # table is wired into Silver.
    "maintenance_flag": lambda: F.when(F.col("operating_state") == "MAINTENANCE", 1).otherwise(0),
}


def add_legacy_aliases(df: DataFrame) -> DataFrame:
    """Add the legacy/simplified column names the rest of the pipeline
    (Silver, Gold, Features) was written against, as aliases of the real
    generator columns. Call this once, right after Bronze ingestion.
    """
    for alias, source in LEGACY_ALIASES.items():
        col_expr = source() if callable(source) else F.col(source)
        df = df.withColumn(alias, col_expr)
    return df


# Columns Bronze actually needs to store (real schema + ingestion metadata).
# Do NOT add LEGACY_ALIASES columns here — those are computed on read by
# add_legacy_aliases(), not persisted, so a future generator schema change
# doesn't leave stale duplicate columns sitting in Bronze.
BRONZE_METADATA_COLUMNS = [
    "ingestion_timestamp", "ingestion_date", "batch_id", "source_file",
]
