"""
Bronze layer ingestion.

FIXES APPLIED vs. the original notebook cell:
1. The CREATE TABLE DDL used column names from an earlier draft of the
   generator (`rotor_speed`, `power_output`, `oil_temperature`, `current`,
   `voltage`, `maintenance_flag`...) that do not exist in the current
   generator output. DDL now matches src.common.schema.RAW_SCHEMA exactly.
2. `source_df = spark.table(temp_source_table)` referenced a variable that
   was never defined anywhere in the notebook (a NameError on execution).
   The data generation step actually saves to `source_scada_table` —
   fixed to take that as an explicit parameter instead of relying on
   notebook global state.
"""

import uuid

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from src.common.schema import RAW_SCHEMA


def _ddl_column_list() -> str:
    # All real generator columns are DOUBLE/STRING/TIMESTAMP/BOOLEAN in
    # practice except the identifiers/flags below; Spark's schema-on-write
    # with mergeSchema off will fail loudly (which is what we want) if a
    # future generator change adds/removes a column, rather than silently
    # dropping data.
    string_cols = {
        "event_id", "turbine_id", "farm_id", "turbine_model",
        "operating_state", "alarm_code", "alarm_severity", "failure_type",
    }
    bool_cols = {
        "availability_flag", "grid_connected_flag", "maintenance_mode_flag",
        "alarm_flag", "failure_flag",
    }
    timestamp_cols = {"timestamp"}

    lines = []
    for col in RAW_SCHEMA:
        if col in string_cols:
            dtype = "STRING"
        elif col in bool_cols:
            dtype = "BOOLEAN"
        elif col in timestamp_cols:
            dtype = "TIMESTAMP"
        else:
            dtype = "DOUBLE"
        lines.append(f"{col} {dtype}")
    return ",\n        ".join(lines)


def create_bronze_table(spark: SparkSession, bronze_table: str) -> None:
    ddl_columns = _ddl_column_list()
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {bronze_table} (
        {ddl_columns},
        ingestion_timestamp TIMESTAMP,
        ingestion_date DATE,
        batch_id STRING,
        source_file STRING
        ) USING DELTA PARTITIONED BY (ingestion_date)
        TBLPROPERTIES ('delta.enableChangeDataFeed'='true')
        COMMENT 'Bronze: Raw turbine sensor data with ingestion metadata'
    """)


def ingest_to_bronze(spark: SparkSession, source_table: str, bronze_table: str) -> DataFrame:
    """Idempotent, incremental Bronze ingestion via MERGE.

    source_table: fully-qualified table the generator actually wrote to
        (e.g. f"{config.catalog}.bronze.source_turbine_sensor_scada").
        This was the undefined `temp_source_table` in the original cell.
    """
    create_bronze_table(spark, bronze_table)

    batch_id = str(uuid.uuid4())
    source_df = spark.table(source_table)

    bronze_df = (
        source_df
        .withColumn("ingestion_timestamp", F.current_timestamp())
        .withColumn("ingestion_date", F.current_date())
        .withColumn("batch_id", F.lit(batch_id))
        .withColumn("source_file", F.lit(source_table))
    )

    window = Window.partitionBy("turbine_id", "timestamp").orderBy(F.desc("timestamp"))
    bronze_df = (
        bronze_df.withColumn("rn", F.row_number().over(window))
        .filter(F.col("rn") == 1)
        .drop("rn")
    )

    bronze_df.createOrReplaceTempView("bronze_updates")
    spark.sql(f"""
        MERGE INTO {bronze_table} t USING bronze_updates s
        ON t.turbine_id = s.turbine_id AND t.timestamp = s.timestamp
        WHEN NOT MATCHED THEN INSERT *
    """)

    result = spark.table(bronze_table)
    print(f"Bronze layer updated: {bronze_table} ({result.count():,} total records)")
    return result
