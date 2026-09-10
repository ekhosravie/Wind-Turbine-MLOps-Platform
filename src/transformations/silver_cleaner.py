"""
Silver layer: cleaning, validation, quality scoring.

FIX APPLIED vs. the original notebook cell: this logic referenced
`power_output`, `rotor_speed`, `generator_speed` — columns that don't exist
in Bronze (see src/common/schema.py for why). The quality-check and fillna
logic itself is unchanged; the only fix is calling `add_legacy_aliases()`
on the Bronze dataframe before doing anything else, so those column
references now resolve.
"""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.common.schema import add_legacy_aliases


def apply_quality_checks(df: DataFrame) -> DataFrame:
    """Apply data quality checks and create quality flags.
    (Unchanged from the original notebook cell.)
    """
    df = df.withColumn(
        "is_wind_speed_valid",
        (F.col("wind_speed").between(0, 50)) | F.col("wind_speed").isNull(),
    )
    df = df.withColumn(
        "is_temperature_valid",
        F.col("gearbox_temperature").between(-50, 150) | F.col("gearbox_temperature").isNull(),
    )
    df = df.withColumn(
        "is_power_valid",
        (F.col("power_output").between(0, 3.0)) | F.col("power_output").isNull(),
    )
    df = df.withColumn(
        "is_vibration_valid",
        (F.col("vibration_x").between(0, 2.0)) | F.col("vibration_x").isNull(),
    )

    df = df.withColumn(
        "is_temp_outlier",
        F.when(
            (F.col("gearbox_temperature") > 120) | (F.col("generator_temperature") > 110),
            True,
        ).otherwise(False),
    )
    df = df.withColumn(
        "is_vibration_outlier",
        F.when(
            (F.col("vibration_x") > 1.5) | (F.col("vibration_y") > 1.5) | (F.col("vibration_z") > 1.2),
            True,
        ).otherwise(False),
    )

    critical_cols = ["wind_speed", "power_output", "gearbox_temperature",
                      "generator_temperature", "rotor_speed"]
    df = df.withColumn(
        "missing_count",
        sum([F.when(F.col(c).isNull(), 1).otherwise(0) for c in critical_cols]),
    )
    df = df.withColumn("has_missing", F.col("missing_count") > 0)

    df = df.withColumn(
        "is_sensor_anomaly",
        (~F.col("is_wind_speed_valid"))
        | (~F.col("is_temperature_valid"))
        | (~F.col("is_power_valid"))
        | F.col("is_temp_outlier")
        | F.col("is_vibration_outlier"),
    )

    df = df.withColumn(
        "quality_score",
        F.lit(100)
        - (F.when(~F.col("is_wind_speed_valid"), 20).otherwise(0))
        - (F.when(~F.col("is_temperature_valid"), 20).otherwise(0))
        - (F.when(~F.col("is_power_valid"), 15).otherwise(0))
        - (F.when(F.col("is_temp_outlier"), 15).otherwise(0))
        - (F.when(F.col("is_vibration_outlier"), 10).otherwise(0))
        - (F.col("missing_count") * 4),
    )
    df = df.withColumn("quality_score", F.greatest(F.col("quality_score"), F.lit(0)))

    df = df.withColumn(
        "is_valid_record",
        F.col("is_wind_speed_valid")
        & F.col("is_temperature_valid")
        & F.col("is_power_valid")
        & (F.col("quality_score") >= 60),
    )
    return df


def build_silver(spark: SparkSession, bronze_table: str, silver_table: str) -> DataFrame:
    bronze_df = add_legacy_aliases(spark.table(bronze_table))

    silver_df = apply_quality_checks(bronze_df)
    silver_df = silver_df.fillna({
        "wind_speed": 0,
        "power_output": 0,
        "gearbox_temperature": 50,
        "generator_temperature": 45,
        "bearing_temperature": 40,
        "vibration_x": 0.3,
        "vibration_y": 0.3,
        "vibration_z": 0.2,
        "rotor_speed": 0,
        "generator_speed": 0,
    })

    silver_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(silver_table)

    print(f"Silver layer created: {silver_table}")
    spark.sql(f"""
        SELECT
            COUNT(*) as total_records,
            SUM(CAST(is_valid_record AS INT)) as valid_records,
            AVG(quality_score) as avg_quality_score,
            SUM(CAST(is_sensor_anomaly AS INT)) as anomaly_count,
            SUM(CAST(has_missing AS INT)) as records_with_missing
        FROM {silver_table}
    """).show()

    return silver_df
