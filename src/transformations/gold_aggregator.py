"""
Gold layer: business KPIs, health scoring, and the ML training dataset.

No fix needed for turbine_daily_kpi / turbine_health (Gold 1 & 2) beyond
reading from Silver, which now persists the legacy alias columns
(power_output, rotor_speed, generator_speed, maintenance_flag) — see
src/common/schema.py and src/transformations/silver_cleaner.py.

FIX APPLIED to build_training_dataset (Gold 3): the final `.select(...)`
referenced `rotor_speed`, `generator_speed`, `oil_temperature`,
`power_output`, `active_power` — all resolvable now via the alias columns
Silver persists, EXCEPT note that `oil_temperature` and `power_output` are
derived approximations (see schema.py docstring), not raw sensors — treat
them accordingly if this becomes a real production feature set.
"""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F, Window


def build_daily_kpi(spark: SparkSession, silver_table: str, gold_kpi_table: str) -> DataFrame:
    silver_df = spark.table(silver_table)

    daily_kpi = (
        silver_df.filter(F.col("is_valid_record"))
        .groupBy("turbine_id", "farm_id", F.to_date("timestamp").alias("date"))
        .agg(
            F.avg("wind_speed").alias("average_wind_speed"),
            F.max("wind_speed").alias("max_wind_speed"),
            F.avg("power_output").alias("average_power"),
            F.sum("power_output").alias("total_energy"),
            F.avg("gearbox_temperature").alias("average_temperature"),
            F.max("gearbox_temperature").alias("max_gearbox_temperature"),
            F.avg(F.sqrt(F.col("vibration_x") ** 2 + F.col("vibration_y") ** 2 + F.col("vibration_z") ** 2)).alias("average_vibration"),
            F.sum("failure_flag").alias("failure_count"),
            F.sum("maintenance_flag").alias("maintenance_count"),
            F.count("*").alias("record_count"),
        )
    )

    daily_kpi = daily_kpi.withColumn(
        "availability", F.when(F.col("failure_count") == 0, 1.0).otherwise(0.8)
    ).withColumn(
        "capacity_factor", F.col("total_energy") / (2.5 * 24)
    )

    daily_kpi.write.format("delta").mode("overwrite").saveAsTable(gold_kpi_table)
    print(f"Created: {gold_kpi_table}")
    return daily_kpi


def build_turbine_health(spark: SparkSession, silver_table: str, gold_health_table: str) -> DataFrame:
    silver_df = spark.table(silver_table)

    health_df = silver_df.filter(F.col("is_valid_record")).withColumn(
        "health_score",
        F.lit(100)
        - F.when(F.col("gearbox_temperature") > 100, 15).otherwise(0)
        - F.when(F.col("generator_temperature") > 90, 10).otherwise(0)
        - F.when(F.col("bearing_temperature") > 80, 10).otherwise(0)
        - (F.col("vibration_x") * 20)
        - (F.col("vibration_y") * 20)
        - (F.col("vibration_z") * 15)
        - F.when((F.col("wind_speed") > 10) & (F.col("power_output") < 1.0), 10).otherwise(0),
    )

    health_df = health_df.withColumn(
        "health_score", F.greatest(F.col("health_score"), F.lit(0))
    ).withColumn(
        "health_status",
        F.when(F.col("health_score") >= 80, "HEALTHY")
        .when(F.col("health_score") >= 60, "WARNING")
        .otherwise("CRITICAL"),
    )

    health_df.write.format("delta").mode("overwrite").saveAsTable(gold_health_table)
    print(f"Created: {gold_health_table}")
    spark.sql(f"SELECT health_status, COUNT(*) as count FROM {gold_health_table} GROUP BY health_status").show()
    return health_df


def build_training_dataset(spark: SparkSession, silver_table: str, gold_training_table: str,
                            horizon_hours: int = 24, interval_minutes: int = 10) -> DataFrame:
    """Builds the failure_within_next_N_hours supervised dataset without
    leaking future information into features (see module docstring)."""
    silver_df = spark.table(silver_table)

    training_df = silver_df.withColumn("timestamp_epoch", F.unix_timestamp("timestamp"))

    failures = silver_df.filter(F.col("failure_flag") == 1).select(
        F.col("turbine_id").alias("fail_turbine_id"),
        F.unix_timestamp("timestamp").alias("failure_time"),
    )

    training_df = training_df.alias("t").join(
        failures.alias("f"),
        (F.col("t.turbine_id") == F.col("f.fail_turbine_id"))
        & (F.col("f.failure_time").between(
            F.col("t.timestamp_epoch"),
            F.col("t.timestamp_epoch") + horizon_hours * 3600,
        )),
        "left",
    )

    training_df = training_df.withColumn(
        "failure_within_next_24_hours",
        F.when(F.col("failure_time").isNotNull(), 1).otherwise(0),
    )

    training_df = training_df.dropDuplicates(["turbine_id", "timestamp"])
    # Exclude records already in failure state (prevent target leakage)
    training_df = training_df.filter(F.col("failure_flag") == 0)

    training_df = training_df.select(
        "turbine_id", "farm_id", "timestamp",
        "wind_speed", "wind_direction", "ambient_temperature",
        "rotor_speed", "generator_speed",
        "gearbox_temperature", "generator_temperature", "bearing_temperature",
        "nacelle_temperature", "oil_temperature",
        "hydraulic_pressure", "vibration_x", "vibration_y", "vibration_z",
        "power_output", "active_power", "blade_pitch_angle",
        "operating_hours", "quality_score",
        "failure_within_next_24_hours",
    )

    training_df.write.format("delta").mode("overwrite").saveAsTable(gold_training_table)
    print(f"Created: {gold_training_table}")
    spark.sql(f"""
        SELECT failure_within_next_24_hours as target, COUNT(*) as count,
               ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
        FROM {gold_training_table}
        GROUP BY failure_within_next_24_hours
    """).show()

    return training_df
