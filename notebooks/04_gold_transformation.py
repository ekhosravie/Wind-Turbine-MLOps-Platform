"""
Gold transformation orchestrator - thin wrapper around src.transformations.gold_aggregator (schema bug fixed there).
"""

from src.transformations.gold_aggregator import build_daily_kpi, build_turbine_health, build_training_dataset

silver_table = config.get_silver_table_name()

build_daily_kpi(spark, silver_table, config.get_gold_kpi_table_name())
build_turbine_health(spark, silver_table, config.get_gold_health_table_name())
build_training_dataset(
    spark, silver_table, config.get_gold_training_table_name(),
    horizon_hours=config.target_horizon_hours,
    interval_minutes=config.sensor_interval_minutes,
)
