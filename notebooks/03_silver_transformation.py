"""
Silver transformation orchestrator - thin wrapper around src.transformations.silver_cleaner (schema bug fixed there).
"""

from src.transformations.silver_cleaner import build_silver

build_silver(spark, bronze_table=config.get_bronze_table_name(), silver_table=config.get_silver_table_name())
