"""
Bronze ingestion orchestrator - thin wrapper around src.ingestion.bronze_loader (schema bug fixed there).
"""

from src.ingestion.bronze_loader import ingest_to_bronze

# `config` and `spark` are expected to already be in scope (Databricks notebook
# widget/context, or set explicitly if run as a script).
source_scada_table = f"{config.catalog}.bronze.source_turbine_sensor_scada"
bronze_table = config.get_bronze_table_name()

ingest_to_bronze(spark, source_table=source_scada_table, bronze_table=bronze_table)
