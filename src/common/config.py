"""
Central project configuration. Verbatim from the original notebook's
config cell - already a clean dataclass, no fix needed.
"""

# Configuration Management
# Central configuration for the entire MLOps pipeline

import os
from dataclasses import dataclass
from typing import List, Dict, Any
from datetime import datetime, timedelta

@dataclass
class ProjectConfig:
    """Central configuration for Wind Turbine MLOps project."""
    
    # Unity Catalog Configuration
    catalog: str = "wind_turbine_ml"
    bronze_schema: str = "bronze"
    silver_schema: str = "silver"
    gold_schema: str = "gold"
    features_schema: str = "features"
    ml_schema: str = "ml"
    monitoring_schema: str = "monitoring"
    
    # Data Generation - Enhanced Configuration
    num_turbines: int = 60  # Total turbines across all farms
    turbines_per_farm: int = 20  # Turbines per farm
    num_farms: int = 3  # Number of wind farms
    data_months: int = 12  # Months of historical data
    sensor_interval_minutes: int = 10  # Sensor measurement frequency
    random_seed: int = 42  # Reproducibility seed
    
    # Data Generation - Date Range
    start_date: str = "2025-01-01"
    end_date: str = "2025-12-31"
    
    # Data Generation - Failure & Maintenance Rates
    failure_rate: float = 0.05  # Base failure probability
    maintenance_rate: float = 0.10  # Maintenance event rate
    degradation_window_hours: int = 72  # Hours of pre-failure degradation
    prediction_horizon_hours: int = 24  # Target prediction horizon
    
    # Data Quality Issues Rates
    duplicate_rate: float = 0.01  # Duplicate record rate
    missing_rate: float = 0.02  # Missing data rate
    outlier_rate: float = 0.005  # Outlier rate
    sensor_drift_rate: float = 0.01  # Sensor drift rate
    
    # Storage Paths
    source_path: str = "/Volumes/wind_turbine_ml/bronze/raw_data"
    
    # Turbine Characteristics
    rated_power_range: tuple = (2000, 5000)  # kW
    rotor_diameter_range: tuple = (120, 180)  # meters
    hub_height_range: tuple = (80, 140)  # meters
    
    # Table Names
    bronze_table: str = "turbine_sensor_raw"
    silver_table: str = "turbine_sensor_clean"
    gold_kpi_table: str = "turbine_daily_kpi"
    gold_health_table: str = "turbine_health"
    gold_training_table: str = "turbine_failure_training"
    feature_table: str = "turbine_failure_features"
    model_comparison_table: str = "model_comparison"
    prediction_table: str = "turbine_failure_predictions"
    
    # ML Training
    target_horizon_hours: int = 24
    train_split: float = 0.70
    val_split: float = 0.15
    test_split: float = 0.15
    
    # Model Selection
    primary_metric: str = "pr_auc"
    minimum_recall: float = 0.80
    
    # MLflow
    experiment_name: str = "/Shared/wind-turbine-failure-prediction"
    registered_model_name: str = "wind_turbine_ml.ml.turbine_failure_model"
    
    # Model Serving
    serving_endpoint_name: str = "wind-turbine-failure-predictor"
    
    # Monitoring
    drift_threshold: float = 0.05
    quality_threshold: float = 0.90
    
    def get_table_name(self, schema: str, table: str) -> str:
        """Get fully qualified table name."""
        return f"{self.catalog}.{schema}.{table}"
    
    def get_bronze_table_name(self) -> str:
        return self.get_table_name(self.bronze_schema, self.bronze_table)
    
    def get_silver_table_name(self) -> str:
        return self.get_table_name(self.silver_schema, self.silver_table)
    
    def get_gold_kpi_table_name(self) -> str:
        return self.get_table_name(self.gold_schema, self.gold_kpi_table)
    
    def get_gold_health_table_name(self) -> str:
        return self.get_table_name(self.gold_schema, self.gold_health_table)
    
    def get_gold_training_table_name(self) -> str:
        return self.get_table_name(self.gold_schema, self.gold_training_table)
    
    def get_feature_table_name(self) -> str:
        return self.get_table_name(self.features_schema, self.feature_table)
    
    def get_model_comparison_table_name(self) -> str:
        return self.get_table_name(self.ml_schema, self.model_comparison_table)
    
    def get_prediction_table_name(self) -> str:
        return self.get_table_name(self.gold_schema, self.prediction_table)

# Default instance for convenience; override per-environment in notebooks
# via ProjectConfig(**yaml.safe_load(open(f"configs/{env}.yml"))).
config = ProjectConfig()

if __name__ == "__main__":
    print("Configuration loaded successfully")
    print(f"\nCatalog: {config.catalog}")
    print(f"Bronze Table: {config.get_bronze_table_name()}")
    print(f"Silver Table: {config.get_silver_table_name()}")
    print(f"Gold Training Table: {config.get_gold_training_table_name()}")
    print(f"Registered Model: {config.registered_model_name}")