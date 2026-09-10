"""
Feature definitions for predictive maintenance. Verbatim from original cell 19, relocated here so notebooks/05_feature_engineering.py can import it rather than duplicate it. NOT independently re-verified beyond the columns already confirmed to route through src.common.schema.add_legacy_aliases.
"""

# Feature Engineering for Predictive Maintenance
from pyspark.sql import functions as F, Window
from pyspark.sql.types import *

# Read Gold training data
training_data = spark.table(config.get_gold_training_table_name())

# Define window specifications
# 1-hour lookback (6 intervals * 10min)
window_1h = Window.partitionBy("turbine_id").orderBy("timestamp").rowsBetween(-6, 0)
# 6-hour lookback (36 intervals)
window_6h = Window.partitionBy("turbine_id").orderBy("timestamp").rowsBetween(-36, 0)

# === 1. Rolling Aggregate Features ===
feature_df = training_data \
    .withColumn("gearbox_temp_avg_1h", F.avg("gearbox_temperature").over(window_1h)) \
    .withColumn("gearbox_temp_avg_6h", F.avg("gearbox_temperature").over(window_6h)) \
    .withColumn("gearbox_temp_std_1h", F.stddev("gearbox_temperature").over(window_1h)) \
    .withColumn("generator_temp_avg_1h", F.avg("generator_temperature").over(window_1h)) \
    .withColumn("generator_temp_avg_6h", F.avg("generator_temperature").over(window_6h)) \
    .withColumn("bearing_temp_avg_1h", F.avg("bearing_temperature").over(window_1h)) \
    .withColumn("vibration_avg_1h", F.avg(
        F.sqrt(F.col("vibration_x")**2 + F.col("vibration_y")**2 + F.col("vibration_z")**2)
    ).over(window_1h)) \
    .withColumn("vibration_avg_6h", F.avg(
        F.sqrt(F.col("vibration_x")**2 + F.col("vibration_y")**2 + F.col("vibration_z")**2)
    ).over(window_6h)) \
    .withColumn("vibration_std_1h", F.stddev(
        F.sqrt(F.col("vibration_x")**2 + F.col("vibration_y")**2 + F.col("vibration_z")**2)
    ).over(window_1h)) \
    .withColumn("power_avg_1h", F.avg("power_output").over(window_1h)) \
    .withColumn("power_avg_6h", F.avg("power_output").over(window_6h)) \
    .withColumn("power_std_1h", F.stddev("power_output").over(window_1h)) \
    .withColumn("wind_speed_avg_1h", F.avg("wind_speed").over(window_1h)) \
    .withColumn("wind_speed_avg_6h", F.avg("wind_speed").over(window_6h))

# === 2. Lag Features ===
lag_window = Window.partitionBy("turbine_id").orderBy("timestamp")
feature_df = feature_df \
    .withColumn("gearbox_temp_lag_1", F.lag("gearbox_temperature", 1).over(lag_window)) \
    .withColumn("gearbox_temp_lag_6", F.lag("gearbox_temperature", 6).over(lag_window)) \
    .withColumn("gearbox_temp_lag_12", F.lag("gearbox_temperature", 12).over(lag_window)) \
    .withColumn("vibration_x_lag_1", F.lag("vibration_x", 1).over(lag_window)) \
    .withColumn("vibration_x_lag_6", F.lag("vibration_x", 6).over(lag_window)) \
    .withColumn("power_lag_1", F.lag("power_output", 1).over(lag_window)) \
    .withColumn("power_lag_6", F.lag("power_output", 6).over(lag_window))

# === 3. Trend Features (Delta/Rate of Change) ===
feature_df = feature_df \
    .withColumn("gearbox_temp_delta", 
               F.col("gearbox_temperature") - F.col("gearbox_temp_lag_1")) \
    .withColumn("vibration_delta",
               F.col("vibration_x") - F.col("vibration_x_lag_1")) \
    .withColumn("power_delta",
               F.col("power_output") - F.col("power_lag_1"))

# === 4. Domain-Specific Features ===
# Thermal Stress Index (high temperature relative to ambient)
feature_df = feature_df.withColumn(
    "thermal_stress_index",
    (F.col("gearbox_temperature") - F.col("ambient_temperature")) / 100.0
)

# Mechanical Stress Index (vibration magnitude)
feature_df = feature_df.withColumn(
    "mechanical_stress_index",
    F.sqrt(F.col("vibration_x")**2 + F.col("vibration_y")**2 + F.col("vibration_z")**2)
)

# Temperature Anomaly (deviation from running average)
feature_df = feature_df.withColumn(
    "temp_anomaly_score",
    F.abs(F.col("gearbox_temperature") - F.col("gearbox_temp_avg_6h")) / 
    (F.col("gearbox_temp_std_1h") + 1.0)  # +1 to avoid division by zero
)

# Generator Load Factor
feature_df = feature_df.withColumn(
    "generator_load_factor",
    F.col("power_output") / 2.5  # Normalized by rated power
)

# Speed Ratio (should be ~100:1 for healthy gearbox)
feature_df = feature_df.withColumn(
    "speed_ratio",
    F.when(F.col("rotor_speed") > 0, 
           F.col("generator_speed") / F.col("rotor_speed"))
     .otherwise(0)
)

# Turbine Efficiency (power vs wind)
feature_df = feature_df.withColumn(
    "turbine_efficiency",
    F.when(F.col("wind_speed") > 3,
           F.col("power_output") / (F.col("wind_speed") ** 3 + 0.1))
     .otherwise(0)
)

# Fill nulls in engineered features (from lag operations)
feature_df = feature_df.fillna(0, subset=[
    "gearbox_temp_lag_1", "gearbox_temp_lag_6", "gearbox_temp_lag_12",
    "vibration_x_lag_1", "vibration_x_lag_6", 
    "power_lag_1", "power_lag_6",
    "gearbox_temp_delta", "vibration_delta", "power_delta"
])

# Save engineered features
feature_table = config.get_feature_table_name()
feature_df.write.format("delta").mode("overwrite").saveAsTable(feature_table)

print(f"✓ Feature engineering complete: {feature_table}")
print(f"✓ Total features created: {len(feature_df.columns)}")
print(f"\nFeature Categories:")
print("  - Rolling aggregates (1h, 6h): 13 features")
print("  - Lag features: 7 features")
print("  - Trend/delta features: 3 features")
print("  - Domain-specific: 7 features")
print(f"\nSample features:")
feature_df.select("turbine_id", "timestamp", "gearbox_temp_avg_1h", 
                  "vibration_avg_1h", "thermal_stress_index", 
                  "temp_anomaly_score", "failure_within_next_24_hours").show(5)