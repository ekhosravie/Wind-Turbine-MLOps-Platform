"""
Time-series train/val/test split + multi-algorithm training with MLflow tracking. Verbatim from original cells 21+22+23. Relies on notebook-global state (feature_df from the previous stage) rather than explicit function arguments - this is sequential-notebook style, not a tested importable module. Recommend refactoring into explicit functions (e.g. split_data(df, config) -> (train, val, test), train_all_models(...) -> results) before this is treated as production code.
"""

# Time-Series Aware Train/Validation/Test Split
# CRITICAL: Use temporal split, NOT random split, to prevent future information leakage

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Reload config if needed after restart
try:
    config
except:
    from dataclasses import dataclass
    @dataclass
    class ProjectConfig:
        catalog: str = "wind_turbine_ml"
        bronze_schema: str = "bronze"
        silver_schema: str = "silver"
        gold_schema: str = "gold"
        features_schema: str = "features"
        ml_schema: str = "ml"
        monitoring_schema: str = "monitoring"
        bronze_table: str = "turbine_sensor_raw"
        silver_table: str = "turbine_sensor_clean"
        gold_kpi_table: str = "turbine_daily_kpi"
        gold_health_table: str = "turbine_health"
        gold_training_table: str = "turbine_failure_training"
        feature_table: str = "turbine_failure_features"
        model_comparison_table: str = "model_comparison"
        prediction_table: str = "turbine_failure_predictions"
        experiment_name: str = "/Shared/wind-turbine-failure-prediction"
        registered_model_name: str = "wind_turbine_ml.ml.turbine_failure_model"
        train_split: float = 0.70
        val_split: float = 0.15
        test_split: float = 0.15
        primary_metric: str = "pr_auc"
        minimum_recall: float = 0.80
        
        def get_table_name(self, schema, table):
            return f"{self.catalog}.{schema}.{table}"
        def get_feature_table_name(self):
            return self.get_table_name(self.features_schema, self.feature_table)
        def get_model_comparison_table_name(self):
            return self.get_table_name(self.ml_schema, self.model_comparison_table)
    
    config = ProjectConfig()

# Load feature data
feature_df = spark.table(config.get_feature_table_name()).toPandas()

print(f"Total records: {len(feature_df):,}")
print(f"Target distribution:")
print(feature_df['failure_within_next_24_hours'].value_counts())

# Sort by timestamp for temporal split
feature_df = feature_df.sort_values('timestamp').reset_index(drop=True)

# Calculate split indices
n = len(feature_df)
train_end = int(n * config.train_split)
val_end = int(n * (config.train_split + config.val_split))

# Temporal split
train_df = feature_df.iloc[:train_end]
val_df = feature_df.iloc[train_end:val_end]
test_df = feature_df.iloc[val_end:]

print(f"\n✓ Time-Series Split:")
print(f"  Train: {len(train_df):,} records ({config.train_split*100:.0f}%) - {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
print(f"  Val:   {len(val_df):,} records ({config.val_split*100:.0f}%) - {val_df['timestamp'].min()} to {val_df['timestamp'].max()}")
print(f"  Test:  {len(test_df):,} records ({config.test_split*100:.0f}%) - {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")

print(f"\n⚠️  Why Temporal Split?")
print("  • Turbine sensor data has temporal dependencies")
print("  • Random split would leak future information into training")
print("  • Temporal split simulates real production: train on past, predict future")
print("  • Each dataset contains sequential time periods with no overlap")

# Define feature columns (exclude ID, timestamp, and target)
id_cols = ['turbine_id', 'farm_id', 'timestamp']
target_col = 'failure_within_next_24_hours'
feature_cols = [c for c in feature_df.columns if c not in id_cols + [target_col]]

print(f"\n✓ Feature columns: {len(feature_cols)}")
print(f"\nTarget distribution by split:")
for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
    pos = (df[target_col] == 1).sum()
    neg = (df[target_col] == 0).sum()
    print(f"  {name}: Failures={pos:,} ({pos/len(df)*100:.2f}%), Normal={neg:,} ({neg/len(df)*100:.2f}%)")

# Prepare X, y for each split
X_train = train_df[feature_cols]
y_train = train_df[target_col]

X_val = val_df[feature_cols]
y_val = val_df[target_col]

X_test = test_df[feature_cols]
y_test = test_df[target_col]

print(f"\n✓ Data prepared for model training")
print(f"  X_train shape: {X_train.shape}")
print(f"  X_val shape: {X_val.shape}")
print(f"  X_test shape: {X_test.shape}")
# Train Multiple ML Algorithms with MLflow Tracking
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Set MLflow experiment
mlflow.set_experiment(config.experiment_name)

def train_and_log_model(model_name, model, X_train, y_train, X_val, y_val, hyperparams):
    """
    Train model and log to MLflow with comprehensive metrics.
    
    For imbalanced failure prediction, prioritize:
    - Recall (minimize missed failures)
    - PR-AUC (best for imbalanced data)
    - F1 (balance precision/recall)
    """
    
    with mlflow.start_run(run_name=model_name) as run:
        # Log parameters
        mlflow.log_params(hyperparams)
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("val_size", len(X_val))
        mlflow.log_param("n_features", X_train.shape[1])
        
        # Create pipeline with preprocessing
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        
        # Train
        pipeline.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = pipeline.predict(X_train)
        y_val_pred = pipeline.predict(X_val)
        y_val_proba = pipeline.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'val_accuracy': accuracy_score(y_val, y_val_pred),
            'val_precision': precision_score(y_val, y_val_pred, zero_division=0),
            'val_recall': recall_score(y_val, y_val_pred),
            'val_f1': f1_score(y_val, y_val_pred),
            'val_roc_auc': roc_auc_score(y_val, y_val_proba),
            'val_pr_auc': average_precision_score(y_val, y_val_proba)
        }
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Confusion matrix
        cm = confusion_matrix(y_val, y_val_pred)
        tn, fp, fn, tp = cm.ravel()
        mlflow.log_metrics({
            'val_true_negatives': int(tn),
            'val_false_positives': int(fp),
            'val_false_negatives': int(fn),
            'val_true_positives': int(tp)
        })
        
        # Log model with signature
        from mlflow.models import infer_signature
        signature = infer_signature(X_train, pipeline.predict(X_train))
        mlflow.sklearn.log_model(
            pipeline, 
            "model",
            signature=signature,
            input_example=X_train.iloc[:3]
        )
        
        # Feature importance (if available)
        if hasattr(pipeline.named_steps['model'], 'feature_importances_'):
            importances = pipeline.named_steps['model'].feature_importances_
            feature_importance = sorted(zip(X_train.columns, importances), 
                                       key=lambda x: x[1], reverse=True)[:10]
            print(f"\nTop 10 Features for {model_name}:")
            for feat, imp in feature_importance:
                print(f"  {feat}: {imp:.4f}")
        
        print(f"\n✓ {model_name} Training Complete")
        print(f"  Run ID: {run.info.run_id}")
        print(f"  Metrics:")
        print(f"    Recall:    {metrics['val_recall']:.4f} (PRIMARY for predictive maintenance)")
        print(f"    PR-AUC:    {metrics['val_pr_auc']:.4f} (PRIMARY for imbalanced data)")
        print(f"    F1:        {metrics['val_f1']:.4f}")
        print(f"    Precision: {metrics['val_precision']:.4f}")
        print(f"    ROC-AUC:   {metrics['val_roc_auc']:.4f}")
        print(f"    Accuracy:  {metrics['val_accuracy']:.4f}")
        print(f"  Confusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
        
        return run.info.run_id, metrics, pipeline

print("=" * 80)
print("TRAINING MULTIPLE ALGORITHMS")
print("=" * 80)

results = {}

# 1. Logistic Regression (Baseline)
print("\n[1/4] Training Logistic Regression...")
lr_params = {'C': 1.0, 'max_iter': 1000, 'class_weight': 'balanced'}
lr = LogisticRegression(**lr_params, random_state=42)
run_id, metrics, model = train_and_log_model("Logistic_Regression", lr, X_train, y_train, X_val, y_val, lr_params)
results['Logistic_Regression'] = {'run_id': run_id, 'metrics': metrics, 'model': model}

# 2. Random Forest
print("\n[2/4] Training Random Forest...")
rf_params = {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 10, 'class_weight': 'balanced'}
rf = RandomForestClassifier(**rf_params, random_state=42, n_jobs=-1)
run_id, metrics, model = train_and_log_model("Random_Forest", rf, X_train, y_train, X_val, y_val, rf_params)
results['Random_Forest'] = {'run_id': run_id, 'metrics': metrics, 'model': model}

# 3. XGBoost
print("\n[3/4] Training XGBoost...")
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb_params = {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'scale_pos_weight': scale_pos_weight}
xgb = XGBClassifier(**xgb_params, random_state=42, eval_metric='logloss')
run_id, metrics, model = train_and_log_model("XGBoost", xgb, X_train, y_train, X_val, y_val, xgb_params)
results['XGBoost'] = {'run_id': run_id, 'metrics': metrics, 'model': model}

# 4. LightGBM
print("\n[4/4] Training LightGBM...")
lgbm_params = {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'class_weight': 'balanced'}
lgbm = LGBMClassifier(**lgbm_params, random_state=42, verbose=-1)
run_id, metrics, model = train_and_log_model("LightGBM", lgbm, X_train, y_train, X_val, y_val, lgbm_params)
results['LightGBM'] = {'run_id': run_id, 'metrics': metrics, 'model': model}

print("\n" + "=" * 80)
print("✓ ALL MODELS TRAINED SUCCESSFULLY")
print("=" * 80)