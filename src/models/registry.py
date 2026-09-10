"""
Unity Catalog Model Registry registration. Verbatim from original cell 27. Uses the current UC registry API (mlflow.register_model against a 3-level name + MlflowClient.set_registered_model_alias) rather than the deprecated stages API - this was already correct in the source notebook.
"""

# Register Champion Model in Unity Catalog
import mlflow
from mlflow import MlflowClient

if champion_name:
    print("=" * 80)
    print("UNITY CATALOG MODEL REGISTRATION")
    print("=" * 80)
    
    client = MlflowClient()
    
    # Get champion model URI from MLflow run
    champion_run_id = comparison_df[comparison_df['is_champion']]['run_id'].values[0]
    model_uri = f"runs:/{champion_run_id}/model"
    
    # Register model to Unity Catalog
    registered_model_name = config.registered_model_name
    
    print(f"\nRegistering model...")
    print(f"  Model URI: {model_uri}")
    print(f"  UC Name: {registered_model_name}")
    
    # Register the model
    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=registered_model_name,
        tags={
            "project": "wind_turbine_ml",
            "domain": "wind_energy",
            "problem_type": "predictive_maintenance",
            "model_type": "classification",
            "environment": "production",
            "status": "champion",
            "algorithm": champion_name,
            "primary_metric": "pr_auc",
            "min_recall": str(config.minimum_recall)
        }
    )
    
    print(f"\n✓ Model registered successfully!")
    print(f"  Model Name: {registered_model_name}")
    print(f"  Version: {model_version.version}")
    
    # Set Champion alias
    client.set_registered_model_alias(
        name=registered_model_name,
        alias="Champion",
        version=model_version.version
    )
    
    print(f"  Alias: Champion")
    print(f"\n  Model URI for serving: models:/{registered_model_name}@Champion")
    
    print("=" * 80)
else:
    print("\n⚠️  Skipping registration - no qualified champion model")