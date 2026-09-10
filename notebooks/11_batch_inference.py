"""
Batch inference: score data with the Champion model and write predictions. Verbatim from original cell 29. Relies on notebook-global state (results, champion_name, test_df, X_test) from 06/07.
"""

# Batch Inference: Score new turbine data
from datetime import datetime
import pandas as pd

if champion_name:
    print("=" * 80)
    print("BATCH INFERENCE PIPELINE")
    print("=" * 80)
    
    # Load champion model
    champion_model = results[champion_name]['model']
    
    # Score test set as example (in production, this would be new data)
    predictions_df = test_df[id_cols].copy()
    predictions_df['failure_probability'] = champion_model.predict_proba(X_test)[:, 1]
    predictions_df['prediction'] = champion_model.predict(X_test)
    predictions_df['risk_level'] = pd.cut(
        predictions_df['failure_probability'],
        bins=[0, 0.3, 0.7, 1.0],
        labels=['LOW', 'MEDIUM', 'HIGH']
    )
    predictions_df['model_version'] = "1"
    predictions_df['model_run_id'] = champion_run_id
    predictions_df['prediction_timestamp'] = datetime.now()
    
    # Save predictions to Gold layer
    prediction_table = config.get_prediction_table_name()
    predictions_spark_df = spark.createDataFrame(predictions_df)
    predictions_spark_df.write.format("delta").mode("overwrite").saveAsTable(prediction_table)
    
    print(f"\n✓ Batch inference complete")
    print(f"  Predictions saved to: {prediction_table}")
    print(f"  Total predictions: {len(predictions_df):,}")
    
    print(f"\nRisk Distribution:")
    print(predictions_df['risk_level'].value_counts())
    
    print(f"\nHigh-Risk Turbines (Top 10):")
    high_risk = predictions_df.nlargest(10, 'failure_probability')[['turbine_id', 'timestamp', 'failure_probability', 'risk_level']]
    print(high_risk.to_string(index=False))
    
    print("\n" + "=" * 80)
else:
    print("\n⚠️  Skipping inference - no champion model available")