"""
Model comparison and Champion selection against the configured minimum-recall gate. Verbatim from original cell 25. Relies on the results dict produced by 06_model_training.py being in scope.
"""

# Model Comparison and Champion Selection
import pandas as pd
from datetime import datetime

print("=" * 80)
print("MODEL COMPARISON & CHAMPION SELECTION")
print("=" * 80)

# Build comparison table
comparison_data = []
for model_name, result in results.items():
    metrics = result['metrics']
    comparison_data.append({
        'model_name': model_name,
        'run_id': result['run_id'],
        'accuracy': round(metrics['val_accuracy'], 4),
        'precision': round(metrics['val_precision'], 4),
        'recall': round(metrics['val_recall'], 4),
        'f1': round(metrics['val_f1'], 4),
        'roc_auc': round(metrics['val_roc_auc'], 4),
        'pr_auc': round(metrics['val_pr_auc'], 4),
        'training_timestamp': datetime.now()
    })

comparison_df = pd.DataFrame(comparison_data)

# Sort by PR-AUC (primary metric for imbalanced data)
comparison_df = comparison_df.sort_values('pr_auc', ascending=False)

print("\n✓ Model Comparison (sorted by PR-AUC):")
print(comparison_df[['model_name', 'recall', 'pr_auc', 'f1', 'precision', 'roc_auc', 'accuracy']].to_string(index=False))

# Champion Selection Logic
min_recall = config.minimum_recall

print(f"\n{'='*80}")
print(f"CHAMPION SELECTION CRITERIA")
print(f"{'='*80}")
print(f"1. Minimum Recall: {min_recall}")
print(f"2. Primary Metric: PR-AUC (best for imbalanced data)")
print(f"3. Tiebreaker: F1 Score")

# Filter models meeting recall threshold
qualified_models = comparison_df[comparison_df['recall'] >= min_recall]

if len(qualified_models) == 0:
    print(f"\n⚠️  NO QUALIFIED CHAMPION")
    print(f"   No model achieves minimum recall of {min_recall}")
    print(f"   Best recall: {comparison_df['recall'].max():.4f}")
    print(f"   ACTION REQUIRED: Retrain with adjusted hyperparameters or more data")
    champion_name = None
else:
    # Select champion: highest PR-AUC among qualified models
    champion_row = qualified_models.iloc[0]
    champion_name = champion_row['model_name']
    champion_run_id = champion_row['run_id']
    
    print(f"\n✓ CHAMPION MODEL SELECTED: {champion_name}")
    print(f"  Run ID: {champion_run_id}")
    print(f"  Metrics:")
    print(f"    Recall:    {champion_row['recall']:.4f} ✓ (meets minimum {min_recall})")
    print(f"    PR-AUC:    {champion_row['pr_auc']:.4f} (HIGHEST among qualified)")
    print(f"    F1:        {champion_row['f1']:.4f}")
    print(f"    Precision: {champion_row['precision']:.4f}")
    print(f"    ROC-AUC:   {champion_row['roc_auc']:.4f}")
    
    # Test set evaluation
    champion_model = results[champion_name]['model']
    y_test_pred = champion_model.predict(X_test)
    y_test_proba = champion_model.predict_proba(X_test)[:, 1]
    
    test_metrics = {
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'test_precision': precision_score(y_test, y_test_pred, zero_division=0),
        'test_recall': recall_score(y_test, y_test_pred),
        'test_f1': f1_score(y_test, y_test_pred),
        'test_roc_auc': roc_auc_score(y_test, y_test_proba),
        'test_pr_auc': average_precision_score(y_test, y_test_proba)
    }
    
    print(f"\n  Test Set Performance:")
    print(f"    Recall:    {test_metrics['test_recall']:.4f}")
    print(f"    PR-AUC:    {test_metrics['test_pr_auc']:.4f}")
    print(f"    F1:        {test_metrics['test_f1']:.4f}")
    print(f"    Precision: {test_metrics['test_precision']:.4f}")
    
    # Save champion info
    comparison_df['is_champion'] = comparison_df['model_name'] == champion_name

# Save comparison table to Unity Catalog
comparison_table = config.get_model_comparison_table_name()
comparison_spark_df = spark.createDataFrame(comparison_df)
comparison_spark_df.write.format("delta").mode("overwrite").saveAsTable(comparison_table)

print(f"\n✓ Model comparison saved to: {comparison_table}")
print("=" * 80)