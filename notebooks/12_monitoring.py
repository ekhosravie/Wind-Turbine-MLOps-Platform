"""
Drift + monitoring orchestration - glue code that calls the pure functions in src.monitoring.drift_detector and writes results to the monitoring schema. Verbatim glue from original cell 30 (the reusable functions themselves were extracted to src/monitoring/drift_detector.py). Relies on notebook-global state (train_baseline, results, champion_name, X_val/y_val, X_test/y_test) from earlier stages.
"""

# ======================================================================================
# COMPREHENSIVE DRIFT DETECTION & MONITORING
# Implements 4 types of drift: Data, Concept, Prediction, and Model
# ======================================================================================

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pyspark.sql import functions as F
from scipy.stats import ks_2samp, chi2_contingency
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("DRIFT DETECTION & MONITORING FRAMEWORK")
print("="*80)

# Create monitoring schema if not exists
monitoring_schema = config.monitoring_schema
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {config.catalog}.{monitoring_schema}")

# ======================================================================================
# 1. DATA DRIFT DETECTION
# Monitors changes in feature distributions between training and production
# ======================================================================================

print("\n[1/4] DATA DRIFT DETECTION")
print("-" * 80)

def calculate_psi(expected, actual, bins=10):
    """
    Calculate Population Stability Index (PSI).
    PSI < 0.1: No significant shift
    PSI 0.1-0.2: Moderate shift (investigate)
    PSI > 0.2: Significant shift (retrain)
    """
    def calculate_psi_single(expected_array, actual_array, bins):
        breakpoints = np.linspace(0, 100, bins + 1)
        expected_percents = np.percentile(expected_array, breakpoints)
        
        expected_counts = np.histogram(expected_array, expected_percents)[0]
        actual_counts = np.histogram(actual_array, expected_percents)[0]
        
        expected_percents = expected_counts / len(expected_array)
        actual_percents = actual_counts / len(actual_array)
        
        # Add small epsilon to avoid log(0)
        expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
        actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
        
        psi_value = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
        return psi_value
    
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
    return calculate_psi_single(expected, actual, bins)

def detect_data_drift(train_df, prod_df, feature_cols, threshold_psi=0.2, threshold_ks=0.05):
    """
    Detect data drift using KS test and PSI.
    """
    drift_results = []
    
    for feature in feature_cols:
        # Skip if feature doesn't exist in both datasets
        if feature not in train_df.columns or feature not in prod_df.columns:
            continue
        
        train_values = train_df[feature].dropna()
        prod_values = prod_df[feature].dropna()
        
        if len(train_values) == 0 or len(prod_values) == 0:
            continue
        
        # Kolmogorov-Smirnov test
        ks_statistic, ks_pvalue = ks_2samp(train_values, prod_values)
        
        # Population Stability Index
        psi_value = calculate_psi(train_values.values, prod_values.values)
        
        # Determine drift status
        drift_detected = (ks_pvalue < threshold_ks) or (psi_value > threshold_psi)
        drift_severity = "CRITICAL" if psi_value > threshold_psi else ("WARNING" if ks_pvalue < threshold_ks else "OK")
        
        drift_results.append({
            'feature': feature,
            'ks_statistic': ks_statistic,
            'ks_pvalue': ks_pvalue,
            'psi_value': psi_value,
            'drift_detected': drift_detected,
            'drift_severity': drift_severity,
            'train_mean': float(train_values.mean()),
            'prod_mean': float(prod_values.mean()),
            'train_std': float(train_values.std()),
            'prod_std': float(prod_values.std()),
            'check_timestamp': datetime.now()
        })
    
    return pd.DataFrame(drift_results)

# Load training data baseline
try:
    train_baseline = spark.table(config.get_feature_table_name()).toPandas()
    
    # Simulate production data (in production, this would be latest feature data)
    # For demo: sample recent data as "production"
    prod_data = train_baseline.tail(int(len(train_baseline) * 0.1))  # Last 10% as "production"
    
    # Select numeric features for drift detection
    numeric_features = train_baseline.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_features if c not in ['failure_within_next_24_hours']]
    
    # Run data drift detection
    data_drift_results = detect_data_drift(
        train_baseline[feature_cols],
        prod_data[feature_cols],
        feature_cols,
        threshold_psi=0.2,
        threshold_ks=0.05
    )
    
    # Save drift results
    drift_table = f"{config.catalog}.{monitoring_schema}.data_drift_monitoring"
    drift_spark_df = spark.createDataFrame(data_drift_results)
    drift_spark_df.write.format("delta").mode("overwrite").saveAsTable(drift_table)
    
    print(f"✓ Data drift monitoring complete: {drift_table}")
    print(f"\nDrift Summary:")
    print(f"  Total features monitored: {len(data_drift_results)}")
    print(f"  Features with drift detected: {data_drift_results['drift_detected'].sum()}")
    print(f"  Critical drift (PSI > 0.2): {(data_drift_results['drift_severity'] == 'CRITICAL').sum()}")
    
    # Show top drifted features
    if data_drift_results['drift_detected'].sum() > 0:
        print(f"\n⚠️  Top Drifted Features:")
        drifted = data_drift_results[data_drift_results['drift_detected']].sort_values('psi_value', ascending=False)
        print(drifted[['feature', 'psi_value', 'ks_pvalue', 'drift_severity']].head(10).to_string(index=False))
    else:
        print("\n✓ No significant data drift detected")
        
except Exception as e:
    print(f"⚠️  Data drift detection skipped: {str(e)}")

# ======================================================================================
# 2. CONCEPT DRIFT DETECTION
# Monitors changes in the relationship between features and target (P(Y|X))
# ======================================================================================

print("\n[2/4] CONCEPT DRIFT DETECTION")
print("-" * 80)

def detect_concept_drift(train_df, prod_df, target_col='failure_within_next_24_hours'):
    """
    Detect concept drift by comparing target distributions and correlations.
    """
    results = {}
    
    # Target distribution shift
    if target_col in train_df.columns and target_col in prod_df.columns:
        train_target_dist = train_df[target_col].value_counts(normalize=True).to_dict()
        prod_target_dist = prod_df[target_col].value_counts(normalize=True).to_dict()
        
        results['train_positive_rate'] = train_target_dist.get(1, 0)
        results['prod_positive_rate'] = prod_target_dist.get(1, 0)
        results['target_shift'] = abs(results['train_positive_rate'] - results['prod_positive_rate'])
        results['concept_drift_detected'] = results['target_shift'] > 0.05  # 5% threshold
        
        # Chi-square test for target distribution
        if len(train_target_dist) > 1 and len(prod_target_dist) > 1:
            train_counts = train_df[target_col].value_counts().values
            prod_counts = prod_df[target_col].value_counts().values
            
            # Ensure same categories
            if len(train_counts) == len(prod_counts):
                contingency_table = np.array([train_counts, prod_counts])
                chi2, pvalue, dof, expected = chi2_contingency(contingency_table)
                results['chi2_statistic'] = chi2
                results['chi2_pvalue'] = pvalue
            else:
                results['chi2_statistic'] = None
                results['chi2_pvalue'] = None
        
        results['check_timestamp'] = datetime.now()
    
    return results

try:
    concept_drift_result = detect_concept_drift(train_baseline, prod_data)
    
    # Save concept drift results
    concept_drift_table = f"{config.catalog}.{monitoring_schema}.concept_drift_monitoring"
    concept_df = pd.DataFrame([concept_drift_result])
    spark.createDataFrame(concept_df).write.format("delta").mode("append").saveAsTable(concept_drift_table)
    
    print(f"✓ Concept drift monitoring complete: {concept_drift_table}")
    print(f"\nConcept Drift Summary:")
    print(f"  Training positive rate: {concept_drift_result['train_positive_rate']:.4f}")
    print(f"  Production positive rate: {concept_drift_result['prod_positive_rate']:.4f}")
    print(f"  Target shift: {concept_drift_result['target_shift']:.4f}")
    print(f"  Concept drift detected: {concept_drift_result['concept_drift_detected']}")
    
    if concept_drift_result['concept_drift_detected']:
        print("\n⚠️  WARNING: Significant concept drift detected - consider retraining")
        
except Exception as e:
    print(f"⚠️  Concept drift detection skipped: {str(e)}")

# ======================================================================================
# 3. PREDICTION DRIFT DETECTION
# Monitors changes in model predictions distribution over time
# ======================================================================================

print("\n[3/4] PREDICTION DRIFT DETECTION")
print("-" * 80)

def detect_prediction_drift(baseline_preds, current_preds, threshold=0.1):
    """
    Detect prediction drift by comparing score distributions.
    """
    results = {}
    
    # Score distribution metrics
    results['baseline_mean_score'] = float(baseline_preds.mean())
    results['current_mean_score'] = float(current_preds.mean())
    results['baseline_std_score'] = float(baseline_preds.std())
    results['current_std_score'] = float(current_preds.std())
    results['mean_score_shift'] = abs(results['baseline_mean_score'] - results['current_mean_score'])
    
    # KS test on prediction scores
    ks_stat, ks_pval = ks_2samp(baseline_preds, current_preds)
    results['ks_statistic'] = ks_stat
    results['ks_pvalue'] = ks_pval
    
    # PSI on prediction scores
    psi = calculate_psi(baseline_preds.values, current_preds.values)
    results['psi_value'] = psi
    
    # Drift detection
    results['prediction_drift_detected'] = (ks_pval < 0.05) or (psi > threshold) or (results['mean_score_shift'] > threshold)
    results['drift_severity'] = "CRITICAL" if psi > 0.2 else ("WARNING" if psi > 0.1 else "OK")
    results['check_timestamp'] = datetime.now()
    
    return results

try:
    # Load prediction data if available
    prediction_table = config.get_prediction_table_name()
    
    # Check if prediction table exists
    table_exists = spark.catalog.tableExists(prediction_table)
    
    if table_exists:
        pred_df = spark.table(prediction_table).toPandas()
        
        if 'failure_probability' in pred_df.columns and len(pred_df) > 100:
            # Split into baseline (first 70%) and current (last 30%)
            split_idx = int(len(pred_df) * 0.7)
            baseline_preds = pred_df['failure_probability'].iloc[:split_idx]
            current_preds = pred_df['failure_probability'].iloc[split_idx:]
            
            pred_drift_result = detect_prediction_drift(baseline_preds, current_preds)
            
            # Save prediction drift results
            pred_drift_table = f"{config.catalog}.{monitoring_schema}.prediction_drift_monitoring"
            pred_drift_df = pd.DataFrame([pred_drift_result])
            spark.createDataFrame(pred_drift_df).write.format("delta").mode("append").saveAsTable(pred_drift_table)
            
            print(f"✓ Prediction drift monitoring complete: {pred_drift_table}")
            print(f"\nPrediction Drift Summary:")
            print(f"  Baseline mean score: {pred_drift_result['baseline_mean_score']:.4f}")
            print(f"  Current mean score: {pred_drift_result['current_mean_score']:.4f}")
            print(f"  Mean score shift: {pred_drift_result['mean_score_shift']:.4f}")
            print(f"  PSI value: {pred_drift_result['psi_value']:.4f}")
            print(f"  Drift severity: {pred_drift_result['drift_severity']}")
            
            if pred_drift_result['prediction_drift_detected']:
                print("\n⚠️  WARNING: Prediction drift detected")
        else:
            print("⚠️  Insufficient prediction data for drift detection")
    else:
        print("⚠️  Prediction table not found - run batch inference first")
        
except Exception as e:
    print(f"⚠️  Prediction drift detection skipped: {str(e)}")

# ======================================================================================
# 4. MODEL DRIFT DETECTION
# Monitors model performance degradation over time (when ground truth is available)
# ======================================================================================

print("\n[4/4] MODEL DRIFT DETECTION")
print("-" * 80)

def detect_model_drift(y_true_baseline, y_pred_baseline, y_true_current, y_pred_current, threshold=0.05):
    """
    Detect model drift by comparing performance metrics over time.
    """
    results = {}
    
    # Baseline metrics
    baseline_precision = precision_score(y_true_baseline, y_pred_baseline, zero_division=0)
    baseline_recall = recall_score(y_true_baseline, y_pred_baseline, zero_division=0)
    baseline_f1 = f1_score(y_true_baseline, y_pred_baseline, zero_division=0)
    
    # Current metrics
    current_precision = precision_score(y_true_current, y_pred_current, zero_division=0)
    current_recall = recall_score(y_true_current, y_pred_current, zero_division=0)
    current_f1 = f1_score(y_true_current, y_pred_current, zero_division=0)
    
    # Calculate degradation
    results['baseline_precision'] = baseline_precision
    results['baseline_recall'] = baseline_recall
    results['baseline_f1'] = baseline_f1
    results['current_precision'] = current_precision
    results['current_recall'] = current_recall
    results['current_f1'] = current_f1
    
    results['precision_degradation'] = baseline_precision - current_precision
    results['recall_degradation'] = baseline_recall - current_recall
    results['f1_degradation'] = baseline_f1 - current_f1
    
    # Model drift detection (any metric drops by more than threshold)
    results['model_drift_detected'] = (
        (results['precision_degradation'] > threshold) or
        (results['recall_degradation'] > threshold) or
        (results['f1_degradation'] > threshold)
    )
    
    # Severity based on recall degradation (most critical for failure prediction)
    if results['recall_degradation'] > 0.1:
        results['drift_severity'] = "CRITICAL"
    elif results['recall_degradation'] > 0.05:
        results['drift_severity'] = "WARNING"
    else:
        results['drift_severity'] = "OK"
    
    results['check_timestamp'] = datetime.now()
    
    return results

try:
    # For demonstration: use test set as baseline and validation as current
    # In production, compare historical performance vs recent performance
    
    if 'y_test' in globals() and 'y_val' in globals():
        # Generate predictions for validation set
        if 'champion_name' in globals() and champion_name:
            champion_model = results[champion_name]['model']
            y_val_pred = champion_model.predict(X_val)
            y_test_pred = champion_model.predict(X_test)
            
            model_drift_result = detect_model_drift(
                y_test, y_test_pred,
                y_val, y_val_pred
            )
            
            # Save model drift results
            model_drift_table = f"{config.catalog}.{monitoring_schema}.model_drift_monitoring"
            model_drift_df = pd.DataFrame([model_drift_result])
            spark.createDataFrame(model_drift_df).write.format("delta").mode("append").saveAsTable(model_drift_table)
            
            print(f"✓ Model drift monitoring complete: {model_drift_table}")
            print(f"\nModel Performance Comparison:")
            print(f"  Baseline Recall: {model_drift_result['baseline_recall']:.4f}")
            print(f"  Current Recall: {model_drift_result['current_recall']:.4f}")
            print(f"  Recall Degradation: {model_drift_result['recall_degradation']:.4f}")
            print(f"  Baseline F1: {model_drift_result['baseline_f1']:.4f}")
            print(f"  Current F1: {model_drift_result['current_f1']:.4f}")
            print(f"  F1 Degradation: {model_drift_result['f1_degradation']:.4f}")
            print(f"  Drift severity: {model_drift_result['drift_severity']}")
            
            if model_drift_result['model_drift_detected']:
                print("\n⚠️  WARNING: Model performance degradation detected - retraining recommended")
        else:
            print("⚠️  No champion model available for drift detection")
    else:
        print("⚠️  Test/validation data not available - run training pipeline first")
        
except Exception as e:
    print(f"⚠️  Model drift detection skipped: {str(e)}")

# ======================================================================================
# DRIFT MONITORING SUMMARY & ALERTING
# ======================================================================================

print("\n" + "="*80)
print("DRIFT MONITORING SUMMARY")
print("="*80)

try:
    # Create comprehensive drift summary
    drift_summary = {
        'timestamp': datetime.now(),
        'data_drift_features_affected': int(data_drift_results['drift_detected'].sum()) if 'data_drift_results' in locals() else 0,
        'data_drift_critical': int((data_drift_results['drift_severity'] == 'CRITICAL').sum()) if 'data_drift_results' in locals() else 0,
        'concept_drift_detected': concept_drift_result.get('concept_drift_detected', False) if 'concept_drift_result' in locals() else False,
        'prediction_drift_detected': pred_drift_result.get('prediction_drift_detected', False) if 'pred_drift_result' in locals() else False,
        'model_drift_detected': model_drift_result.get('model_drift_detected', False) if 'model_drift_result' in locals() else False,
        'overall_drift_status': 'CRITICAL' if any([
            data_drift_results['drift_detected'].sum() > 5 if 'data_drift_results' in locals() else False,
            concept_drift_result.get('concept_drift_detected', False) if 'concept_drift_result' in locals() else False,
            model_drift_result.get('drift_severity') == 'CRITICAL' if 'model_drift_result' in locals() else False
        ]) else 'OK',
        'retraining_recommended': False
    }
    
    # Determine if retraining is recommended
    drift_summary['retraining_recommended'] = (
        drift_summary['data_drift_critical'] > 3 or
        drift_summary['concept_drift_detected'] or
        drift_summary['model_drift_detected']
    )
    
    # Save summary
    summary_table = f"{config.catalog}.{monitoring_schema}.drift_summary"
    summary_df = pd.DataFrame([drift_summary])
    spark.createDataFrame(summary_df).write.format("delta").mode("append").saveAsTable(summary_table)
    
    print(f"\n✓ Drift summary saved: {summary_table}")
    print(f"\n📊 Overall Status: {drift_summary['overall_drift_status']}")
    print(f"   Data drift features: {drift_summary['data_drift_features_affected']}")
    print(f"   Concept drift: {'YES' if drift_summary['concept_drift_detected'] else 'NO'}")
    print(f"   Prediction drift: {'YES' if drift_summary['prediction_drift_detected'] else 'NO'}")
    print(f"   Model drift: {'YES' if drift_summary['model_drift_detected'] else 'NO'}")
    print(f"   Retraining recommended: {'YES ⚠️' if drift_summary['retraining_recommended'] else 'NO'}")
    
    if drift_summary['retraining_recommended']:
        print("\n" + "="*80)
        print("⚠️  ACTION REQUIRED: RETRAINING RECOMMENDED")
        print("="*80)
        print("Reasons:")
        if drift_summary['data_drift_critical'] > 3:
            print(f"  • {drift_summary['data_drift_critical']} features show critical data drift")
        if drift_summary['concept_drift_detected']:
            print("  • Concept drift detected (target distribution shift)")
        if drift_summary['model_drift_detected']:
            print("  • Model performance degradation detected")
        print("\nRecommended Actions:")
        print("  1. Run data generation with latest date range")
        print("  2. Trigger model retraining pipeline")
        print("  3. Evaluate new model against quality gates")
        print("  4. Promote to champion if performance improves")
    else:
        print("\n✓ All drift metrics within acceptable thresholds")
        print("✓ No immediate retraining required")
    
except Exception as e:
    print(f"⚠️  Drift summary generation failed: {str(e)}")

print("\n" + "="*80)
print("DRIFT MONITORING COMPLETE")
print("="*80)