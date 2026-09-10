"""
Pure drift-detection functions, extracted from the monitoring cell of the
original notebook via AST parsing (not regex - a first regex-based
extraction attempt accidentally pulled in the surrounding orchestration
try/except blocks too, which fire on import; caught by actually running the
import and seeing stray print output, not by inspection).

These five functions (calculate_psi, detect_data_drift, detect_concept_drift,
detect_prediction_drift, detect_model_drift) were already written as clean,
self-contained functions taking dataframes/arrays as arguments - unlike most
of the rest of the notebook, this part didn't need a schema fix, just
separating from the orchestration/glue code that calls them (which still
depends on notebook-global state like `train_baseline`, `champion_name`,
`X_val`/`y_val` - see notebooks/12_monitoring.py for that part).
"""

import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import ks_2samp, chi2_contingency
from sklearn.metrics import precision_score, recall_score, f1_score


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
