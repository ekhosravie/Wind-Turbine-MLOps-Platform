"""
Synthetic wind turbine SCADA data generator.

generate_wind_turbine_data() is verbatim from the original notebook's
generator cell - it was already written as a single self-contained function
taking `config` and returning a DataFrame, so no restructuring was needed
here. This IS the ground-truth schema referenced by src/common/schema.py -
if you change a column here, update RAW_SCHEMA and LEGACY_ALIASES there too.
"""

# Enhanced Synthetic Wind Turbine Data Generator
# Demonstrates multi-dataset architecture with realistic physics and degradation patterns

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pyspark.sql import functions as F
from pyspark.sql.types import *
import warnings
import uuid
warnings.filterwarnings('ignore')

np.random.seed(config.random_seed)

def generate_wind_turbine_data(config: ProjectConfig) -> pd.DataFrame:
    """
    Generate realistic synthetic wind turbine sensor data with:
    - Multi-turbine, multi-farm structure
    - Physics-based correlations (wind→power, load→temperature)
    - Degradation-driven failures (not random)
    - Comprehensive sensor suite
    - Data quality issues
    """
    
    # Date range
    start_date = datetime.strptime(config.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(config.end_date, "%Y-%m-%d")
    timestamps = pd.date_range(start=start_date, end=end_date, 
                               freq=f'{config.sensor_interval_minutes}min')
    
    # Turbine and farm structure
    turbine_ids = [f'T{i:03d}' for i in range(1, config.num_turbines + 1)]
    farm_ids = [f'FARM_{chr(65+i)}' for i in range(config.num_farms)]
    
    # Turbine models with different characteristics
    turbine_models = [
        {'model': 'Vestas V150', 'rated_power': 4.2, 'rotor_diameter': 150, 'hub_height': 120},
        {'model': 'Siemens SG 5.0', 'rated_power': 5.0, 'rotor_diameter': 154, 'hub_height': 110},
        {'model': 'GE 3.2-130', 'rated_power': 3.2, 'rotor_diameter': 130, 'hub_height': 100}
    ]
    
    all_data = []
    failure_events_list = []
    maintenance_events_list = []
    
    for idx, turbine_id in enumerate(turbine_ids):
        # Assign farm and model
        farm_id = farm_ids[idx % config.num_farms]
        model_info = turbine_models[idx % len(turbine_models)]
        
        # Turbine age (older turbines = higher failure probability)
        turbine_age_years = np.random.uniform(1, 10)
        age_factor = turbine_age_years / 10.0  # 0.1 to 1.0
        
        n_records = len(timestamps)
        
        # === COMPONENT HEALTH TRAJECTORY ===
        # Initialize health at 1.0 (perfect health)
        health_status = np.ones(n_records)
        
        # Simulate realistic failure events with degradation
        base_failure_prob = config.failure_rate * age_factor
        num_failures = np.random.poisson(base_failure_prob * n_records / 1000)
        
        failure_types_pool = ['GEARBOX', 'BEARING', 'GENERATOR', 'HYDRAULIC', 'ELECTRICAL']
        
        for failure_num in range(num_failures):
            # Random failure time (not in first/last 10%)
            failure_idx = np.random.randint(int(n_records * 0.1), int(n_records * 0.9))
            failure_type = np.random.choice(failure_types_pool)
            
            # Degradation window: 3-7 days (72-168 hours)
            degradation_hours = config.degradation_window_hours + np.random.randint(-24, 48)
            degradation_periods = int((degradation_hours * 60) / config.sensor_interval_minutes)
            start_idx = max(0, failure_idx - degradation_periods)
            
            # Gradual health degradation
            health_status[start_idx:failure_idx] = np.linspace(1.0, 0.15, failure_idx - start_idx)
            
            # Failure duration: 2-5 days
            failure_duration = int(np.random.randint(2, 6) * 24 * 60 / config.sensor_interval_minutes)
            failure_end = min(n_records, failure_idx + failure_duration)
            health_status[failure_idx:failure_end] = 0.0
            
            # Recovery after maintenance: 1-2 days
            recovery_duration = int(np.random.randint(1, 3) * 24 * 60 / config.sensor_interval_minutes)
            recovery_end = min(n_records, failure_end + recovery_duration)
            if recovery_end > failure_end:
                health_status[failure_end:recovery_end] = np.linspace(0.4, 1.0, recovery_end - failure_end)
            
            # Log failure event
            failure_events_list.append({
                'failure_id': f'F{len(failure_events_list)+1:05d}',
                'turbine_id': turbine_id,
                'farm_id': farm_id,
                'failure_timestamp': timestamps[failure_idx],
                'failure_type': failure_type,
                'component': failure_type,
                'severity': np.random.choice(['MODERATE', 'SEVERE', 'CRITICAL'], p=[0.3, 0.5, 0.2]),
                'downtime_hours': failure_duration * config.sensor_interval_minutes / 60,
                'repair_cost_eur': np.random.uniform(5000, 50000),
                'root_cause': f'{failure_type}_DEGRADATION'
            })
            
            # Log maintenance event
            maintenance_events_list.append({
                'maintenance_id': f'M{len(maintenance_events_list)+1:05d}',
                'turbine_id': turbine_id,
                'farm_id': farm_id,
                'maintenance_timestamp': timestamps[failure_end],
                'maintenance_type': 'CORRECTIVE',
                'component': failure_type,
                'downtime_hours': failure_duration * config.sensor_interval_minutes / 60,
                'cost_eur': np.random.uniform(5000, 50000),
                'planned_flag': False
            })
        
        # === ENVIRONMENTAL CONDITIONS (Physics-Based) ===
        # Wind speed: Gamma distribution (realistic wind behavior)
        wind_speed = np.random.gamma(5, 2, n_records)
        wind_speed = np.clip(wind_speed, 0, 30)  # 0-30 m/s range
        wind_direction = np.random.uniform(0, 360, n_records)
        
        # Temperature: Seasonal variation
        day_of_year = np.array([(ts - timestamps[0]).days % 365 for ts in timestamps])
        seasonal_temp = 15 + 10 * np.sin(2 * np.pi * day_of_year / 365)
        ambient_temperature = seasonal_temp + np.random.normal(0, 5, n_records)
        
        humidity = np.clip(np.random.beta(5, 2, n_records) * 100, 0, 100)
        pressure = np.random.normal(1013, 15, n_records)
        air_density = 1.225 * (pressure / 1013) * (288 / (ambient_temperature + 273))
        precipitation = np.random.exponential(0.1, n_records)
        
        # === POWER CURVE (Cubic Wind-Power Relationship) ===
        cut_in_speed = 3.0
        rated_speed = 12.0
        cut_out_speed = 25.0
        rated_power = model_info['rated_power']  # MW
        
        # Power = 0.5 * air_density * swept_area * Cp * wind_speed^3
        # Simplified cubic relationship with cut-in/cut-out
        base_power = np.zeros(n_records)
        operating_mask = (wind_speed >= cut_in_speed) & (wind_speed <= cut_out_speed)
        
        # Cubic region (cut-in to rated)
        cubic_mask = operating_mask & (wind_speed < rated_speed)
        base_power[cubic_mask] = rated_power * ((wind_speed[cubic_mask] - cut_in_speed) / (rated_speed - cut_in_speed)) ** 3
        
        # Rated region (rated to cut-out)
        rated_mask = operating_mask & (wind_speed >= rated_speed)
        base_power[rated_mask] = rated_power
        
        # === OPERATIONAL PARAMETERS ===
        rotor_speed = np.where(wind_speed > cut_in_speed, 
                               5 + (wind_speed - cut_in_speed) * 0.75, 0)
        rotor_speed = np.clip(rotor_speed, 0, 15)  # Max 15 RPM
        
        generator_speed = rotor_speed * 100  # Gearbox ratio ~100:1
        
        blade_pitch_angle = np.where(wind_speed > rated_speed, 
                                     (wind_speed - rated_speed) * 2, 0)
        blade_pitch_angle = np.clip(blade_pitch_angle, 0, 30)
        
        yaw_angle = wind_direction + np.random.normal(0, 5, n_records)
        yaw_error = np.abs(np.random.normal(0, 3, n_records))
        
        # === TEMPERATURES (Load-Dependent + Degradation-Driven) ===
        degradation_factor = 1 - health_status
        load_factor = base_power / rated_power
        
        gearbox_temperature = (ambient_temperature + 30 + 
                               rotor_speed * 2.5 + 
                               load_factor * 15 + 
                               degradation_factor * 25)
        
        generator_temperature = (ambient_temperature + 25 + 
                                 generator_speed * 0.02 + 
                                 load_factor * 20 + 
                                 degradation_factor * 20)
        
        bearing_temperature = (ambient_temperature + 20 + 
                               rotor_speed * 2 + 
                               degradation_factor * 18)
        
        main_bearing_temperature = bearing_temperature + np.random.normal(0, 2, n_records)
        
        nacelle_temperature = ambient_temperature + 10 + load_factor * 8
        
        brake_temperature = ambient_temperature + 15 + np.random.normal(0, 5, n_records)
        
        # === HYDRAULIC SYSTEM ===
        hydraulic_pressure = np.random.normal(160, 8, n_records)
        hydraulic_pressure[health_status < 0.5] -= 40  # Pressure drop during failure
        hydraulic_pressure = np.clip(hydraulic_pressure, 0, 200)
        
        hydraulic_oil_temperature = ambient_temperature + 25 + load_factor * 10
        
        # === GEARBOX OIL ===
        gearbox_oil_temperature = gearbox_temperature + np.random.normal(0, 3, n_records)
        gearbox_oil_pressure = np.random.normal(2.5, 0.2, n_records)
        gearbox_oil_pressure[health_status < 0.5] -= 0.5
        gearbox_oil_pressure = np.clip(gearbox_oil_pressure, 0, 5)
        
        # === VIBRATION (Degradation-Sensitive) ===
        vibration_x = np.random.normal(0.3, 0.08, n_records) + degradation_factor * 0.6
        vibration_y = np.random.normal(0.3, 0.08, n_records) + degradation_factor * 0.6
        vibration_z = np.random.normal(0.2, 0.06, n_records) + degradation_factor * 0.5
        vibration_rms = np.sqrt(vibration_x**2 + vibration_y**2 + vibration_z**2)
        vibration_peak = vibration_rms * (1 + np.random.uniform(0.2, 0.5, n_records))
        vibration_kurtosis = 3.0 + degradation_factor * 2.0 + np.random.normal(0, 0.5, n_records)
        
        # === ELECTRICAL MEASUREMENTS ===
        # Degradation reduces power efficiency
        active_power = base_power * (1 - degradation_factor * 0.25)
        reactive_power = active_power * 0.08  # Power factor ~0.99
        apparent_power = np.sqrt(active_power**2 + reactive_power**2)
        
        voltage = np.random.normal(690, 8, n_records)  # 690V nominal
        current = np.where(voltage > 0, 
                          active_power * 1000 / (voltage * np.sqrt(3) * 0.99), 0)
        power_factor = np.where(apparent_power > 0, active_power / apparent_power, 1.0)
        grid_frequency_hz = np.random.normal(50.0, 0.05, n_records)
        
        generator_load_pct = (active_power / rated_power) * 100
        electrical_efficiency = np.clip(85 + load_factor * 10 - degradation_factor * 15, 0, 100)
        
        # === OPERATING STATE ===
        operating_state = np.where(wind_speed < cut_in_speed, 'STOPPED',
                          np.where(health_status < 0.3, 'FAULT',
                          np.where(wind_speed > cut_out_speed, 'EMERGENCY_STOP',
                          np.where((wind_speed >= cut_in_speed) & (wind_speed < cut_in_speed + 1), 'STARTING',
                          'RUNNING'))))
        
        availability_flag = (health_status >= 0.3).astype(int)
        grid_connected_flag = (operating_state == 'RUNNING').astype(int)
        maintenance_mode_flag = ((health_status < 1.0) & (health_status >= 0.3)).astype(int)
        alarm_flag = (health_status < 0.7).astype(int)
        
        # === ALARM INFORMATION ===
        alarm_code = np.where(gearbox_temperature > 100, 'ALM-GEARBOX-TEMP',
                     np.where(generator_temperature > 95, 'ALM-GENERATOR-TEMP',
                     np.where(vibration_rms > 1.0, 'ALM-HIGH-VIBRATION',
                     np.where(hydraulic_pressure < 120, 'ALM-HYDRAULIC-PRESSURE',
                     None))))
        
        alarm_severity = np.where(alarm_flag == 1,
                         np.where(health_status < 0.3, 'CRITICAL',
                         np.where(health_status < 0.5, 'WARNING', 'INFO')),
                         None)
        
        # === FAILURE FLAGS AND TYPES ===
        failure_flag = (health_status < 0.3).astype(int)
        failure_type_arr = np.where(failure_flag == 1, 
                                   np.random.choice(failure_types_pool, n_records), 
                                   None)
        
        # === CUMULATIVE METRICS ===
        operating_hours = np.cumsum(np.where(rotor_speed > 0, 
                                             config.sensor_interval_minutes / 60, 0))
        
        # Create DataFrame for this turbine
        turbine_data = pd.DataFrame({
            # Identification
            'event_id': [f'{turbine_id}_{i:08d}' for i in range(n_records)],
            'turbine_id': turbine_id,
            'farm_id': farm_id,
            'timestamp': timestamps,
            'turbine_model': model_info['model'],
            'turbine_age_years': turbine_age_years,
            
            # Environmental
            'wind_speed': wind_speed,
            'wind_direction': wind_direction,
            'ambient_temperature': ambient_temperature,
            'humidity': humidity,
            'atmospheric_pressure': pressure,
            'air_density': air_density,
            'precipitation': precipitation,
            
            # Rotor & Mechanical
            'rotor_speed_rpm': rotor_speed,
            'generator_speed_rpm': generator_speed,
            'blade_pitch_angle': blade_pitch_angle,
            'yaw_angle': yaw_angle,
            'yaw_error': yaw_error,
            
            # Temperatures
            'gearbox_temperature': gearbox_temperature,
            'generator_temperature': generator_temperature,
            'bearing_temperature': bearing_temperature,
            'main_bearing_temperature': main_bearing_temperature,
            'nacelle_temperature': nacelle_temperature,
            'brake_temperature': brake_temperature,
            
            # Hydraulic
            'hydraulic_pressure': hydraulic_pressure,
            'hydraulic_oil_temperature': hydraulic_oil_temperature,
            
            # Gearbox Oil
            'gearbox_oil_temperature': gearbox_oil_temperature,
            'gearbox_oil_pressure': gearbox_oil_pressure,
            
            # Vibration
            'vibration_x': vibration_x,
            'vibration_y': vibration_y,
            'vibration_z': vibration_z,
            'vibration_rms': vibration_rms,
            'vibration_peak': vibration_peak,
            'vibration_kurtosis': vibration_kurtosis,
            
            # Electrical
            'active_power_kw': active_power * 1000,  # Convert to kW
            'reactive_power_kvar': reactive_power * 1000,
            'apparent_power_kva': apparent_power * 1000,
            'voltage_v': voltage,
            'current_a': current,
            'power_factor': power_factor,
            'grid_frequency_hz': grid_frequency_hz,
            'generator_load_pct': generator_load_pct,
            'electrical_efficiency': electrical_efficiency,
            
            # Operational
            'operating_state': operating_state,
            'availability_flag': availability_flag,
            'grid_connected_flag': grid_connected_flag,
            'maintenance_mode_flag': maintenance_mode_flag,
            'alarm_flag': alarm_flag,
            
            # Alarms
            'alarm_code': alarm_code,
            'alarm_severity': alarm_severity,
            
            # Failure
            'failure_flag': failure_flag,
            'failure_type': failure_type_arr,
            
            # Cumulative
            'operating_hours': operating_hours,
            'health_status': health_status
        })
        
        all_data.append(turbine_data)
        
        if (idx + 1) % 10 == 0:
            print(f"  Generated data for {idx+1}/{config.num_turbines} turbines...")
    
    # Combine all turbines
    df = pd.concat(all_data, ignore_index=True)
    
    # === DATA QUALITY ISSUES ===
    print("\n[Module 2/6] Injecting Data Quality Issues...")
    np.random.seed(config.random_seed + 1)
    
    # 1. Missing data (realistic outage patterns)
    print(f"  - Missing data: {config.missing_rate*100}% rate")
    null_columns = ['wind_speed', 'gearbox_temperature', 'vibration_x', 'vibration_y', 'active_power_kw']
    for col in null_columns:
        null_mask = np.random.random(len(df)) < config.missing_rate
        df.loc[null_mask, col] = None
    
    # Simulated outages (30-minute to 2-hour blocks)
    num_outages = int(config.num_turbines * 0.1)  # 10% of turbines have outages
    for _ in range(num_outages):
        turbine = np.random.choice(df['turbine_id'].unique())
        outage_duration = np.random.randint(3, 12)  # 30 min to 2 hours
        turbine_mask = df['turbine_id'] == turbine
        turbine_records = df[turbine_mask].index
        if len(turbine_records) > outage_duration:
            outage_start = np.random.choice(turbine_records[:-outage_duration])
            df.loc[outage_start:outage_start+outage_duration, null_columns] = None
    
    # 2. Sensor spikes (outliers)
    print(f"  - Outliers: {config.outlier_rate*100}% rate")
    spike_columns = ['vibration_x', 'vibration_y', 'vibration_z', 'gearbox_temperature', 'generator_temperature']
    for col in spike_columns:
        spike_mask = np.random.random(len(df)) < config.outlier_rate
        df.loc[spike_mask, col] = df[col].mean() + df[col].std() * np.random.uniform(4, 8, spike_mask.sum())
    
    # 3. Duplicate records
    print(f"  - Duplicates: {config.duplicate_rate*100}% rate")
    dup_count = int(len(df) * config.duplicate_rate)
    dup_indices = np.random.choice(df.index, size=dup_count, replace=False)
    duplicates = df.loc[dup_indices].copy()
    df = pd.concat([df, duplicates], ignore_index=True)
    
    # 4. Invalid values
    print(f"  - Invalid values: sensor errors")
    invalid_mask = np.random.random(len(df)) < 0.003
    df.loc[invalid_mask, 'wind_speed'] = -999
    df.loc[invalid_mask, 'humidity'] = 150  # Invalid humidity > 100%
    
    # 5. Sensor drift (gradual baseline shift)
    print(f"  - Sensor drift: {config.sensor_drift_rate*100}% of turbines")
    num_drift_turbines = int(config.num_turbines * config.sensor_drift_rate)
    drift_turbines = np.random.choice(df['turbine_id'].unique(), num_drift_turbines, replace=False)
    for turbine in drift_turbines:
        turbine_mask = df['turbine_id'] == turbine
        n_turbine_records = turbine_mask.sum()
        drift_curve = np.linspace(0, 0.3, n_turbine_records)  # Gradual 0-0.3 offset
        df.loc[turbine_mask, 'vibration_x'] += drift_curve
    
    # === ADDITIONAL DATASETS ===
    failure_events_df = pd.DataFrame(failure_events_list) if failure_events_list else pd.DataFrame()
    maintenance_events_df = pd.DataFrame(maintenance_events_list) if maintenance_events_list else pd.DataFrame()
    
    print("\n" + "="*80)
    print("GENERATION COMPLETE")
    print("="*80)
    print(f"SCADA Records        : {len(df):,}")
    print(f"Unique Turbines      : {df['turbine_id'].nunique()}")
    print(f"Date Range           : {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Failure Events       : {df['failure_flag'].sum():,} records ({len(failure_events_df)} distinct events)")
    print(f"Maintenance Events   : {len(maintenance_events_df)}")
    print(f"Missing Values       : {df.isnull().sum().sum():,}")
    print(f"Duplicate Records    : ~{dup_count:,}")
    print(f"Outlier Records      : ~{int(len(df) * config.outlier_rate):,}")
    print(f"Sensor Drift Turbines: {num_drift_turbines}")
    print("="*80)
    
    # Store additional datasets in global scope for later use
    global failure_events_data, maintenance_events_data
    failure_events_data = failure_events_df
    maintenance_events_data = maintenance_events_df
    
    return df
