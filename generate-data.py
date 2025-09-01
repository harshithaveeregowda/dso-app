import pandas as pd
import numpy as np

# --- Configuration ---
N_SAMPLES = 1000
N_CUSTOMERS = 250
FILE_NAME = "data/dso_showcase_dataset.csv"

# Set a seed for reproducibility
rng = np.random.default_rng(42)

# --- Generate Base Data ---

# Create a pool of customer IDs and then sample from it to simulate repeat business
customer_ids_pool = [f'CUST_{1000 + i}' for i in range(N_CUSTOMERS)]
customer_ids = rng.choice(customer_ids_pool, size=N_SAMPLES, replace=True)

# Define business segments and regions
regions = ['NA', 'EMEA', 'APAC', 'LATAM']
segments = ['Enterprise', 'Mid-Market', 'SMB']

# Create the initial DataFrame
df = pd.DataFrame({
    'Customer ID': customer_ids,
    'Region': rng.choice(regions, size=N_SAMPLES, p=[0.4, 0.3, 0.2, 0.1]),
    'Customer Segment': rng.choice(segments, size=N_SAMPLES, p=[0.2, 0.5, 0.3]),
    'Invoice Amount': rng.lognormal(mean=8, sigma=1.5, size=N_SAMPLES).round(2),
    'Payment Terms Days': rng.choice([30, 60, 90], size=N_SAMPLES, p=[0.3, 0.6, 0.1]),
})

# --- Create Feature Variables with Interdependencies ---

# Invoice Error Rate: Assume it varies slightly by region
region_error_rates = {'NA': 0.05, 'EMEA': 0.07, 'APAC': 0.10, 'LATAM': 0.12}
df['Invoice Error Rate'] = df['Region'].map(lambda r: rng.uniform(low=region_error_rates[r] - 0.02, high=region_error_rates[r] + 0.02)).round(3)
df['Invoice Error Rate'] = df['Invoice Error Rate'].clip(0, 1)

# Forecast Accuracy: Assume it varies by customer segment
segment_accuracy = {'Enterprise': 0.95, 'Mid-Market': 0.88, 'SMB': 0.82}
df['Forecast Accuracy'] = df['Customer Segment'].map(lambda s: rng.uniform(low=segment_accuracy[s] - 0.05, high=segment_accuracy[s] + 0.05)).round(3)
df['Forecast Accuracy'] = df['Forecast Accuracy'].clip(0, 1)

# Avg Days Late: Dependent on payment terms (longer terms might have more variance)
lam_map = {30: 4, 60: 6, 90: 8}
df['Avg Days Late Last3 Days'] = df['Payment Terms Days'].map(lambda pt: rng.poisson(lam=lam_map[pt]))

# Contract Extension Days: A rarer event
df['Contract Extension Days'] = rng.choice([0, 15, 30, 45], size=N_SAMPLES, p=[0.90, 0.05, 0.04, 0.01])

# --- Simulate the Target Variable (DSO) ---

# Base DSO
base_dso = df['Payment Terms Days'] + df['Avg Days Late Last3 Days']

# Effects from different features
invoice_error_effect = (df['Invoice Error Rate'] * 80)
forecast_accuracy_effect = ((1 - df['Forecast Accuracy']) * 60)
extension_effect = df['Contract Extension Days'] * 0.75

# Effects from new categorical variables
segment_effect = df['Customer Segment'].map({'Enterprise': -2, 'Mid-Market': 1, 'SMB': 4})
region_effect = df['Region'].map({'NA': 0, 'EMEA': 1, 'APAC': 3, 'LATAM': 5})
amount_effect = np.log1p(df['Invoice Amount'] / 1000) # Larger invoices might be paid slightly slower

# Combine all effects and add random noise
df['DSO actual Days'] = (
    base_dso + 
    invoice_error_effect + 
    forecast_accuracy_effect + 
    extension_effect +
    segment_effect +
    region_effect +
    amount_effect +
    rng.normal(0, 4, size=N_SAMPLES)
).round(1)

# Ensure DSO is not negative
df['DSO actual Days'] = df['DSO actual Days'].clip(0, None)


# --- Introduce Missing Values to Simulate Real Data ---
missing_indices_avg_late = df.sample(frac=0.05, random_state=1).index
df.loc[missing_indices_avg_late, 'Avg Days Late Last3 Days'] = np.nan

missing_indices_forecast = df.sample(frac=0.03, random_state=2).index
df.loc[missing_indices_forecast, 'Forecast Accuracy'] = np.nan


# --- Finalize and Save ---
# Reorder columns for clarity
final_columns = [
    'Customer ID', 'Region', 'Customer Segment', 'Invoice Amount', 
    'Payment Terms Days', 'Invoice Error Rate', 'Forecast Accuracy', 
    'Contract Extension Days', 'Avg Days Late Last3 Days', 'DSO actual Days'
]
df = df[final_columns]

# Save to CSV
df.to_csv(FILE_NAME, index=False)

print(f"✅ Successfully generated dataset with {N_SAMPLES} samples.")
print(f"💾 Saved to '{FILE_NAME}'.")
print("\nData Head:")
print(df.head())