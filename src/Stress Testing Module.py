import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score, brier_score_loss, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Loan Portfolio Simulation

np.random.seed(42)
n_loans = 5000

# Generate realistic base variables
# LTV (Loan to Value): Centered around 65%, standard deviation 15%
ltv = np.random.normal(0.65, 0.15, n_loans)
# DSCR centered around 1.25
dscr = np.random.normal(1.25, 0.3, n_loans)
# Macro (Unemployment Rate): Random between 3% and 10%
unemployment = np.random.uniform(0.03, 0.10, n_loans)

# True log-odds of default
# Higher LTV increases risk (+), higher DSCR lowers risk (-), higher unemployment increases risk (+)
log_odds = -3.5 + (4.0 * ltv) - (2.5 * dscr) + (15.0 * unemployment)
true_pd = 1 / (1 + np.exp(-log_odds))

# Actual default events (1 = Default, 0 = Performing) based on the true PD
defaults = np.random.binomial(1, true_pd)

# Empty Data Frame
df = pd.DataFrame({
    'loan_id': range(1, n_loans + 1),
    'LTV': ltv,
    'DSCR': dscr,
    'Macro_Unemp': unemployment,
    'Default': defaults
})

# "Dirty" data to simulate real-world banking databases
df.loc[10:20, 'LTV'] = np.nan # Missing LTV data
df.loc[50:55, 'LTV'] = 2.5    # Impossible LTV
df.loc[100:110, 'DSCR'] = -0.5 # Impossible negative DSCR

# Data Quality Checks
print("--- DATA INTEGRITY AUDIT ---")

# Checking for missing values
missing_data = df.isnull().sum()
print(f"Missing Values:\n{missing_data[missing_data > 0]}\n")

# Checking for boundary logic violations
invalid_ltv = df[(df['LTV'] < 0) | (df['LTV'] > 1.05)]
print(f"Boundary Violation: {len(invalid_ltv)} loans with LTV outside 0-105%.")

invalid_dscr = df[df['DSCR'] < 0]
print(f"Boundary Violation: {len(invalid_dscr)} loans with negative DSCR.")

# Remediation: Dropping/capping bad data
df_clean = df.dropna().copy()
df_clean = df_clean[(df_clean['LTV'] >= 0) & (df_clean['LTV'] <= 1.05)]
df_clean = df_clean[df_clean['DSCR'] >= 0]

print(f"Data cleaning complete. Retained {len(df_clean)} of {n_loans} records.\n")

# Train/Test Split (80% Build, 20% Out-of-Sample Validation)
train_df, test_df = train_test_split(df_clean, test_size=0.20, random_state=42)

# Fitting Logistic Regression Model
model_formula = 'Default ~ LTV + DSCR + Macro_Unemp'
pd_model = smf.logit(formula=model_formula, data=train_df).fit(disp=0)

# Statistical summary for Conceptual Soundness review
print("--- MODEL CONCEPTUAL SOUNDNESS (IN-SAMPLE) ---")
print(pd_model.summary())
# 4. Out-of-Sample (OOS) Backtesting
print("\n--- OUT-OF-SAMPLE BACKTESTING ---")

# Generates predicted probabilities for the test set
test_df['Predicted_PD'] = pd_model.predict(test_df)

# Metric 1: ROC-AUC (Area Under the Receiver Operating Characteristic Curve)
# Measures the model's ability to rank-order risk (1.0 is perfect, 0.5 is a coin toss)
auc = roc_auc_score(test_df['Default'], test_df['Predicted_PD'])
print(f"ROC-AUC Score: {auc:.4f} (Industry standard: > 0.70 is acceptable)")

# Metric 2: Brier Score
# Measures the accuracy of the probability outputs (0.0 is perfect)
brier = brier_score_loss(test_df['Default'], test_df['Predicted_PD'])
print(f"Brier Score: {brier:.4f}")

# Metric 3: Classification Report at a 15% probability threshold
# If the model says PD > 15%, we flag it as a predicted default
threshold = 0.15
test_df['Predicted_Class'] = (test_df['Predicted_PD'] > threshold).astype(int)

print("\nClassification Report (15% Threshold):")
print(classification_report(test_df['Default'], test_df['Predicted_Class']))

# STRESS TEST

print("--- MACROECONOMIC STRESS TESTING ---")

# Copy of Portfolio for Stress scenario
stress_df = test_df.copy()

# Apply Severe Adverse Shock
# Shock 1: Unemployment spikes to 12% for all loans
stress_df['Macro_Unemp'] = 0.12

# Shock 2: Property values drop 20%, increases LTV
# New LTV = LTV / (1 - 0.20)
stress_df['LTV'] = stress_df['LTV'] / 0.80

# Leaving DSCR alone for this specific test

# Predict new Probability of Defaults (PDs) under stress
test_df['Baseline_PD'] = pd_model.predict(test_df)
stress_df['Stressed_PD'] = pd_model.predict(stress_df)

# Calculate Portfolio-Level Impact
baseline_avg_pd = test_df['Baseline_PD'].mean()
stressed_avg_pd = stress_df['Stressed_PD'].mean()

pd_multiplier = stressed_avg_pd / baseline_avg_pd

print(f"Baseline Portfolio Average PD: {baseline_avg_pd:.2%}")
print(f"Stressed Portfolio Average PD: {stressed_avg_pd:.2%}")
print(f"Risk Multiplier: The portfolio is {pd_multiplier:.1f}x riskier under the severe scenario.")

# Tail Risk Evaluation
# Let's say any loan with a PD > 20% is on the "Watchlist"
baseline_watchlist = (test_df['Baseline_PD'] > 0.20).sum()
stressed_watchlist = (stress_df['Stressed_PD'] > 0.20).sum()

print(f"\nBaseline Watchlist (PD > 20%): {baseline_watchlist} loans")
print(f"Stressed Watchlist (PD > 20%): {stressed_watchlist} loans")


print("--- EXPECTED LOSS (EL) CALCULATION ---")

#  Generate outstanding loan balances (EAD)
# Assume these are commercial loans ranging from $1M to $25M
np.random.seed(42)
test_df['EAD'] = np.random.uniform(1_000_000, 25_000_000, len(test_df))
stress_df['EAD'] = test_df['EAD'] # The balances remain the same in the stress scenario

# 2. Assign Loss Given Default (LGD)
# Assign a standard 45% for LGD in this case.
# Assign a stress LGD of 65 %
test_df['LGD'] = 0.45
stress_df['LGD'] = 0.65

# Expected Loss per loan
test_df['Baseline_EL'] = test_df['Baseline_PD'] * test_df['LGD'] * test_df['EAD']
stress_df['Stressed_EL'] = stress_df['Stressed_PD'] * stress_df['LGD'] * stress_df['EAD']

# Portfolio Totals
total_portfolio_balance = test_df['EAD'].sum()

baseline_total_el = test_df['Baseline_EL'].sum()
stressed_total_el = stress_df['Stressed_EL'].sum()

# Financial Impact
print(f"Total Portfolio Exposure (EAD): ${total_portfolio_balance:,.2f}")
print("-" * 40)
print(f"Baseline Expected Loss:         ${baseline_total_el:,.2f}")
print(f"Stressed Expected Loss:         ${stressed_total_el:,.2f}")
print("-" * 40)

capital_shortfall = stressed_total_el - baseline_total_el
print(f"Required Capital Buffer:        ${capital_shortfall:,.2f}")