\# SR 11-7 Commercial Real Estate Default Model \& Validation



\*\*Objective:\*\*  

An end-to-end development, stress testing, and independent validation of a Commercial Real Estate (CRE) Probability of Default (PD) model. This project is structured to comply with the Federal Reserve’s SR 11-7 guidance on Model Risk Management (MRM), demonstrating conceptual soundness, data integrity auditing, and quantitative stress testing.



\*\*🔗 \[Read the full Model Validation Document (PDF)](docs/CRE-Default%20MVD.pdf)\*\*



\## 📊 Project Overview



This repository simulates the workflow of a Quantitative Risk Analyst validating a credit risk model at a financial institution. It moves from raw data ingestion and integrity auditing, through mathematical modeling and out-of-sample backtesting, and concludes with dollar-value Expected Loss (EL) projections under severe macroeconomic stress (CCAR/DFAST scenarios).



\*\*Core Technologies \& Libraries:\*\*  

`Python 3.10` | `statsmodels` | `scikit-learn` | `pandas` | `numpy`



\## 🧠 Methodology



1\.  \*\*Mathematical Framework:\*\* 

&#x20;   \*   Implemented a Multivariate Binomial Logistic Regression to estimate the 12-month Probability of Default (PD).

&#x20;   \*   \*\*Features:\*\* Loan-to-Value (LTV), Debt Service Coverage Ratio (DSCR), and Regional Unemployment (Macro).

2\.  \*\*Data Integrity Audit:\*\* 

&#x20;   \*   Engineered automated checks to flag missing values, boundary violations (e.g., negative DSCR), and logical impossibilities before model ingestion.

3\.  \*\*Capital Stress Testing:\*\* 

&#x20;   \*   Subjected the baseline portfolio to a Severe Adverse Scenario (12% unemployment spike, 20% property value depreciation).

&#x20;   \*   Calculated capital reserves using the Expected Loss framework: $EL = PD \\times LGD \\times EAD$.



\## 🔍 Key Validation Findings



The formal validation audit (detailed in the MVD) uncovered the following structural behaviors:

\*   \*\*Non-Linear Stress Scaling:\*\* While the macroeconomic shock increased baseline PD by roughly 3x, the simultaneous deterioration of recovery rates (LGD) caused the portfolio's Expected Loss to scale by 5x, exposing hidden tail risk.

\*   \*\*Boundary Limitations:\*\* The model violates the linearity of log-odds assumption at extreme thresholds; default risk spikes exponentially when LTV crosses the 90% boundary, requiring specific guardrails for high-leverage originations.

\*   \*\*Stationarity Risk:\*\* The baseline coefficients are highly sensitive to the economic regime of the training data and risk under-predicting defaults in a prolonged high-interest-rate environment.



