# loan-approval-prediction

An AI-assisted data science project that investigates which borrower characteristics most strongly predict loan approval. Results are presented in an interactive Streamlit report.

This is an individual project for UNCG CSC 405, Fall 2026.

## Project Objective
**Research question:** Which borrower characteristics most strongly predict loan approval outcomes?

The project cleans and explores the data, tests the relationship between each characteristic and approval with formal statistical tests, and compares a baseline model (logistic regression) with a more flexible one (random forest). Feature importance from both models is used to identify the strongest predictors. Google Gemini (through LangChain) was consulted at five decision points; each answer is logged and then accepted, modified, or rejected with a written justification.

## Dataset
- **Name:** Loan Approval Dataset 2026 | Credit Risk & Bank A
- **Source:** Kaggle User mmumairkhattak: https://www.kaggle.com/datasets/mmumairkhattak/loan-approval-dataset-2026-credit-risk-and-bank-a
- **Size:** 30,000 rows, 36 columns
- **Contents:** Applicant demographics, income, employment, credit history, banking behavior, loan request details, collateral indicators, risk scores, and approval outcomes.
- **Target variable:** `Loan_Approved`

## Repository Structure
```
loan-approval-prediction/
├── data/loan_approval_dataset.csv   # dataset
├── src/
│   ├── app.py                       # Streamlit report (analysis + write-up)
│   └── cache_helper.py              # Gemini calls through LangChain, with a saved-answer cache
├── figures/                         # figures exported by the app
├── report/
│   ├── ai_log.json                  # logged Gemini prompts and responses
│   └── final_report.pdf             # PDF export of the report
├── requirements.txt
├── .env.example
└── README.md
```

## Methodology
1. **Overview and audit:** structure, missing values, and data types.
2. **Leakage and outlier review:** `Approved_Amount`, `Default_Risk`, `Risk_Score`, `Interest_Rate`, `Monthly_Income`, and `Application_ID` were removed because they are outcomes of the bank's own decision process or carry no predictive information. Outliers were judged to be legitimate extremes and kept; skewed features are log-transformed for logistic regression only.
3. **Exploratory data analysis:** outcome balance, approval rates by group, distributions, and correlations.
4. **Statistical tests:** Welch's t-tests (5 numeric variables) and chi-square tests of independence (8 categorical variables) with Cohen's d and Cramér's V as effect sizes, a Bonferroni correction across all 13 tests, and a Spearman correlation as a robustness check.
5. **Machine learning:** feature selection (VIF and collinearity checks, demographic features excluded), stratified 80/20 split, 5-fold grid search on ROC-AUC, logistic regression (baseline) versus random forest (advanced), permutation importance, and a gender fairness audit.

## AI-Assisted Workflow
Gemini was consulted at five steps: leakage review, outlier treatment, statistical test selection, feature selection, and model setup. Every prompt and response is saved in `report/ai_log.json`. For each step the report documents what was accepted, modified, or rejected and why. Saved answers are reused on later runs, so the app does not call Gemini again unless an entry is removed from the log.

## Key Findings
- Credit score is by far the strongest predictor in both models, followed by loan-to-income ratio and previous loan status. Employment status, late payments, collateral, and debt-to-income ratio come next.
- Logistic regression and random forest perform almost identically on the held-out test set (ROC-AUC 0.889 for both; accuracy 0.811 and 0.814), so the relationships appear to be largely additive.
- Gender was not used as a model input, and the audit found no evidence that the models treat genders differently.

## Limitations
- The data appears synthetic, and some values are clipped at fixed bounds.
- Several bank-derived columns were removed to avoid leakage, so the results describe borrower characteristics only, not the bank's full decision process.
- Both models over-approve slightly relative to the true approval rate, and test AUC plateaus near 0.89.
- Correlated features can split importance between them, so individual rankings should be read with care.
- Associations are not causal.

## How to Run

### Prerequisites
- Python 3.14 (3.12 or 3.13 should also work, 3.15 may not)
- Git
- A Google Gemini API key

### 1. Clone the repository
```bash
git clone https://github.com/clairenfarmer/loan-approval-prediction.git
cd loan-approval-prediction
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**
```powershell
py -3.14 -m venv .venv
.venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3.14 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Set up your API key
Copy `.env.example` to `.env` and replace the placeholder with your own key:
```
GOOGLE_API_KEY=your_key_here
```
The `.env` file is excluded from version control, so the app will not run its AI features without it.

### 5. Confirm the dataset is present
The dataset is included at `data/loan_approval_dataset.csv`. If it is missing, download it from the Kaggle link above and place it in the `data/` folder using that filename.

### 6. Run the app
From the repository root:
```bash
streamlit run src/app.py
```
Streamlit will open the report in your browser (default: http://localhost:8501).