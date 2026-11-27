import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import HistGradientBoostingRegressor
from cache_helper import ask_ai, save_decision, load_log

st.set_page_config(page_title="Loan Approval Analysis", layout="wide")
st.title("Loan Approval Prediction: AI-Assisted Data Science Report")
st.write("Research question: which borrower characteristics most strongly predict loan approval outcomes?")


@st.cache_data
def load_data(path):
    return pd.read_csv(path)


df = load_data("data/loan_approval_dataset.csv")

# ---------------------------------------------------------------
# 1. Dataset overview
# ---------------------------------------------------------------
st.header("1. Dataset Overview")
st.write(f"{df.shape[0]:,} rows and {df.shape[1]} columns.")
st.dataframe(df.head())
st.write(df.describe().T)

# ---------------------------------------------------------------
# 2. Data cleaning
# ---------------------------------------------------------------
st.header("2. Data Cleaning")

# 2a. Audit
st.subheader("2a. Data Quality Audit")
num = df.select_dtypes(include="number")
cat_cols = [c for c in df.columns if c not in num.columns]
audit = {
    "Missing values (total)": int(df.isnull().sum().sum()),
    "Duplicate rows": int(df.duplicated().sum()),
    "Application_ID unique": bool(df["Application_ID"].is_unique),
    "Negative values (numeric columns)": int((num < 0).sum().sum()),
    "Credit history longer than adult life": int((df["Credit_History_Years"] > df["Age"] - 18).sum()),
    "Job experience longer than adult life": int((df["Job_Experience_Years"] > df["Age"] - 18).sum()),
    "Approved amount above requested amount": int((df["Approved_Amount"] > df["Loan_Amount"]).sum()),
    "Max |Monthly_Income - Annual_Income/12|": round(float((df["Monthly_Income"] - df["Annual_Income"] / 12).abs().max()), 4),
}
st.table(pd.Series(audit, name="Result").astype(str))

st.write("Unique values in each categorical column (checking for hidden placeholders like 'Unknown' or 'N/A'):")
st.dataframe(pd.DataFrame({"column": cat_cols,
                           "unique values": [", ".join(sorted(df[c].unique())) for c in cat_cols]}))

zeros = (num == 0).sum()
st.write("Zeros per column (kept, because they are plausible: no late payments, no dependents, and so on):")
st.write(zeros[zeros > 0].rename("zero count"))
st.write("**Finding:** no missing values, duplicates, negatives, or placeholder labels were found, "
         "and the related columns are consistent with each other.")

st.write("**Decision:** Remove Application_ID and Monthly_Income")
st.write(
    "Application_ID is just an identifier, not a borrower characteristic. Since we have verified",
    " that all IDs are unique, we can safely drop it. Monthly_Income is redundant with Annual_Income",
    ", and the latter is more reliable, especially due to rounding errors when dividing by 12. We will keep Annual_Income and drop Monthly_Income."
)   
# 2b. Leakage review
st.subheader("2b. Leakage Review")


@st.cache_data
def r2_from_borrower_features(data):
    drop = ["Application_ID", "Risk_Score", "Default_Risk", "Approved_Amount",
            "Interest_Rate", "Monthly_Income", "Loan_Approved"]
    X = pd.get_dummies(data.drop(columns=drop), drop_first=True)
    return {t: round(cross_val_score(HistGradientBoostingRegressor(), X, data[t],
                                     cv=5, scoring="r2").mean(), 2)
            for t in ["Risk_Score", "Interest_Rate"]}


r2 = r2_from_borrower_features(df)
risk_bins = df.groupby("Default_Risk")["Risk_Score"].agg(["min", "max"]).round(2)
rejected_zero = (df.loc[df["Loan_Approved"] == "Rejected", "Approved_Amount"] == 0).mean()
risk_by_outcome = df.groupby("Loan_Approved")["Risk_Score"].mean().round(1).to_dict()
rate_by_outcome = df.groupby("Loan_Approved")["Interest_Rate"].mean().round(1).to_dict()
default_vs_outcome = pd.crosstab(df["Default_Risk"], df["Loan_Approved"], normalize="index").round(3)

st.write("**Evidence**")
st.write(f"- `Approved_Amount` is 0 for {rejected_zero:.1%} of rejected applications.")
st.write("- Range of `Risk_Score` within each `Default_Risk` category:")
st.dataframe(risk_bins)
st.write("- Approval outcome by `Default_Risk` (row proportions):")
st.dataframe(default_vs_outcome)
st.write(f"- Mean `Risk_Score` by outcome: {risk_by_outcome}")
st.write(f"- Mean `Interest_Rate` by outcome: {rate_by_outcome}")
st.write(f"- R² predicting `Risk_Score` from borrower features: {r2['Risk_Score']}; "
         f"`Interest_Rate`: {r2['Interest_Rate']}")

leakage_prompt = f"""
You are a senior data scientist reviewing a loan dataset for target leakage.

CONTEXT
- Dataset: 30,000 loan applications, target column Loan_Approved (Approved/Rejected).
- Research question: which borrower characteristics most strongly predict approval?
- I want to keep only information a bank would have about the applicant BEFORE making the decision, and exclude outputs of the bank's own process.

EVIDENCE
- Approved_Amount equals 0 for {rejected_zero:.1%} of rejected applications.
- Min and max Risk_Score within each Default_Risk category:
{risk_bins.to_string()}
- Share of each outcome within each Default_Risk category:
{default_vs_outcome.to_string()}
- Mean Risk_Score by outcome: {risk_by_outcome}
- Mean Interest_Rate by outcome: {rate_by_outcome}
- A gradient boosting model predicts Risk_Score from the other borrower features with R² = {r2['Risk_Score']}, and Interest_Rate with R² = {r2['Interest_Rate']}.

TASK
For each of these columns, say whether I should REMOVE or KEEP it for modeling: Approved_Amount, Default_Risk, Risk_Score, Interest_Rate. Give a one-sentence reason that cites the evidence above. Point out any evidence that is ambiguous or that you disagree with.

OUTPUT FORMAT
A markdown table with columns: Column | Decision | Reason. After the table, add 2-3 sentences on risks or anything I may have missed.
"""

st.write("**AI consultation (Gemini)**")
with st.spinner("Consulting Gemini (the saved answer is reused on later runs)..."):
    try:
        st.markdown(ask_ai("leakage_review", leakage_prompt))
    except RuntimeError as e:
        st.warning("The AI service is temporarily unavailable. Reload in a minute to retry.")
        st.caption(str(e))

st.write("**My evaluation of the AI recommendation**")
st.write("**Decision:** Modified")
st.write(
    "Geminis recommendation to remove Approved_Amount, Default_Risk, and Risk_Score is appropriate. ",
    "Approved_Amount is decided after the loan is approved, so it is not known at the time of application. ",
    "This is verified by the fact that it is 0 for 100% of rejected applications. ",
    "Default_Risk is just Risk_Score split into three bins, making it a post-decision categorization. ",
    "Risk_Score can be predicted from other borrower features (R² about 0.89) and is most likely a derived metric from the bank.",
    "Gemini suggested to KEEP Interest_Rate, but I disagree. It is also a post-decision metric, while it has a moderate correlation with borrower features (R² about 0.67). ",
    "it is unclear wether it was set before or after the approval decision. For this reason, I am excluding it to avoid possible leakage.",
    "especially since Interest_Rate is not a borrower characteristic."
)   

# based on my evaluation...
DROP_COLS = ["Application_ID", "Monthly_Income", "Approved_Amount",
             "Default_Risk", "Risk_Score", "Interest_Rate"]
df_clean = df.drop(columns=DROP_COLS)
st.write(f"Removed: {', '.join(DROP_COLS)}. {df_clean.shape[1]} columns remain.")

# 2c. Outlier detection
st.subheader("2c. Outlier Detection")
out_cols = ["Annual_Income", "Loan_Amount", "Total_Assets", "Total_Liabilities",
            "Savings_Balance", "Monthly_Debt_Payment", "Debt_To_Income_Ratio",
            "Credit_Score", "Age"]
rows = []
for c in out_cols:
    q1, q3 = df_clean[c].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    flagged = (df_clean[c] < lo) | (df_clean[c] > hi)
    rows.append({"column": c, "n_outliers": int(flagged.sum()),
                 "pct": round(100 * flagged.mean(), 2),
                 "skew": round(df_clean[c].skew(), 2),
                 "approval rate among outliers": round((df_clean.loc[flagged, "Loan_Approved"] == "Approved").mean(), 3) if flagged.any() else None})
st.dataframe(pd.DataFrame(rows))
st.write(f"Overall approval rate: {(df_clean['Loan_Approved'] == 'Approved').mean():.3f}")

fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for ax, c in zip(axes.ravel(), ["Annual_Income", "Loan_Amount", "Total_Assets",
                                "Total_Liabilities", "Monthly_Debt_Payment", "Debt_To_Income_Ratio"]):
    sns.boxplot(x="Loan_Approved", y=c, data=df_clean, ax=ax)
    ax.set_title(c)
plt.tight_layout()
fig.savefig("figures/outlier_boxplots.png", dpi=150)
st.pyplot(fig)
st.caption("Figure 1. Boxplots by approval outcome. (Add your interpretation here after reviewing the plots.)")

skewed = ["Annual_Income", "Loan_Amount", "Total_Assets",
          "Total_Liabilities", "Monthly_Debt_Payment", "Debt_To_Income_Ratio"]

log_check = pd.DataFrame({
    "skew_raw": df_clean[skewed].skew().round(2),
    "skew_after_log1p": np.log1p(df_clean[skewed]).skew().round(2),
})
medians_by_outcome = df_clean.groupby("Loan_Approved")[skewed].median().round(1).T
outlier_table = pd.DataFrame(rows).to_string(index=False)
cap_info = {
    "Annual_Income == 5,000": int((df_clean["Annual_Income"] == 5000).sum()),
    "Loan_Amount == 500,000": int((df_clean["Loan_Amount"] == 500000).sum()),
    "Debt_To_Income_Ratio == 95 (max)": int((df_clean["Debt_To_Income_Ratio"] == 95).sum()),
}

st.write("**Skewness before and after a log transform**")
st.dataframe(log_check)

outlier_prompt = f"""
You are a senior data scientist advising on outlier treatment for a loan dataset.

CONTEXT
- 30,000 loan applications; target is Loan_Approved (Approved/Rejected).
- Research question: which borrower characteristics most strongly predict approval?
- Modeling plan: a stratified 80/20 split, then Logistic Regression (baseline) and Random Forest (advanced). The goal is interpretation, not just accuracy: I will compare logistic regression coefficients (on standardized features) with permutation importance from the forest to decide which borrower characteristics matter most.
- Preprocessing will run inside a scikit-learn Pipeline so that anything fitted from the data (scalers, percentile caps) uses only the training set.
- No missing values or impossible values were found. The data appears to be simulated.

EVIDENCE (IQR rule, 1.5 x IQR)
{outlier_table}

Overall approval rate: {(df_clean['Loan_Approved'] == 'Approved').mean():.3f}

Values sitting exactly at apparent bounds: {cap_info}

Skewness before and after a log1p transform:
{log_check.to_string()}

Median by approval outcome:
{medians_by_outcome.to_string()}

TASK
For each column, recommend a treatment separately for Logistic Regression and for Random Forest (keep as is, log-transform, winsorize, or remove rows), and justify each in one sentence using the evidence. Say whether the outliers look like errors or legitimate extremes. Explain how your recommendation affects the interpretability of the coefficients and importances.

OUTPUT FORMAT
A markdown table with columns: Column | Logistic Regression | Random Forest | Reason. Then 2-3 sentences on risks.
"""

st.write("**AI consultation (Gemini)**")
with st.spinner("Consulting Gemini (the saved answer is reused on later runs)..."):
    try:
        st.markdown(ask_ai("outlier_treatment_v2", outlier_prompt))
    except RuntimeError as e:
        st.warning("The AI service is unavailable right now. Reload in a minute to retry.")
        st.caption(str(e))

st.write("**My evaluation of the AI recommendation**")
st.write("**Decision:** Modified")
st.write(
    "**Decision: Modified.** I accepted most of Gemini's recommendations. "
    "For Random Forest, I agree with keeping every feature as is, since tree-based models "
    "are unaffected by monotonic transformations and robust to outliers. "
    "For Logistic Regression, I agree with keeping Annual_Income, Loan_Amount, Credit_Score, "
    "and Age as is, because their skews are low and their outliers are legitimate extremes "
    "(a log transform would actually worsen Annual_Income, moving its skew from 0.07 to about -1.75). "
    "I also agree with applying log1p to Total_Liabilities and Monthly_Debt_Payment, "
    "whose skews (1.28 and 2.37) are high and which log1p corrects. "
    "I modified two recommendations. First, Gemini proposed winsorizing Debt_To_Income_Ratio at 95, "
    "but 95 is already the column's maximum, so the cap would change nothing. "
    "I apply log1p instead to reduce its skew (2.44). "
    "Second, Gemini kept Savings_Balance (skew 1.51) as is while transforming Total_Liabilities "
    "(skew 1.28), which is inconsistent, so I apply log1p to every column whose skew exceeds about 1. "
    "Total_Assets (skew 0.74) is moderately skewed but below that threshold, so it stays as is. "
    "No rows will be removed, because the outliers appear to be legitimate extremes "
    "and deleting them would discard real borrower profiles. "
    "Values at apparent bounds, such as incomes of exactly 5,000, are kept and "
    "noted as likely artifacts of the simulated data."
)