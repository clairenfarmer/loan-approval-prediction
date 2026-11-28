import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import HistGradientBoostingRegressor
from cache_helper import ask_ai, load_log
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve)
from sklearn.inspection import permutation_importance
from sklearn.base import clone

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
    "Gemini suggested to KEEP Interest_Rate, but I disagree. While it has a moderate correlation with borrower features (R² about 0.67). ",
    "it is unclear wether it was set before or after the approval decision. For this reason, I am excluding it to avoid possible leakage.",
    "especially since Interest_Rate is not a borrower characteristic."
)   

# Drop columns based on my evaluation
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

skewed = ["Annual_Income", "Loan_Amount", "Total_Assets",
          "Total_Liabilities", "Monthly_Debt_Payment", "Debt_To_Income_Ratio"]

box_cols = skewed + ["Savings_Balance"]

fig, axes = plt.subplots(2, 4, figsize=(18, 7))
for ax, c in zip(axes.ravel(), box_cols):
    sns.boxplot(x="Loan_Approved", y=c, data=df_clean, ax=ax)
    ax.set_title(c)
axes.ravel()[-1].axis("off")   # 7 plots in 8 slots
plt.tight_layout()
fig.savefig("figures/outlier_boxplots.png", dpi=150)
st.pyplot(fig)
st.caption("Figure 1. Boxplots by approval outcome. Rejected applicants have higher loan amounts, total liabilities, monthly debt payments, and DTI, while approved applicants have higher annual income, total assets, and savings balances. Outliers echo the tail of a right skewed distribution for most plots. One interesting outlier group is approved applicants with very low annual income. This is a small group pattern.")

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

hist_cols = skewed + ["Savings_Balance"]   

fig2, axes2 = plt.subplots(2, 4, figsize=(18, 7))
for ax, c in zip(axes2.ravel(), hist_cols):
    sns.histplot(df_clean, x=c, hue="Loan_Approved", bins=50, log_scale=True,
                 element="step", stat="density", common_norm=False, ax=ax)
    ax.set_title(f"{c} (log scale)")
axes2.ravel()[-1].axis("off")   # 7 plots in 8 slots
plt.tight_layout()
fig2.savefig("figures/outlier_histograms.png", dpi=150)
st.pyplot(fig2)
st.caption("Figure 2. Distributions on a log x-axis, by approval outcome. Total_Liabilities, Monthly_Debt_Payment, and Debt_To_Income_Ratio look symetric after the log transformation. For income, loan amount, and total assets, the log transformation overcorrects, so they should be kept as is. Rejected applicants are shifted toward higher debt measures and lower income, assets, and savings, but there is substantial overlap. Small spikes near $5,000 income and $1,000 loan amount reflect minimum values in the data.")

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
    "I accepted most of Gemini's recommendations. "
    "For Random Forest, I agree with keeping every feature as is, since tree-based models "
    "are unaffected by monotonic transformations and robust to outliers. "
    "For Logistic Regression, I agree with keeping Annual_Income, Loan_Amount, Credit_Score, "
    "and Age as is, because their skews are low and their outliers are legitimate extremes "
    "(a log transform would actually worsen Annual_Income, moving its skew from 0.07 to about -1.75). "
    "I also agree with keeping Total_Assets (skew 0.74) and Savings_Balance (skew 0.87) as is. "
    "Both skews are below about 1, and the high values are legitimate wealthy applicants, "
    "so these outliers carry real signal. A log transform would also worsen Savings_Balance "
    "(skew -4.11), because about 2.5% of applicants have a balance of exactly 0. "
    "I also agree with applying log1p to Total_Liabilities and Monthly_Debt_Payment, "
    "whose skews (1.28 and 2.37) are high and which log1p reduces. "
    "I modified one recommendation. Gemini proposed winsorizing Debt_To_Income_Ratio at 95, "
    "but 95 is already the column's maximum, so the cap would change nothing. "
    "I apply log1p instead to reduce its skew (2.44). "
    "Overall, I apply log1p only to columns whose skew exceeds about 1. "
    "No rows will be removed, because the outliers appear to be legitimate extremes "
    "and deleting them would discard real borrower profiles. "
    "Values at apparent bounds, such as incomes of exactly 5,000, are kept and "
    "noted as likely artifacts of the simulated data. "
    "(Note that Monthly_Debt_Payment and Loan_Amount were later removed as inputs in section 5.)"
)

# ---------------------------------------------------------------
# 3. EDA
# ---------------------------------------------------------------
st.header("3. Exploratory Data Analysis")
y = (df_clean["Loan_Approved"] == "Approved")
num_cols = df_clean.select_dtypes(include="number").columns.tolist()
cat_cols_clean = [c for c in df_clean.columns if c not in num_cols and c != "Loan_Approved"]

# 3a. Target balance
st.subheader("3a. Outcome Balance")
fig, ax = plt.subplots(figsize=(3, 2))
df_clean["Loan_Approved"].value_counts().plot.bar(ax=ax)
ax.set_ylabel("Applications")
st.pyplot(fig, use_container_width=False)
fig.savefig("figures/target_balance.png", dpi=150)
st.caption("Figure 3. Approved vs. rejected counts. Shows (57.6% approved, 42.4% rejected). The dataset is not perfectly balanced, but the imbalance is not extreme.")

# 3b. Descriptive statistics and group-by aggregation
st.subheader("3b. Descriptive Statistics by Outcome")
st.dataframe(df_clean[num_cols].describe().T.round(2))
st.write("Mean and median of each numeric variable by approval outcome (group-by aggregation):")
st.dataframe(df_clean.groupby("Loan_Approved")[num_cols].agg(["mean", "median"]).round(1).T)

# 3c. Approval rate by category (group-by)
st.subheader("3c. Approval Rate by Category")
fig, axes = plt.subplots(4, 3, figsize=(15, 14))
for ax, c in zip(axes.ravel(), cat_cols_clean):
    y.groupby(df_clean[c]).mean().sort_values().plot.barh(ax=ax)
    ax.axvline(y.mean(), color="red", linestyle="--")
    ax.set_title(c)
    ax.set_xlabel("Approval rate")
plt.tight_layout()
st.pyplot(fig)
fig.savefig("figures/approval_rate_by_category.png", dpi=150)
st.caption("Figure 4. Approval rate within each category; the dashed line is the overall rate. Gender, Property_Ownership, Residential_Status, Application_Channel, Loan_Purpose, Region and Marital_Status show little to no variation in approval rate. Collateral, Co_Applicant, and Education_Level show some gaps (section 4 later clarifies these gaps as small to neglible), while Employment_Status and Previous_Loan_Status show the largest gaps. Applicants with a Co_Applicant or Collateral are more likely to be approved, and unemployed applicants or applicants who have previously defaulted are much less likely to be approved.")

# 3d. Approval rate by binned numeric variables
st.subheader("3d. Approval Rate Across Credit Score and Debt-to-Income Ratio")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

credit_bins = pd.cut(df_clean["Credit_Score"], bins=8, precision=0)
y.groupby(credit_bins, observed=True).mean().plot(marker="o", ax=axes[0])
axes[0].set_title("Approval rate by Credit_Score bin")
axes[0].set_ylabel("Approval rate")

dti_bins = pd.qcut(df_clean["Debt_To_Income_Ratio"], q=8, precision=1)
y.groupby(dti_bins, observed=True).mean().plot(marker="o", ax=axes[1])
axes[1].set_title("Approval rate by Debt_To_Income_Ratio octile")
axes[1].set_ylabel("Approval rate")

for ax in axes:
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_ha("right")
plt.tight_layout()
st.pyplot(fig)
fig.savefig("figures/approval_rate_by_bins.png", dpi=150)
st.caption("Figure 5. Approval rate across bins. Credit_Score shows a strong S-shaped relationship with approval: near 0% below a score of about 575, a jump to roughly 42% in the 575 to 645 bin, and about 88% in the top bin. Debt_To_Income_Ratio shows a steady negative relationship: approval falls from about 77% in the lowest octile to about 30% in the highest.")
# 3e. Correlation heatmap
st.subheader("3e. Correlation Heatmap")
fig, ax = plt.subplots(figsize=(13, 10))
sns.heatmap(df_clean[num_cols].corr(), annot=True, fmt=".2f", annot_kws={"size": 7},
            cmap="coolwarm", center=0, ax=ax)
st.pyplot(fig)
fig.savefig("figures/correlation_heatmap.png", dpi=150)
st.caption("Figure 6. Pearson correlation heatmap of numeric features. Credit_History_Years and Bank_Account_Age_Years (0.94), and DTI with Monthly_Debt_Payment (0.81), are highly correlated, which creates multicollinearity concerns for interpreting logistic regression coefficients and may split importance across features in the Random Forest. ")

st.subheader("3f. Income vs. Loan Amount")
fig, ax = plt.subplots(figsize=(7, 5))
sns.scatterplot(data=df_clean.sample(3000, random_state=42), x="Annual_Income",
                y="Loan_Amount", hue="Loan_Approved", alpha=0.5, ax=ax)
st.pyplot(fig)
fig.savefig("figures/income_vs_loan_amount.png", dpi=150)
st.caption("Figure 7. Annual income vs. loan amount for a random sample of 3,000 applications, colored by outcome. Loan amounts rise with income, and the upper limit appears to scale with income. Rejected applicants cluster toward lower income and larger loans, while approved applicants cluster toward higher income and smaller loans, suggesting that the size of the loan relative to income matters more than either value alone. The groups overlap substantially, so income and loan amount alone do not determine approval.")

# ---------------------------------------------------------------
# 4. Statistical Analysis
# ---------------------------------------------------------------
st.header("4. Statistical Analysis")
test_vars = ["Credit_Score", "Annual_Income", "Debt_To_Income_Ratio",
             "Late_Payments_Last_2_Years", "Credit_History_Years"]
test_info = pd.DataFrame({
    "skew": df_clean[test_vars].skew().round(2),
    "share_zero": (df_clean[test_vars] == 0).mean().round(3),
    "n_unique_values": df_clean[test_vars].nunique(),
})
group_sizes = df_clean["Loan_Approved"].value_counts().to_dict()

stat_prompt = f"""
You are a statistician advising on hypothesis tests for a loan dataset.

CONTEXT
- Target: Loan_Approved (Approved/Rejected). Group sizes: {group_sizes}.
- Research question: which borrower characteristics most strongly predict approval?
- Numeric variables I want to compare between approved and rejected applicants:
{test_info.to_string()}
- Categorical variables (each with 2 to 7 categories): Employment_Status, Collateral, Loan_Purpose, Education_Level, Gender, Region.
- With 30,000 rows I expect nearly every p-value to be very small.

TASK
Recommend which formal tests to use for the numeric and for the categorical variables, and justify each choice using the evidence above. Say whether any variable needs a different test, how to handle multiple comparisons, and which effect size to report alongside each p-value.

OUTPUT FORMAT
A markdown table with columns: Variable type | Test | Why | Effect size. Then 2-3 sentences on risks.
"""

st.write("**AI consultation (Gemini)**")
with st.spinner("Consulting Gemini (the saved answer is reused on later runs)..."):
    try:
        st.markdown(ask_ai("stat_test_selection", stat_prompt))
    except RuntimeError as e:
        st.warning("The AI service is unavailable right now. Reload in a minute to retry.")
        st.caption(str(e))

st.write("**My evaluation of the AI recommendation**")
st.write("**Decision:** Modified")
st.write(
    "I accepted Gemini's decision to use the Welch's t-test for the numeric variables, chi-squared tests for the categorical variables, effect sizes alongisde p-values, and ",
    "a multiple testing correction (I am, however, choosing standard Bonferroni for simplicity). I rejected its recommendation of the Mann-Whitney U test for some variables as that test was not covered in the course and Gemini's argument ",
    "for chosing it was inconsistent - It said the CLT justifies the t-test for N = 30,000. I will instead use Welch's t-test for all 5 numeric variables, report medians because of the skew, ",
    "and use Spearman correlation as a robustness check for Late_Payments (54% 0 count). I also rejected its cutoffs for judging importance since that would label most real effects as ",
    "unimportant. I will check the chi-square expected-count assumption directly instead. Gemini counted 11 tests as provided in my prompt, however I added Previous_Loan_Status and Co_Applicant tests later ",
    "after revisiting my Figure 4 interpretation."
)   

appr = df_clean[df_clean["Loan_Approved"] == "Approved"]
rej = df_clean[df_clean["Loan_Approved"] == "Rejected"]
num_tests = ["Credit_Score", "Annual_Income", "Debt_To_Income_Ratio",
             "Late_Payments_Last_2_Years", "Credit_History_Years"]
cat_tests = ["Employment_Status", "Previous_Loan_Status", "Co_Applicant", "Collateral",
             "Loan_Purpose", "Education_Level", "Gender", "Region"]

def size_label(x, small, medium, large):
    x = abs(x)
    return "negligible" if x < small else "small" if x < medium else "medium" if x < large else "large"

results = []

# Welch t-tests
for c in num_tests:
    a, r = appr[c], rej[c]
    t, p = stats.ttest_ind(a, r, equal_var=False)
    pooled = np.sqrt(((len(a)-1)*a.var() + (len(r)-1)*r.var()) / (len(a)+len(r)-2))
    d = (a.mean() - r.mean()) / pooled
    results.append({"variable": c, "test": "Welch t-test", "statistic": round(t, 2), "p_raw": p,
                    "effect": round(d, 3), "effect type": "Cohen's d", "size": size_label(d, 0.2, 0.5, 0.8),
                    "mean approved": round(a.mean(), 2), "mean rejected": round(r.mean(), 2),
                    "median approved": round(a.median(), 2), "median rejected": round(r.median(), 2)})

# Chi-square tests
min_expected = {}
for c in cat_tests:
    ct = pd.crosstab(df_clean[c], df_clean["Loan_Approved"])
    chi2, p, dof, expected = stats.chi2_contingency(ct)
    v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
    min_expected[c] = round(expected.min(), 1)
    results.append({"variable": c, "test": "Chi-square", "statistic": round(chi2, 2), "p_raw": p,
                    "effect": round(v, 3), "effect type": "Cramér's V", "size": size_label(v, 0.1, 0.3, 0.5)})

res = pd.DataFrame(results)
m = len(res)
res["p_bonferroni"] = (res["p_raw"] * m).clip(upper=1).map(lambda x: f"{x:.2e}")
res["p_raw"] = res["p_raw"].map(lambda x: f"{x:.2e}")
st.dataframe(res)
st.caption(f"Bonferroni correction across {m} tests (adjusted alpha = {0.05/m:.4f}). "
           "Effect-size labels follow Cohen's conventions and are descriptive, not pass/fail cutoffs.")
st.write("Smallest expected cell count in each chi-square table (should be at least 5):", min_expected)

for c in ["Previous_Loan_Status", "Co_Applicant"]:
    st.write(f"Approval rate by {c}:")
    st.dataframe(
        df_clean.groupby(c)["Loan_Approved"]
        .apply(lambda s: (s == "Approved").mean())
        .round(3)
        .rename("approval rate")
        .to_frame()
    )

# Correlation analysis (robustness check for the numeric variables)
y_bin = (df_clean["Loan_Approved"] == "Approved").astype(int)
corr_rows = []
for c in num_tests:
    rho, p = stats.spearmanr(df_clean[c], y_bin)
    corr_rows.append({"variable": c, "Spearman rho with approval": round(rho, 3), "p": f"{p:.2e}"})
st.write("Spearman correlation of each numeric variable with approval (robustness check):")
st.dataframe(pd.DataFrame(corr_rows))

st.markdown("""
**Interpretation.** All 13 tests are judged against a Bonferroni-adjusted threshold of 0.05 / 13 = 0.0038.
P-valuesshown as "< 0.001" are smaller than the display limit of the software. With 30,000 applicants, nearly every 
test is statistically significant, so the effect sizes (Cohen's d and Cramer's V for numeric and categorical variables 
respectively) matter more than p-values.

**Welch's t-tests.** Each test compares the mean of a numeric variable between 2 independent groups - approved and rejected 
appplicants. With large groups, as in this dataset, the CLT makes the comparison of means reliable even for skewed variables.
 For every test, H0 = The mean is the same for approved and rejected applicants.
- **Credit_Score:** t = 90.27, p < 0.001, d = 1.08 (large). Approved applicants average 714.7 vs. 630.6 for rejected
  applicants, making credit score the strongest numeric separator of the outcome.
- **Annual_Income:** t = 33.28, p < 0.001, d = 0.39 (small). Approved applicants earn about 11,700 dollars more on average.
- **Debt_To_Income_Ratio:** t = -47.46, p < 0.001, d = -0.60 (medium). Rejected applicants carry a higher ratio
  (mean 13.8 vs. 8.3; median 10.7 vs. 6.3).
- **Late_Payments_Last_2_Years:** t = -52.83, p < 0.001, d = -0.68 (medium). Rejected applicants average 1.17 late
  payments vs. 0.47 (medians of 1 vs. 0). Because this variable is a zero-heavy count, I checked it with a Spearman
  correlation, which agrees in direction (rho = -0.28).
- **Credit_History_Years:** t = 15.79, p < 0.001, d = 0.18 (very small). Approved applicants have about 1.2 more years of
  credit history on average, a statistically significant but practically small difference.

  **Chi-square tests of independence.** Each test asks whether approval rates differ across the categories of a variable. Both 
  since both variables are categorical, a chi-square test is appropriate here. Every table has expected cell counts above 500, 
  so the assumption of at least 5 is met. For every test, H0 = approval is independent of the variable.
- **Employment_Status:** χ2 = 2073.79 (df = 4), p < 0.001, V = 0.26 (small to moderate). The association is driven mainly
  by unemployed applicants, who are approved far less often than every other group (Figure 4).
- **Previous_Loan_Status:** χ2 = 1712.71 (df = 3), p < 0.001, V = 0.24 (small to moderate). Applicants with a previously
  defaulted loan are approved only about 15% of the time, compared with about 65% of those who paid off a previous loan.
- **Co_Applicant:** χ2 = 138.91 (df = 1), p < 0.001, V = 0.07 (negligible). Applications with a co-applicant are approved
  somewhat more often (about 63% vs. 55%), but the association is weak.
- **Collateral:** χ2 = 307.35 (df = 1), p < 0.001, V = 0.10 (small). Applicants offering collateral are approved more often.
- **Loan_Purpose:** χ2 = 93.68 (df = 6), p < 0.001, V = 0.06 (negligible).
- **Education_Level:** χ2 = 42.03 (df = 3), p < 0.001, V = 0.04 (negligible), with a mild upward gradient from High
  School to PhD.
- **Gender:** χ2 = 0.71 (df = 1), p = 0.40, V = 0.005. There is no evidence that approval differs by gender.
- **Region:** χ2 = 9.33 (df = 4), raw p = 0.053 (adjusted p ≈ 0.69), V = 0.02. There is no evidence of a regional difference.

**Takeaway.** For numeric variables, credit score shows by far the largest difference between approved and rejected  applicants. Late payments and debt-to-income
 ratio are the next most significant. For categorical variables, employment status and previous loan status show the largest associations. Gender 
 and region on the other hand, show no association. These comparrisons cannot separate overlapping effects due to the one-variable-at-a-time 
 method (credit score is correlated with late payments and savings), so the following section will fit multivariable models.
  """)

# ---------------------------------------------------------------
# 5. ML
# ---------------------------------------------------------------
st.header("5. Machine Learning")
st.subheader("5a. Feature Selection")

# Evidence 1: highly correlated numeric pairs
corr = df_clean[num_cols].corr()
pairs = (corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
         .stack().rename("corr").reset_index())
pairs.columns = ["feature_1", "feature_2", "corr"]
high_pairs = pairs[pairs["corr"].abs() > 0.65].sort_values("corr", ascending=False).round(2)

# Evidence 2: VIF, using the planned log1p transforms
log_cols = ["Total_Liabilities", "Monthly_Debt_Payment", "Debt_To_Income_Ratio"]
X_vif = df_clean[num_cols].copy()
X_vif[log_cols] = np.log1p(X_vif[log_cols])
X_vif = (X_vif - X_vif.mean()) / X_vif.std()
vif = pd.Series([variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])],
                index=X_vif.columns).round(1).sort_values(ascending=False)

# Evidence 3: association of each variable with approval (from section 4 and Figure 4)
assoc = res[["variable", "effect type", "effect"]].to_string(index=False)

st.write("Highly correlated pairs (|r| > 0.65):")
st.dataframe(high_pairs)
st.write("Variance inflation factors (after planned log1p transforms and standardization):")
st.dataframe(vif.rename("VIF"))

feature_prompt = f"""
You are a senior data scientist advising on feature selection for a loan approval model.

CONTEXT
- 30,000 applications; target is Loan_Approved (57.6% Approved, 42.4% Rejected).
- Research question: which borrower characteristics most strongly predict approval? The goal is interpretation, not only accuracy.
- Models: Logistic Regression (baseline, standardized features, log1p on Total_Liabilities, Monthly_Debt_Payment, Debt_To_Income_Ratio) and Random Forest (advanced, raw features). I will compare logistic regression coefficients with permutation importance from the forest.
- Bank outputs (Risk_Score, Default_Risk, Approved_Amount, Interest_Rate) were already removed as leakage.
- Remaining columns: Age, Gender, Marital_Status, Education_Level, Employment_Status, Job_Experience_Years, Annual_Income, Credit_Score, Credit_History_Years, Existing_Loans, Late_Payments_Last_2_Years, Loan_Amount, Loan_Term_Months, Loan_Purpose, Debt_To_Income_Ratio, Savings_Balance, Checking_Balance, Total_Assets, Total_Liabilities, Monthly_Debt_Payment, Property_Ownership, Residential_Status, Dependents, Co_Applicant, Collateral, Application_Channel, Region, Bank_Account_Age_Years, Previous_Loan_Status.

EVIDENCE
Highly correlated numeric pairs:
{high_pairs.to_string(index=False)}

Variance inflation factors:
{vif.to_string()}

Association of each tested variable with approval (Cohen's d for numeric, Cramér's V for categorical):
{assoc}

Observations from exploratory analysis: gender and region show no association with approval (V about 0.005 and 0.02); approval rises with credit score in an S-shape around scores of 575 to 645; the approved and rejected groups separate along the diagonal of the income vs. loan amount scatterplot, suggesting the loan-to-income ratio matters.

TASK
1. For each highly correlated pair, say which feature to keep or drop for the Logistic Regression model and whether the Random Forest should differ, with a one-sentence reason citing the evidence.
2. Say whether Gender, Age, and Marital_Status should be included as predictors, and what I should do about fairness.
3. Say whether to add a Loan_Amount / Annual_Income ratio feature, and whether any other engineered feature is justified by the evidence.

OUTPUT FORMAT
Three short sections matching the three tasks, each with a table or bullet list, then 2-3 sentences on risks.
"""

st.write("**AI consultation (Gemini)**")
with st.spinner("Consulting Gemini (the saved answer is reused on later runs)..."):
    try:
        st.markdown(ask_ai("feature_selection", feature_prompt))
    except RuntimeError as e:
        st.warning("The AI service is unavailable right now. Reload in a minute to retry.")
        st.caption(str(e))

# extra checks before final decision
st.write("**Extra checks before making a final decision**")
# 1. Fair comparison: effect size of Bank_Account_Age_Years vs. Credit_History_Years
for c in ["Credit_History_Years", "Bank_Account_Age_Years"]:
    a, r = appr[c], rej[c]
    pooled = np.sqrt(((len(a)-1)*a.var() + (len(r)-1)*r.var()) / (len(a)+len(r)-2))
    st.write(f"{c}: Cohen's d = {(a.mean()-r.mean())/pooled:.3f}")

# 2. VIF after the planned drops, with and without the loan-to-income ratio
final_drop = ["Age", "Gender", "Marital_Status", "Monthly_Debt_Payment", "Bank_Account_Age_Years"]
num_final = [c for c in num_cols if c not in final_drop]
log_cols_final = ["Total_Liabilities", "Debt_To_Income_Ratio"]

def vif_table(X):
    X = X.copy()
    X[[c for c in log_cols_final if c in X.columns]] = np.log1p(X[[c for c in log_cols_final if c in X.columns]])
    Xs = (X - X.mean()) / X.std()
    return pd.Series([variance_inflation_factor(Xs.values, i) for i in range(Xs.shape[1])],
                     index=Xs.columns).round(1)

X_base = df_clean[num_final]
X_lti = X_base.assign(Loan_to_Income=df_clean["Loan_Amount"] / df_clean["Annual_Income"])
st.dataframe(pd.DataFrame({"VIF without LTI": vif_table(X_base),
                           "VIF with LTI": vif_table(X_lti)}))

# 3. Loan-to-income effect size
lti = df_clean["Loan_Amount"] / df_clean["Annual_Income"]
a, r = lti[y], lti[~y]
pooled = np.sqrt(((len(a)-1)*a.var() + (len(r)-1)*r.var()) / (len(a)+len(r)-2))
st.write(f"Loan_to_Income: Cohen's d = {(a.mean()-r.mean())/pooled:.3f}")

# 4. Is credit score roughly linear in log-odds? (if so, no spline is needed)
bins = pd.cut(df_clean["Credit_Score"], bins=8, precision=0)
rate = y.groupby(bins, observed=True).mean().clip(0.005, 0.995)
st.dataframe(pd.DataFrame({"approval rate": rate.round(3),
                           "log-odds": np.log(rate / (1 - rate)).round(2)}))

# 5. check final VIFs
final_num = [c for c in num_cols if c not in
             ["Age", "Monthly_Debt_Payment", "Bank_Account_Age_Years", "Loan_Amount"]]
X_final = df_clean[final_num].assign(Loan_to_Income=df_clean["Loan_Amount"] / df_clean["Annual_Income"])
st.write("VIF for the final numeric feature set:")
st.dataframe(vif_table(X_final).rename("VIF"))
st.write(f"Loan_to_Income skew: {X_final['Loan_to_Income'].skew():.2f}")

st.markdown("""
**Decision: Modified.** I accepted most of Gemini's recommendations. I dropped Monthly_Debt_Payment (VIF 33.3, and highly
correlated with Debt_To_Income_Ratio and Total_Liabilities), Bank_Account_Age_Years (r = 0.94 with Credit_History_Years),
and Age (r = 0.78 with Job_Experience_Years). After these drops, the highest remaining VIF among the numeric features was 4.4
(Total_Liabilities).

Gemini recommended keeping Credit_History_Years, and I accepted this for a slightly different reason. It called the feature's
effect "proven" without computing an effect size for Bank_Account_Age_Years. When I checked, both effects were very small
(d = 0.184 vs. 0.165), so I am keeping Credit_History_Years because it is a more direct measure of creditworthiness.

I also accepted excluding Gender, Age, and Marital_Status as predictors on fairness grounds, since these attributes are
commonly treated as protected in lending. I will audit predicted approval rates by gender after fitting the models. I describe
the legal basis less strictly than Gemini did, since legal analysis is not the focus of this project, but I acknowledge that
excluding Age means the models cannot show whether age contributes to the outcome.

I accepted the loan-to-income ratio, because Figure 7 suggests that approval depends on loan size relative to income and the
ratio has a medium effect (d = -0.571), larger than annual income's (d = 0.393). Adding it without modification, however, raised
the VIFs of Loan_Amount (10.9) and Annual_Income (6.6), so I modified the recommendation: the ratio replaces Loan_Amount, and
Annual_Income is kept. With this change, the maximum VIF is 4.4. This takes priority over Gemini's suggestion to keep
Loan_Amount and Total_Liabilities together, although Total_Liabilities is kept. Loan_to_Income has a skew of 0.13 so it is left untransformed.

I rejected the credit score spline. The empirical log-odds of approval rise from -5.29 to 1.97 across eight credit score bins,
roughly one unit per bin, so a linear term is a reasonable approximation. The steps are steeper at the lowest scores and between 506
and 644, but not strongly enough to justify a transformation that would make the coefficient harder to interpret. The Random
Forest will show whether the nonlinearity matters, and I note the possible underfit in the logistic regression as a limitation.
I also rejected the net worth feature (assets minus liabilities), because it is a linear combination of Total_Assets and
Total_Liabilities and would create exact multicollinearity.

Monthly_Debt_Payment and Loan_Amount, which my earlier outlier analysis treated as model inputs, are therefore no longer used.
""")

# Implement decision
DROP_FEATURES = ["Age", "Gender", "Marital_Status",
                 "Monthly_Debt_Payment", "Bank_Account_Age_Years", "Loan_Amount"]
df_model = df_clean.drop(columns=DROP_FEATURES)
audit_gender = df_clean["Gender"]   # same index as df_model; not a model input, used only for the post-model audit
df_model["Loan_to_Income"] = df_clean["Loan_Amount"] / df_clean["Annual_Income"]

NUMERIC_FEATURES = [c for c in df_model.select_dtypes(include="number").columns]
CATEGORICAL_FEATURES = [c for c in df_model.columns
                        if c not in NUMERIC_FEATURES and c != "Loan_Approved"]
assert len(NUMERIC_FEATURES) == 14 and len(CATEGORICAL_FEATURES) == 10
assert not {"Age", "Gender", "Marital_Status", "Monthly_Debt_Payment",
            "Bank_Account_Age_Years", "Loan_Amount"} & set(df_model.columns)
LOG_FEATURES = ["Total_Liabilities", "Debt_To_Income_Ratio"]  # logistic regression only

st.write(f"Final model dataset: {len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)} features "
         f"({len(NUMERIC_FEATURES)} numeric, {len(CATEGORICAL_FEATURES)} categorical).")
st.dataframe(pd.DataFrame({
    "type": ["numeric"] * len(NUMERIC_FEATURES) + ["categorical"] * len(CATEGORICAL_FEATURES),
    "feature": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
}))


st.subheader("5b. Train/Test Split and Preprocessing")

X = df_model.drop(columns=["Loan_Approved"])
y_model = (df_model["Loan_Approved"] == "Approved").astype(int)   # 1 = Approved
X_train, X_test, y_train, y_test = train_test_split(
    X, y_model, test_size=0.2, stratify=y_model, random_state=42)

other_numeric = [c for c in NUMERIC_FEATURES if c not in LOG_FEATURES]

# Logistic regression: log1p on skewed columns, standardize all numerics, one-hot categoricals
prep_lr = ColumnTransformer([
    ("log", Pipeline([("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                      ("scale", StandardScaler())]), LOG_FEATURES),
    ("num", StandardScaler(), other_numeric),
    ("cat", OneHotEncoder(drop="first", sparse_output=False), CATEGORICAL_FEATURES),
])

# Random forest: raw numerics, one-hot categoricals
prep_rf = ColumnTransformer([
    ("num", "passthrough", NUMERIC_FEATURES),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
])

n_encoded_lr = prep_lr.fit(X_train).get_feature_names_out().shape[0]
n_encoded_rf = prep_rf.fit(X_train).get_feature_names_out().shape[0]
st.write(f"Training set: {len(X_train):,} rows; test set: {len(X_test):,} rows (stratified, random_state=42). "
         f"Approval rate: train {y_train.mean():.3f}, test {y_test.mean():.3f}.")
st.write(f"Features after encoding: {n_encoded_lr} (logistic regression), {n_encoded_rf} (random forest).")

st.subheader("5c. Model Setup")

setup_prompt = f"""
You are a senior data scientist advising on how to train and evaluate models for a loan approval dataset.

CONTEXT
- Target: Loan_Approved (1 = Approved, 0 = Rejected). Class balance: {y_model.mean():.1%} approved.
- Research question: which borrower characteristics most strongly predict approval? The goal is interpretation, not only accuracy.
- Split: stratified 80/20, {len(X_train):,} training rows and {len(X_test):,} test rows. I plan 5-fold cross-validation on the training set.
- Features: {len(NUMERIC_FEATURES)} numeric and {len(CATEGORICAL_FEATURES)} categorical, which become {n_encoded_lr} columns for logistic regression and {n_encoded_rf} for the random forest after one-hot encoding.
- Logistic Regression (baseline): numerics standardized; Total_Liabilities and Debt_To_Income_Ratio log1p-transformed first; categoricals one-hot encoded with the first level dropped.
- Random Forest (advanced): raw numerics, one-hot categoricals.
- Planned metrics: accuracy, precision, recall, F1, ROC-AUC, and a confusion matrix on the test set.
- Remaining correlations between features are moderate (maximum VIF {vif_table(X_final).max():.1f}). Previous exploratory analysis found strong effects for Credit_Score, Late_Payments_Last_2_Years, Debt_To_Income_Ratio, Employment_Status, and Previous_Loan_Status.
- I have limited time and compute, so the tuning must be small.

TASK
1. For each model, recommend which hyperparameters to tune and a small search range, and say whether to use grid or randomized search.
2. Say which metric to tune on and which metrics to report, justified by the class balance, and say what each reported metric would tell me in a lending context.
3. Explain how to measure and compare feature importance between the two models, including how to handle the one-hot encoded columns and the remaining correlations.
4. Name the main risks to interpreting the results.

OUTPUT FORMAT
Four short sections matching the four tasks, using tables where helpful, then 2-3 sentences on anything I may have missed.
"""

st.write("**AI consultation (Gemini)**")
with st.spinner("Consulting Gemini (the saved answer is reused on later runs)..."):
    try:
        st.markdown(ask_ai("model_setup", setup_prompt))
    except RuntimeError as e:
        st.warning("The AI service is unavailable right now. Reload in a minute to retry.")
        st.caption(str(e))

st.markdown("""
**Decision: Modified.** I accepted Gemini's core plan: grid search with 5-fold cross-validation, ROC-AUC as the tuning metric
(the classes are 57.6% / 42.4%, so PR-AUC is not needed), a regularized random forest grid (max_depth, min_samples_leaf,
max_features), reporting accuracy, precision, recall, F1, ROC-AUC, and confusion matrices, and measuring importance with permutation importance on the
test set instead of default Gini importance, with a Spearman correlation to compare the two models' rankings.

I corrected several details. Gemini described the grids as "8 fits" and "18 fits," but those are parameter combinations - with
5-fold cross-validation they are 40 and 90 fits. I simplified the logistic regression search to the regularization strength C with the default L2 penalty,
since L1 would require a slower solver, and its justification (stabilizing coefficients at a maximum VIF of 4.4) overstates the
problem. I use permutation importance for both models, not just the forest, so the rankings are measured the same way, with the pipeline
shuffling each original feature so that one-hot columns are grouped automatically. Odds ratios from the logistic regression show direction.
I skipped SHAP to limit complexity and time.

Gemini's business interpretation of the metrics treated the target as default risk, but the target is the bank's approval
decision. Precision is therefore the share of predicted approvals that the bank actually approved, and recall is the share of
approved applicants the model identifies. I rejected its suggestions to tune the decision threshold using a 3x to 5x default cost, which
has no support in my data, and to calibrate probabilities and generate adverse action codes, which are outside the research question. Gemini
did not mention checking for overfitting or a baseline, so I also report training and test AUC and the majority-class accuracy.
""")

st.subheader("5d. Model Training and Tuning")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

pipe_lr = Pipeline([("prep", clone(prep_lr)),
                    ("model", LogisticRegression(max_iter=2000))])
pipe_rf = Pipeline([("prep", clone(prep_rf)),
                    ("model", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1))])

grid_lr = {"model__C": [0.01, 0.1, 1.0, 10.0]}
grid_rf = {"model__max_depth": [6, 10, 15],
           "model__min_samples_leaf": [10, 30, 50],
           "model__max_features": ["sqrt", 0.3]}


@st.cache_resource(show_spinner="Tuning model (only the first run is slow)...")
def tune(name, _pipe, _grid):
    search = GridSearchCV(_pipe, _grid, scoring="roc_auc", cv=cv, n_jobs=-1)
    search.fit(X_train, y_train)
    return search


search_lr = tune("logistic_regression", pipe_lr, grid_lr)
search_rf = tune("random_forest", pipe_rf, grid_rf)
best_lr, best_rf = search_lr.best_estimator_, search_rf.best_estimator_
st.write(f"Best logistic regression parameters: {search_lr.best_params_}")
st.write(f"Best random forest parameters: {search_rf.best_params_}")

st.subheader("5e. Model Evaluation")


def summarize(name, search, best):
    p_test = best.predict_proba(X_test)[:, 1]
    pred = (p_test >= 0.5).astype(int)
    return {"model": name,
            "CV ROC-AUC": round(search.best_score_, 3),
            "train ROC-AUC": round(roc_auc_score(y_train, best.predict_proba(X_train)[:, 1]), 3),
            "test ROC-AUC": round(roc_auc_score(y_test, p_test), 3),
            "accuracy": round(accuracy_score(y_test, pred), 3),
            "precision": round(precision_score(y_test, pred), 3),
            "recall": round(recall_score(y_test, pred), 3),
            "F1": round(f1_score(y_test, pred), 3)}


st.dataframe(pd.DataFrame([summarize("Logistic Regression", search_lr, best_lr),
                           summarize("Random Forest", search_rf, best_rf)]))
st.caption(f"Majority-class baseline accuracy: {max(y_test.mean(), 1 - y_test.mean()):.3f}. "
           "Metrics use a 0.5 threshold on the held-out test set.")

fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for ax, (name, best) in zip(axes[:2], [("Logistic Regression", best_lr), ("Random Forest", best_rf)]):
    cm = confusion_matrix(y_test, (best.predict_proba(X_test)[:, 1] >= 0.5).astype(int))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Rejected", "Approved"], yticklabels=["Rejected", "Approved"])
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
for name, best in [("Logistic Regression", best_lr), ("Random Forest", best_rf)]:
    fpr, tpr, _ = roc_curve(y_test, best.predict_proba(X_test)[:, 1])
    axes[2].plot(fpr, tpr, label=name)
axes[2].plot([0, 1], [0, 1], "k--", label="Chance")
axes[2].set_title("ROC curves")
axes[2].set_xlabel("False positive rate")
axes[2].set_ylabel("True positive rate")
axes[2].legend()
plt.tight_layout()
fig.savefig("figures/model_evaluation.png", dpi=150)
st.pyplot(fig)
st.caption("Figure 8. Test-set confusion matrices (left, middle) and ROC curves (right). Logistic regression classifies 4,866 of 6,000 "
           "applications correctly (1,877 rejections and 2,989 approvals) and the random forest 4,881 (1,838 and 3,043). Both models make more "
           "false approvals (665 and 704 applications the bank rejected but the model approves) than false rejections (469 and 415), which is "
           "why recall is higher than precision. The two ROC curves are nearly identical and rise steeply above the chance line, consistent "
           "with test AUCs of 0.889 for both models.")

st.subheader("5f. Feature Importance")


@st.cache_data(show_spinner="Computing permutation importance...")
def perm_importance(name, _best):
    r = permutation_importance(_best, X_test, y_test, scoring="roc_auc",
                               n_repeats=5, random_state=42, n_jobs=-1)
    return pd.Series(r.importances_mean, index=X_test.columns)


imp = pd.DataFrame({"Logistic Regression": perm_importance("lr", best_lr),
                    "Random Forest": perm_importance("rf", best_rf)})
imp["rank LR"] = imp["Logistic Regression"].rank(ascending=False).astype(int)
imp["rank RF"] = imp["Random Forest"].rank(ascending=False).astype(int)
imp = imp.sort_values("rank RF")
st.dataframe(imp.round(4))
rho, p = stats.spearmanr(imp["Logistic Regression"], imp["Random Forest"])
st.write(f"Spearman rank correlation between the two importance rankings: {rho:.2f} (p = {p:.1e})")

top = imp.assign(avg=imp[["Logistic Regression", "Random Forest"]].mean(axis=1)) \
         .sort_values("avg").tail(12)[["Logistic Regression", "Random Forest"]]
fig, ax = plt.subplots(figsize=(8, 5))
top.plot.barh(ax=ax)
ax.set_xlabel("Drop in test ROC-AUC when the feature is shuffled")
plt.tight_layout()
fig.savefig("figures/feature_importance.png", dpi=150)
st.pyplot(fig)
st.caption("Figure 9. Permutation importance (drop in test ROC-AUC when a feature is shuffled) for the 12 most important features. "
           "Credit_Score dominates in both models (about 0.11 for logistic regression and 0.10 for the random forest), followed by "
           "Loan_to_Income and Previous_Loan_Status. The two rankings agree closely (Spearman correlation 0.93). Logistic regression "
           "assigns larger importance than the forest to most features, notably Debt_To_Income_Ratio and Late_Payments_Last_2_Years, "
           "while Savings_Balance receives about the same importance in both models.")

# Direction and size of effects from the logistic regression
names = best_lr.named_steps["prep"].get_feature_names_out()
coefs = best_lr.named_steps["model"].coef_[0]
odds = pd.DataFrame({"term": names, "coefficient": coefs.round(3), "odds ratio": np.exp(coefs).round(3)})
odds = odds.reindex(odds["coefficient"].abs().sort_values(ascending=False).index).head(20)
st.write("Largest logistic regression effects. Numeric terms are per standard deviation; categorical terms are "
         "relative to the first level alphabetically, which was dropped.")
st.dataframe(odds)

st.subheader("5g. Predicted Approval Rate by Gender (audit only; Gender is not a model input)")
g = audit_gender.loc[X_test.index]
pred_lr = pd.Series((best_lr.predict_proba(X_test)[:, 1] >= 0.5).astype(int), index=X_test.index)
pred_rf = pd.Series((best_rf.predict_proba(X_test)[:, 1] >= 0.5).astype(int), index=X_test.index)
st.dataframe(pd.DataFrame({"actual approval rate": y_test.groupby(g).mean(),
                           "LR predicted approval rate": pred_lr.groupby(g).mean(),
                           "RF predicted approval rate": pred_rf.groupby(g).mean()}).round(3))

st.markdown("""
**Model performance.** Both models perform far above the majority-class baseline of 57.6% accuracy. The logistic regression reaches a test
ROC-AUC of 0.889 and accuracy of 0.811, and the random forest reaches 0.889 and 0.814. The AUCs are identical to three decimals, so the
nonlinear model adds almost nothing over the linear baseline, which suggests the relationships between borrower characteristics and approval
are largely additive. The random forest's training AUC (0.953) is much higher than its test AUC (0.889), but its cross-validated AUC (0.892)
matches the test result, so this overfitting does not hurt generalization. The logistic regression shows almost no such gap (0.895 train, 0.889 test).

The confusion matrices (Figure 8) show how the models err. Of the 6,000 test applications, 3,458 were approved by the bank and 2,542 were rejected.
The logistic regression correctly identifies 2,989 of the approvals (recall 0.864) and 1,877 of the rejections, while the random forest
identifies 3,043 approvals (recall 0.880) and 1,838 rejections. Both models make more false approvals (665 and 704 applications the model
would approve but the bank rejected) than false rejections (469 and 415), so precision (about 0.81 to 0.82) is lower than recall, and the models
reject only about 72 to 74% of the applications the bank actually rejected. This is the behavior a model tends to show when the
majority class is approval and the remaining decision logic is not captured by the available features.

**Feature importance.** The two models agree closely (Spearman rank correlation = 0.93, p < 0.001), although the lower ranks are
dominated by features whose importance is near zero, where the ordering is mostly noise. Credit_Score is by far the most important feature,
followed by Loan_to_Income and Previous_Loan_Status. Employment_Status, Late_Payments_Last_2_Years, Collateral, and Debt_To_Income_Ratio
come next. Region, Education_Level, Property_Ownership, Loan_Purpose, and Dependents contribute essentially nothing, which is consistent with
their negligible effect sizes in the chi-square tests. Annual_Income ranks low even though it differed between groups in the t-test, because
its information is largely captured by Loan_to_Income and Credit_Score, which shows why one-variable comparisons alone can overstate a
feature's importance.

**Direction and size of effects.** In the logistic regression, the largest effects are a previous default (odds ratio 0.05 relative to an
active loan), unemployment (0.10 relative to business owners), and credit score (3.17 per standard deviation). A higher loan-to-income
ratio (0.43 per standard deviation), more late payments (0.52), and a higher debt-to-income ratio (0.57) lower the odds of approval, while
collateral (2.29) and a co-applicant (1.66) raise them. Coefficients are conditional on the other features, so the negative coefficient on
Annual_Income and the positive one on Total_Liabilities should not be read as standalone effects, since the loan-to-income ratio and
debt-to-income ratio already contain these variables.

**Gender audit.** Gender was not a model input, so any difference in predictions would have to arise through correlated features.
In the test set, the actual approval rate is 57.4% for women and 57.8% for men. The logistic regression predicts approval for
61.7% of women and 60.3% of men, and the random forest for 63.2% and 61.9%. The gaps between genders (1.4 and 1.3 percentage points)
are smaller than the roughly 2.5-point margin of sampling noise expected with about 3,000 applicants per group, so this audit finds no
evidence that the models treat the genders differently. This agrees with the chi-square test on the full data, which found no
association between gender and approval (V = 0.005). The audit has limits: it compares overall predicted approval rates and does not compare
error rates between groups, and it covers only gender, since Age and Marital_Status were also excluded from the models but were not audited.
""")

# ---------------------------------------------------------------
# 6. Final Conclusions
# ---------------------------------------------------------------
st.header("6. Final Conclusions")
st.markdown("""
**Answer to the research question.** Credit score is the borrower characteristic that most strongly predicts loan approval.
It ranks first in both models (permutation importance of 0.106 for logistic regression and 0.101 for the random forest, more than
twice that of any other feature). It also had the largest effect in the statistical tests (Cohen's d = 1.08; approved applicants
average 715 vs. 631 for rejected applicants), and one standard deviation of credit score multiplies the odds of approval by 3.17.
After credit score, the strongest predictors are the loan-to-income ratio (relative loan size), previous loan status, employment status,
late payments, debt-to-income ratio, and collateral. A previous default (odds ratio 0.05, approval rate about 16%) and unemployment
(odds ratio 0.10, approval rate about 14%) are the most severe negative signals. Gender, region, education level, and loan purpose
contribute essentially nothing.

**Agreement between methods.** The statistical tests, the logistic regression, and the random forest tell a consistent story.
The two models rank the features almost identically (Spearman correlation 0.93) and reach the same test ROC-AUC (0.889), so the
relationships appear to be largely additive and a nonlinear model adds little. The one-variable tests and the multivariable models disagree in
one useful way: Annual_Income differs significantly between approved and rejected applicants (d = 0.39) but ranks low in
both models, because the loan-to-income ratio and credit score carry most of its information. This shows why a significant
difference in a single comparison does not by itself make a feature a strong predictor.

**Fairness check.** Gender was excluded from the models, and the audit found no evidence of different treatment (predicted approval
rates differ by 1.3 to 1.4 percentage points, within sampling noise), in line with the chi-square result (V = 0.005).

**Limitations.**
- *Synthetic data.* The data appears to be simulated (values are clipped at fixed bounds, such as incomes of exactly 5,000), so
  the results describe this dataset and should not be read as a description of real lending decisions.
- *Removed columns.* Approved_Amount, Default_Risk and Risk_Score were removed as post-decision variables, and Interest_Rate was removed
  because it may be set after the decision. If it is in fact a legitimate input, excluding it may understate how much of the
  decision the available features can explain. The results describe borrower characteristics, not the bank's full process.
- *Predictive ceiling and over-approval.* Both models misclassify about 19% of test applications, and both approve more applicants than
  the bank did (61 to 63% predicted vs. 57.6% actual), so the available features leave part of the decision unexplained.
- *Correlated features.* When features overlap, permutation importance is shared between them (for example Annual_Income and
  Loan_to_Income), so individual rankings, especially among the low-ranked features, should be read with care. Loan_to_Income also
  replaces Loan_Amount, so its importance reflects loan size and income together.
- *Linear credit score.* The logistic regression treats credit score linearly, although the empirical log-odds are steeper at
  the lowest scores. The random forest's matching AUC suggests the effect on performance is small, but it is a possible
  underfit.
- *Excluded demographics.* Age, Gender and Marital_Status were left out on fairness grounds, so the models cannot show whether
  age contributes to the outcome. The gender audit compares overall predicted rates, not error rates, and does not cover age or marital status.
- *Single split and association only.* Results come from one stratified train-test split, and importance values have no confidence
  intervals. All findings are associations, not causal effects.
""")

# ---------------------------------------------------------------
# 7. AI Assistance Summary
# ---------------------------------------------------------------
st.header("7. AI Assistance Summary")
st.markdown("""
Google Gemini was consulted through LangChain at five decision points. Each prompt and response is stored in `report/ai_log.json`.
The first two steps used Gemini 3.5 Flash-Lite and the last three used Gemini 3.5 Flash; the model name is recorded with each entry.
In every case I checked Gemini's claims against the data before deciding, and I modified every recommendation in some way; none was accepted unchanged.

| Step | What Gemini recommended | My decision | Main reason |
|---|---|---|---|
| Leakage review | Remove Approved_Amount, Default_Risk and Risk_Score; keep Interest_Rate | **Modified**: accepted the three removals and rejected keeping Interest_Rate | Approved_Amount is 0 for every rejected application, Default_Risk is a binned Risk_Score, and Risk_Score is predictable from borrower features (R² about 0.89). Interest_Rate may be set after the decision, so I excluded it as a precaution. Application_ID and Monthly_Income (exactly Annual_Income / 12) were also dropped. |
| Outlier treatment | Keep most columns as is, log-transform Total_Liabilities, winsorize Debt_To_Income_Ratio at 95 | **Modified**: accepted most; replaced the DTI cap with log1p | 95 is already the maximum DTI, so the cap would change nothing. log1p is applied only where skew exceeds about 1, and no rows were removed because the outliers look like legitimate extremes. |
| Statistical test selection | Welch's t-test for symmetric variables, Mann-Whitney U for skewed ones, chi-square for categorical variables, effect sizes, BH or Holm correction, importance cutoffs of \\|d\\| > 0.5 and V > 0.3 | **Modified**: accepted Welch's t-test, chi-square and effect sizes; rejected Mann-Whitney and the cutoffs; used Bonferroni | Mann-Whitney was not covered in the course, and Gemini justified the t-test with the CLT in the same answer, which was inconsistent. The cutoffs would label most real effects unimportant. I used Welch's t-test for all five numeric variables with Spearman as a robustness check, and I added chi-square tests for Previous_Loan_Status and Co_Applicant (which my prompt had omitted), giving 13 tests. |
| Feature selection | Drop collinear features, keep Credit_History_Years, add a loan-to-income ratio, add a credit score spline, add net worth, exclude demographics | **Modified**: accepted most; rejected the spline and net worth; adjusted the ratio | Dropped Monthly_Debt_Payment (VIF 33.3), Bank_Account_Age_Years and Age. The ratio replaces Loan_Amount to keep the maximum VIF at 4.4. The empirical log-odds are close to linear, and net worth is an exact linear combination of Total_Assets and Total_Liabilities. |
| Model setup | Grid search with 5-fold CV on ROC-AUC, regularized random forest, permutation importance, decision-threshold tuning, calibration | **Modified**: accepted the core plan; corrected details and rejected the extras | Gemini counted "8 and 18 fits" but the grids need 40 and 90 fits, treated the target as default risk (it is the bank's approval decision), and suggested a cost-based threshold with no support in my data. I added train/test AUC and the majority-class baseline. |

**What worked and what did not.** Gemini was useful for generating a checklist of issues to investigate (leakage candidates, test assumptions,
collinearity) and for suggesting the loan-to-income feature, which became the second most important predictor. Its answers
contained errors that I found only by checking them against the data, including a cap that does nothing, an inconsistent justification for a test,
an unsupported "proven" effect, and a business interpretation that did not match the target.
The prompts also shaped the answers: two categorical variables were missing from my statistical-test prompt, so Gemini never advised on them.
I treated Gemini's output as a starting point for decisions that I made and justified myself.
""")