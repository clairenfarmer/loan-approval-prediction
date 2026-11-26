# loan-approval-prediction

An AI assisted data science project that investigates which borrower characteristics most strongly predict loan approval. Results are presented in an interactive Streamlit report.

## Dataset
- **Name:** Loan Approval Dataset 2026 | Credit Risk & Bank A
- **Source:** https://www.kaggle.com/datasets/mmumairkhattak/loan-approval-dataset-2026-credit-risk-and-bank-a
- **Size:** 30,000 rows, 36 columns
- **Contents:** Applicant demographics, income, employment, credit history, banking behavior, loan request details, collateral indicators, risk scores, and approval outcomes.
- **Target variable:** `Loan_Approved`



## How to Run

### Prerequisites
- Python 3.14 (3.12 or 13 should also work, 3.15 may not)
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