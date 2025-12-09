                                                 Early Risk Signals (ERS) Prototype
 
                             ~ Early Risk Signals (ERS) — Credit Card Risk Detection Prototype

A lightweight rule-based Early Risk Scoring system designed to identify customers at risk of missing payments using behavioral signals and DPD (Days Past Due) severity.
Built using Python + Streamlit, this prototype provides:

**Real-time ERS scoring

Interactive analytics & dashboards

Rule-based risk classification

Outreach logging for customer follow-ups

Modern UI with badges, charts, and animations**

~ Features
✅ 1. Upload & Analyze Portfolio

Upload any credit-card customer CSV

Auto-detects column names (normalizes multiple naming styles)

✅ 2. ERS Scoring Engine (P1–P8 + DPD Severity)

Flags calculated:

P1: High Utilisation

P2: Low Payment Ratio

P3: Minimum-Due Trap

P4: Liquidity Stress (High Cash Withdrawal %)

P5: Sudden Spend Drop

P6: Concentrated Merchant Mix

P7: DPD Severity (0/2/4 points)

P8: Overlimit Behaviour

DPD Severity Rule
DPD Value	P7_dpd_severity
0	0
1	2
≥2	4
✅ 3. Final ERS Score & Tier

Score determines risk tier:

ERS Score	Tier	Color
0–3	Low	Blue
4–5	Medium	Yellow
≥6	High	Red
✅ 4. Customer Search & Detail View

Shows flags, metrics, ERS score, risk level

Auto-suggested recommended action

Outreach simulation (SMS / Call)

✅ 5. Interactive Visual Dashboard

Includes:

Pie chart of risk tiers

Scatter plot (Utilisation vs ERS Score)

Top-K risky customers export

Full portfolio AG-Grid table

✅ 6. Outreach Logging

Actions are saved in:

outreach_log.csv


Stored automatically during interaction.

📂 Project Structure
├── app.py                 # Main Streamlit application
├── sample_ers_input.csv   # Sample customer dataset
├── outreach_log.csv       # Auto-created action log
└── README.md              # Project documentation

🧠 ERS Scoring Flow (Architecture)
Customer Input CSV
        ↓
Column Normalization
        ↓
Behavioral Feature Extraction
 (Utilisation, Payment Ratio, Spend Drop, Merchant Mix, Cash Withdrawal)
        ↓
DPD Severity Conversion (0, 2, 4)
        ↓
P1–P8 Flag Creation
        ↓
Weighted ERS Score Calculation
        ↓
Risk Tier Assignment (Low/Medium/High)
        ↓
Dashboard + Customer Detail + Outreach Logging

🛠️ Tech Stack
Category	Tools:-
Frontend UI	Streamlit
Backend Logic	Python
Data Processing	Pandas, NumPy
Visualization	Plotly
UI Enhancers	Lottie Animations, AG-Grid
Storage	CSV (Prototype Mode)
📥 How to Run Locally
1. Clone the repository
git clone https:https://github.com/chandrasekhar02/ERS-PROTOTYPE
cd ERS-PROTOTYPE

2. Install dependencies
pip install -r requirements.txt

3. Run the app
streamlit run app.py

🧪 Sample Dataset

A demo file sample_ers_input.csv is included so the app can run without uploading anything.

📌 Key Business Value
✔ Early detection of customers likely to miss payments
✔ Prioritization of outreach to High-risk accounts
✔ Helps reduce delinquency roll-rate
✔ Rule-based system — transparent, explainable, audit-friendly
✔ Real-time insights through a simple interface

This prototype mimics how banks build Early Warning Systems (EWS) for credit cards.

🚀 Future Enhancements 
 Replace CSV with database (Postgres / Snowflake)
 Add user authentication (RMs, analysts, managers)
 Upgrade to ML model for risk prediction
 Add trend graphs for each customer
 Automated outreach triggers
