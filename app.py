import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

# =====================================
# PAGE CONFIG (PREMIUM UI)
# =====================================
st.set_page_config(
    page_title="Aadhaar Enrolment Dashboard",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.big-font {
    font-size:28px !important;
    font-weight:600;
}
.metric-box {
    background-color:#161b22;
    padding:20px;
    border-radius:15px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='big-font'>📊 Aadhaar Enrolment Trends & Insights</div>", unsafe_allow_html=True)
st.caption("Real-Time Government Data Analytics | Premium Data Science Project")

# =====================================
# LOAD DATA (ERROR-FREE)
# =====================================
@st.cache_data
def load_data():
    df = pd.read_csv("api_data_aadhar_enrolment_0_500000.csv")

    # Rename to standard names
    df = df.rename(columns={
        'date': 'Date',
        'state': 'State',
        'district': 'District',
        'age_0_5': 'Age_0_5',
        'age_5_17': 'Age_5_17',
        'age_18_greater': 'Age_18_plus'
    })

    # Date conversion
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # 🔥 CALCULATE TOTAL (THIS FIXES YOUR ERROR)
    df['Total_Enrolments'] = (
        df['Age_0_5'] +
        df['Age_5_17'] +
        df['Age_18_plus']
    )

    return df.dropna(subset=['Date'])

df = load_data()

# =====================================
# SIDEBAR FILTERS
# =====================================
st.sidebar.header("🔎 Filters")

states = sorted(df['State'].unique())
selected_states = st.sidebar.multiselect(
    "Select State(s)", states, default=states
)

min_date, max_date = df['Date'].min(), df['Date'].max()
start_date, end_date = st.sidebar.date_input(
    "Select Date Range", [min_date, max_date]
)

filtered_df = df[
    (df['State'].isin(selected_states)) &
    (df['Date'] >= pd.to_datetime(start_date)) &
    (df['Date'] <= pd.to_datetime(end_date))
]

# =====================================
# KPI METRICS (ANIMATED LOOK)
# =====================================
c1, c2, c3 = st.columns(3)

c1.metric("🧾 Total Enrolments", f"{int(filtered_df['Total_Enrolments'].sum()):,}")
c2.metric("🏛️ States Covered", filtered_df['State'].nunique())
c3.metric("📍 Districts Covered", filtered_df['District'].nunique())

st.divider()

# =====================================
# TREND GRAPH
# =====================================
st.subheader("📈 Enrolment Trend Over Time")

trend_df = filtered_df.groupby('Date')['Total_Enrolments'].sum().reset_index()

fig1, ax1 = plt.subplots(figsize=(12,5))
sns.lineplot(data=trend_df, x='Date', y='Total_Enrolments', ax=ax1)
ax1.set_title("Aadhaar Enrolment Trend")
ax1.set_xlabel("Date")
ax1.set_ylabel("Total Enrolments")
plt.xticks(rotation=45)

st.pyplot(fig1)

# =====================================
# AGE DISTRIBUTION
# =====================================
st.subheader("👶🧒🧑 Age Group Distribution")

age_data = [
    filtered_df['Age_0_5'].sum(),
    filtered_df['Age_5_17'].sum(),
    filtered_df['Age_18_plus'].sum()
]

fig2, ax2 = plt.subplots()
ax2.pie(
    age_data,
    labels=["0–5 Years", "5–17 Years", "18+ Years"],
    autopct="%1.1f%%",
    startangle=90
)
ax2.set_title("Age Group Share")

st.pyplot(fig2)

# =====================================
# ML FORECASTING (PRO LEVEL)
# =====================================
st.subheader("🤖 ML Forecast: Future Aadhaar Enrolments")

trend_df['t'] = np.arange(len(trend_df))

X = trend_df[['t']]
y = trend_df['Total_Enrolments']

model = LinearRegression()
model.fit(X, y)

future_steps = 6
future_t = np.arange(len(trend_df), len(trend_df) + future_steps).reshape(-1, 1)
future_preds = model.predict(future_t)

forecast_df = pd.DataFrame({
    "Month_Index": future_t.flatten(),
    "Predicted_Enrolments": future_preds.astype(int)
})

st.dataframe(forecast_df)

# =====================================
# FOOTER
# =====================================
st.markdown("---")
st.markdown("""
✅ **Company-Grade Dashboard**  
🧠 **Role:** Data Scientist  
🛠️ **Tech:** Python, Streamlit, ML  

""")

