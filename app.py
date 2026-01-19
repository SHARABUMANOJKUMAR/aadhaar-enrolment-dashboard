# ================== IMPORTS ==================
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="Aadhaar Enrolment Dashboard",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
body { background-color: #0e1117; }
.big-font { font-size:28px !important; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ================== LOAD DATA ==================
@st.cache_data
def load_data():
    df = pd.read_csv("aadhaar_clean1_states.csv")

    df = df.rename(columns={
        'date': 'Date',
        'state': 'State',
        'district': 'District',
        'age_0_5': 'Age 0-5',
        'age_5_17': 'Age 5-17',
        'age_18_greater': '18+ Adults'
    })

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)
    df = df.dropna(subset=['Date'])

    for c in ['Age 0-5', 'Age 5-17', '18+ Adults']:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    df['Total_Enrolments'] = df['Age 0-5'] + df['Age 5-17'] + df['18+ Adults']
    return df[df['Total_Enrolments'] > 0]

df = load_data()

# ================== SIDEBAR CONTROLS ==================
st.sidebar.header("🔎 Controls")

states = sorted(df['State'].unique())
selected_states = st.sidebar.multiselect(
    "Select State(s)",
    states,
    default=states,
    placeholder="Choose one or more states"
)

model_choice = st.sidebar.radio(
    "Forecasting Model",
    ["Auto (Best)", "ML (Linear Regression)", "ARIMA (Time Series)"]
)

filtered_df = df[df['State'].isin(selected_states)]

# ================== EMPTY STATE GUARD ==================
if filtered_df.empty:
    st.markdown("""
    <div style="
        background-color:#161b22;
        padding:30px;
        border-radius:15px;
        text-align:center;
        margin-top:30px;
        border:1px solid #30363d;
    ">
        <h2 style="color:#58a6ff;">📍 Please Select at Least One State</h2>
        <p style="font-size:16px; color:#c9d1d9;">
            Use the sidebar on the left to select one or more states<br>
            to view Aadhaar enrolment trends, forecasts, and insights.
        </p>
        <p style="color:#8b949e; font-size:14px;">
            This dashboard supports multi-state comparison and policy analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ================== PAGE HEADER ==================
st.markdown("<div class='big-font'>📊 Aadhaar Enrolment Trends & Insights</div>", unsafe_allow_html=True)
st.caption("Government-Grade Data Science | Forecasting • Segmentation • Policy Intelligence")

# ================== ANIMATED KPI METRICS ==================
import time

def animated_metric(container, label: str, value: int):
    """Display animated counter for KPI metrics"""
    placeholder = container.empty()
    
    # Animate from 0 to value
    if value > 0:
        step = max(1, value // 20)  # 20 frames animation
        for i in range(0, value + 1, step):
            placeholder.metric(label, f"{min(i, value):,}")
            time.sleep(0.01)  # Small delay for smooth animation
    else:
        placeholder.metric(label, f"{value:,}")

c1, c2, c3 = st.columns(3)

total_enrol = int(filtered_df['Total_Enrolments'].sum())
states_count = filtered_df['State'].nunique()
districts_count = filtered_df['District'].nunique()

animated_metric(c1, "🧾 Total Enrolments", total_enrol)
animated_metric(c2, "🏛️ States Covered", states_count)
animated_metric(c3, "📍 Districts Covered", districts_count)

st.divider()

# ================== TIME SERIES PREPARATION ==================
daily = filtered_df.groupby('Date')['Total_Enrolments'].sum()
monthly = daily.resample("MS").sum().asfreq("MS", fill_value=0)

forecast_horizon = 6

# ================== FORECASTING ENGINE ==================
forecast = None
lower = upper = None
rmse = None
method_used = None

if model_choice in ["Auto (Best)", "ARIMA (Time Series)"] and len(monthly) >= 12:
    try:
        model = ARIMA(monthly, order=(1, 1, 1))
        fit = model.fit()
        forecast = fit.forecast(forecast_horizon)

        sigma = fit.resid.std()
        lower = forecast - 1.96 * sigma
        upper = forecast + 1.96 * sigma

        rmse = np.sqrt(
            mean_squared_error(
                monthly[-forecast_horizon:],
                fit.fittedvalues[-forecast_horizon:]
            )
        )
        method_used = "ARIMA"

    except Exception:
        forecast = None

if forecast is None:
    if model_choice == "ARIMA (Time Series)":
        st.warning("⚠ ARIMA requires ≥ 12 months. Auto-switched to ML.")

    X = np.arange(len(monthly)).reshape(-1, 1)
    y = monthly.values

    lr = LinearRegression()
    lr.fit(X, y)

    future_X = np.arange(len(monthly), len(monthly) + forecast_horizon).reshape(-1, 1)
    preds = lr.predict(future_X)

    forecast = pd.Series(
        preds,
        index=pd.date_range(monthly.index[-1], periods=forecast_horizon + 1, freq="MS")[1:]
    )

    sigma = np.std(y - lr.predict(X))
    lower = forecast - 1.96 * sigma
    upper = forecast + 1.96 * sigma

    rmse = np.sqrt(
        mean_squared_error(y[-forecast_horizon:], lr.predict(X)[-forecast_horizon:])
    )

    method_used = "ML (Linear Regression)"

# ================== GROWTH METRICS & RECOMMENDATIONS ==================
growth = ((monthly.iloc[-1] - monthly.iloc[-2]) / monthly.iloc[-2] * 100) if len(monthly) > 1 else 0

if growth > 5:
    rec = "Rapid growth: Increase enrolment centres and staffing."
elif growth < -5:
    rec = "Decline detected: Launch awareness campaigns and mobile units."
else:
    rec = "Stable trend: Maintain current infrastructure."

# ================== RISK ALERTS ==================
st.subheader("🚨 Enrolment Risk Alerts")

if growth < -10:
    st.error("🔴 Sharp decline detected – immediate intervention required.")
elif growth < -5:
    st.warning("🟠 Moderate decline – awareness campaigns recommended.")
elif growth > 10:
    st.success("🟢 Surge detected – scale infrastructure and manpower.")
else:
    st.info("🟢 No abnormal risk detected.")

# ================== TREND VISUALIZATION ==================
st.subheader("📈 Past • Present • Future Enrolment Trend")

z_scores = (monthly - monthly.mean()) / monthly.std()
anomalies = monthly[abs(z_scores) > 2]

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(monthly.index, monthly.values, label="Past", linewidth=2)
ax.scatter(monthly.index[-1], monthly.values[-1], color="black", s=80, label="Present")
ax.plot(forecast.index, forecast.values, "--", color="red", label="Forecast")
ax.fill_between(forecast.index, lower, upper, color="red", alpha=0.2, label="Confidence Band")
ax.scatter(anomalies.index, anomalies.values, color="orange", s=60, label="Anomalies")

ax.set_title(f"Enrolment Trend ({method_used})")
ax.set_xlabel("Time")
ax.set_ylabel("Total Enrolments")
ax.legend()
ax.grid(alpha=0.3)

st.pyplot(fig, use_container_width=True)
st.success(f"📏 Model Accuracy (RMSE): {rmse:,.0f}")

# ================== STATE HEATMAP ==================
st.subheader("🗺️ State-wise Aadhaar Enrolment Heatmap")

state_map = filtered_df.groupby("State")["Total_Enrolments"].sum().reset_index()

fig_map = px.choropleth(
    state_map,
    geojson="https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/india_states.geojson",
    featureidkey="properties.ST_NM",
    locations="State",
    color="Total_Enrolments",
    color_continuous_scale="Reds"
)

fig_map.update_geos(fitbounds="locations", visible=False)
st.plotly_chart(fig_map, use_container_width=True)

# ================== DISTRICT CLUSTERING ==================
st.subheader("🧠 District-Level Clustering & Heatmap Drill-Down")

cluster_df = filtered_df.groupby("District")[['Age 0-5','Age 5-17','18+ Adults']].sum()

if len(cluster_df) >= 3:
    scaled = StandardScaler().fit_transform(cluster_df)
    cluster_df["Cluster"] = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(scaled)
    
    cluster_df["Total"] = cluster_df['Age 0-5'] + cluster_df['Age 5-17'] + cluster_df['18+ Adults']
    cluster_df = cluster_df.sort_values("Total", ascending=False)

    col1, col2 = st.columns([1.5, 1])
    
    # Left: Clustering scatter
    with col1:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        scatter = sns.scatterplot(
            x=cluster_df["18+ Adults"],
            y=cluster_df["Age 0-5"],
            hue=cluster_df["Cluster"],
            size=cluster_df["Total"],
            palette="Set2",
            ax=ax2,
            sizes=(100, 500),
            alpha=0.7
        )
        ax2.set_xlabel("18+ Adults", fontsize=11, fontweight='bold')
        ax2.set_ylabel("Age 0-5", fontsize=11, fontweight='bold')
        ax2.set_title("District Clustering (Size = Total Enrolments)", fontsize=12, fontweight='bold')
        ax2.legend(title="Cluster", bbox_to_anchor=(1.05, 1))
        st.pyplot(fig2, use_container_width=True)
    
    # Right: District selector
    with col2:
        st.markdown("**🔍 Select District for Details**")
        selected_district = st.selectbox(
            "Drill-down to district:",
            cluster_df.index.tolist(),
            label_visibility="collapsed"
        )
    
    # District Heatmap Drill-Down
    st.markdown("---")
    st.subheader(f"📊 District Heatmap: {selected_district}")
    
    district_detail = filtered_df[filtered_df['District'] == selected_district].copy()
    
    if not district_detail.empty:
        # Heatmap by State and Age Group
        district_pivot = district_detail.groupby("State")[['Age 0-5', 'Age 5-17', '18+ Adults']].sum()
        
        fig_heat, ax_heat = plt.subplots(figsize=(10, 4))
        sns.heatmap(
            district_pivot.T,
            annot=True,
            fmt='.0f',
            cmap='YlOrRd',
            cbar_kws={'label': 'Enrolments'},
            ax=ax_heat,
            linewidths=0.5
        )
        ax_heat.set_title(f"Age-wise Enrolments by State in {selected_district}", fontsize=12, fontweight='bold')
        ax_heat.set_xlabel("State", fontsize=10)
        ax_heat.set_ylabel("Age Group", fontsize=10)
        st.pyplot(fig_heat, use_container_width=True)
        
        # Time series for selected district
        st.markdown("**📈 Enrolment Trend in Selected District**")
        district_ts = district_detail.groupby('Date')[['Age 0-5', 'Age 5-17', '18+ Adults']].sum()
        
        fig_ts = px.area(
            district_ts.reset_index().melt(id_vars='Date', var_name='Age Group', value_name='Enrolments'),
            x='Date',
            y='Enrolments',
            color='Age Group',
            title=f"Timeline Trends - {selected_district}",
            labels={'Date': 'Date', 'Enrolments': 'Enrolments'},
            height=400
        )
        st.plotly_chart(fig_ts, use_container_width=True)
        
        # District metrics
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("🧒 Age 0-5", f"{int(district_pivot['Age 0-5'].sum()):,}")
        col_m2.metric("👶 Age 5-17", f"{int(district_pivot['Age 5-17'].sum()):,}")
        col_m3.metric("👨 18+ Adults", f"{int(district_pivot['18+ Adults'].sum()):,}")
        col_m4.metric("🎯 Total", f"{int(district_pivot.sum().sum()):,}")
    
    else:
        st.info("No data available for selected district")

# ================== STATE RANKING ==================
st.subheader("🏆 State Ranking Score")

state_score = (
    filtered_df.groupby("State")["Total_Enrolments"]
    .sum()
    .sort_values(ascending=False)
)

st.dataframe(state_score.reset_index().rename(columns={"Total_Enrolments":"Score"}))

# ================== EXECUTIVE SUMMARY ==================
st.subheader("🧾 Executive Summary")

st.markdown(f"""
**Period Analysed:** {monthly.index.min().strftime('%Y-%m')} → {monthly.index.max().strftime('%Y-%m')}

**Key Findings**
- Trend is **{"increasing" if growth > 0 else "declining"}**
- Forecast uncertainty quantified via confidence bands
- District behavioural clusters identified

**Recommended Actions**
- {rec}
- Monitor anomaly districts
- Plan infrastructure using forecasts
""")

# ================== DATA DOWNLOAD ==================
st.subheader("⬇️ Download Cleaned Data")

st.download_button(
    "Download Selected States CSV",
    filtered_df.to_csv(index=False),
    "aadhaar_selected_states_cleaned.csv",
    "text/csv"
)

st.markdown("---")
st.caption("Developed by UIADAI • Government-Grade Data Science System")
