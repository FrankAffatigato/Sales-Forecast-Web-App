import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet
from datetime import timedelta
import plotly.express as px

# ---------- CONFIG ----------
st.set_page_config(page_title="Retail Sales Forecast Dashboard", layout="wide")
st.markdown("""
    <style>
    .big-font {
        font-size:24px !important;
        font-weight:600;
        color:#003366;
        font-family: 'Segoe UI', sans-serif;
    }
    .metric-box {
        background-color: #e8edf3;
        padding: 10px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0px 1px 3px rgba(0,0,0,0.1);
        font-family: 'Segoe UI', sans-serif;
        color: #000000;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Retail Forecasting Dashboard")

# ---------- LOAD DATA ----------
@st.cache_data

def load_data():
    df = pd.read_csv("enhanced_sample_sales.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df

raw_df = load_data()

# ---------- FILTERS ----------
st.sidebar.header("🔍 Filter Options")
state = st.sidebar.selectbox("Select State", sorted(raw_df["state"].unique()))
stores = raw_df[raw_df["state"] == state]["store_id"].unique()
store = st.sidebar.selectbox("Select Store", sorted(stores))
products = raw_df[(raw_df["state"] == state) & (raw_df["store_id"] == store)]["product_id"].unique()
product = st.sidebar.selectbox("Select Product", sorted(products))

filtered_df = raw_df[
    (raw_df["state"] == state) &
    (raw_df["store_id"] == store) &
    (raw_df["product_id"] == product)
]

# ---------- KPIs ----------
latest = filtered_df.sort_values("date").iloc[-1]
avg_sales = filtered_df["sales"].mean()
avg_margin = filtered_df["margin"].mean()
avg_outstock = filtered_df["out_of_stock_pct"].mean()

col1, col2, col3, col4 = st.columns(4)
col1.markdown("<div class='metric-box'><p class='big-font'>🛒 Latest Sales</p><h3 style='color:#000'>{}</h3></div>".format(latest["sales"]), unsafe_allow_html=True)
col2.markdown("<div class='metric-box'><p class='big-font'>💰 Avg. Margin</p><h3 style='color:#000'>${:.2f}</h3></div>".format(avg_margin), unsafe_allow_html=True)
col3.markdown("<div class='metric-box'><p class='big-font'>📦 Avg. Out-of-Stock %</p><h3 style='color:#000'>{:.1f}%</h3></div>".format(avg_outstock), unsafe_allow_html=True)
col4.markdown("<div class='metric-box'><p class='big-font'>📉 Avg. Daily Sales</p><h3 style='color:#000'>{:.1f}</h3></div>".format(avg_sales), unsafe_allow_html=True)

# ---------- PROPHET FORECAST ----------
df_prophet = filtered_df[["date", "sales"]].rename(columns={"date": "ds", "sales": "y"})
model = Prophet()
model.fit(df_prophet)
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)
merged = pd.merge(future, forecast[["ds", "yhat"]], on="ds")
cutoff = df_prophet["ds"].max()

# ---------- VISUAL: Forecast Trend ----------
st.markdown("### 📈 Forecast vs Actual Sales")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_prophet["ds"], y=df_prophet["y"], mode='lines', name='Actual Sales'))
fig.add_trace(go.Scatter(x=merged["ds"], y=merged["yhat"], mode='lines+markers', name='Forecast'))
fig.add_vline(x=cutoff, line_width=2, line_dash='dash', line_color='red')
fig.add_annotation(x=cutoff, y=max(df_prophet['y']), text="Forecast Starts", showarrow=True, arrowhead=1)
fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Units Sold",
    legend_title="Legend",
    height=500,
    template="plotly_white",
    margin=dict(l=20, r=20, t=40, b=20)
)
st.plotly_chart(fig, use_container_width=True)

# ---------- VISUAL: Top 5 Products by Total Sales ----------
st.markdown("### 🏆 Top 5 Products by Total Sales")
top5_df = raw_df.groupby("product_id")["sales"].sum().nlargest(5).reset_index()
fig_top5 = px.bar(top5_df, x="product_id", y="sales", text_auto=True, color="product_id",
                  title="Top 5 Selling Products", labels={"product_id": "Product", "sales": "Total Sales"})
fig_top5.update_layout(template="plotly_white", height=400, showlegend=False)
st.plotly_chart(fig_top5, use_container_width=True)

# ---------- VISUAL: Store Sales Comparison ----------
st.markdown("### 🏬 Store Sales Comparison (Last 7 Days)")
latest_date = raw_df["date"].max()
store_agg = raw_df[raw_df["date"] >= latest_date - pd.Timedelta(days=7)].groupby("store_id")["sales"].sum().reset_index()
fig_store = px.bar(store_agg, x="store_id", y="sales", color="store_id",
                   labels={"store_id": "Store", "sales": "7-Day Sales"},
                   title="Total Sales by Store - Last 7 Days")
fig_store.update_layout(template="plotly_white", height=400, showlegend=False)
st.plotly_chart(fig_store, use_container_width=True)

# ---------- TABLE ----------
st.markdown("### 🧾 Forecast Table")
with st.expander("📋 View Forecast Data Table"):
    display_df = merged.copy()
    display_df = display_df[display_df['ds'] > cutoff].rename(columns={'ds': 'Date', 'yhat': 'Forecasted Sales'})
    st.dataframe(display_df.round(2), use_container_width=True)