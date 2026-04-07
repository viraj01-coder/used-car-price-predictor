import os
import requests
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Used Car Price Predictor", page_icon="🚗", layout="wide")

CURRENCY_RATES = {
    "GBP (£)": 1.0,
    "USD ($)": 1.27,
    "EUR (€)": 1.17,
    "INR (₹)": 105.0,
}
CURRENCY_SYMBOLS = {
    "GBP (£)": "£",
    "USD ($)": "$",
    "EUR (€)": "€",
    "INR (₹)": "₹",
}

@st.cache_resource
def load_model():
    csv_file = "used_car_price_analysis.csv"
    if not os.path.exists(csv_file):
        url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMSkillsNetwork-AI0271EN-SkillsNetwork/labs/v1/m3/data/used_car_price_analysis.csv"
        r = requests.get(url)
        with open(csv_file, "wb") as f:
            f.write(r.content)
    df = pd.read_csv(csv_file)
    df = df.dropna(subset=['year', 'mileage', 'tax', 'mpg', 'engineSize', 'price'])
    X = df[['year', 'mileage', 'tax', 'mpg', 'engineSize']]
    y = df['price']
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2)),
        ('ridge', Ridge(alpha=100))
    ])
    model.fit(X, y)
    return model, df

model, df = load_model()

st.title("🚗 Used Car Price Predictor")
st.markdown("IBM Data Science Capstone Project | **Virajbhai Mavani**")
st.divider()

# Currency selector
col1, col2 = st.columns([1, 3])
with col1:
    selected_currency = st.selectbox("💱 Currency", list(CURRENCY_RATES.keys()))
rate = CURRENCY_RATES[selected_currency]
symbol = CURRENCY_SYMBOLS[selected_currency]

st.divider()
st.subheader("Enter Car Details")

col1, col2 = st.columns(2)
with col1:
    year = st.slider("Year of Manufacture", min_value=int(df['year'].min()), max_value=int(df['year'].max()), value=2018)
    mileage = st.number_input("Mileage (km)", min_value=0, max_value=300000, value=30000, step=1000)
    tax = st.number_input("Tax (£)", min_value=0, max_value=600, value=150)
with col2:
    mpg = st.number_input("Miles Per Gallon (MPG)", min_value=10.0, max_value=200.0, value=50.0, step=0.5)
    engine_size = st.selectbox("Engine Size (L)", sorted(df['engineSize'].unique()))

st.divider()
if st.button("🔍 Predict Price", use_container_width=True):
    input_data = pd.DataFrame({'year': [year], 'mileage': [mileage], 'tax': [tax], 'mpg': [mpg], 'engineSize': [engine_size]})
    predicted_price_gbp = max(0, model.predict(input_data)[0])
    predicted_price = predicted_price_gbp * rate
    st.success(f"### 💰 Estimated Price: {symbol}{predicted_price:,.0f}")
    st.info(f"""**Analysis:**
- A {year} car with {mileage:,} km mileage
- Engine: {engine_size}L | Fuel efficiency: {mpg} MPG
- Model confidence (R²): **75.6%**
- Currency: {selected_currency}""")

st.divider()
st.markdown("### 📊 Key Findings from Analysis")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Dataset Size", f"{len(df):,} cars")
with col2:
    avg_converted = df['price'].mean() * rate
    st.metric("Avg Price", f"{symbol}{avg_converted:,.0f}")
with col3:
    st.metric("Model R²", "75.6%")

st.divider()
st.markdown("### 📈 Data Insights")

col1, col2 = st.columns(2)

# Chart 1: Year wise Average Price Trend (Line Chart)
with col1:
    year_avg = df.groupby('year')['price'].mean().reset_index()
    year_avg['price_converted'] = year_avg['price'] * rate
    fig1 = px.line(year_avg, x='year', y='price_converted',
                   title='📈 Year wise Average Price Trend',
                   labels={'year': 'Year', 'price_converted': f'Avg Price ({symbol})'},
                   markers=True,
                   color_discrete_sequence=['#636EFA'])
    fig1.update_layout(template='plotly_dark')
    st.plotly_chart(fig1, use_container_width=True)

# Chart 2: Correlation Heatmap
with col2:
    corr_cols = ['year', 'mileage', 'tax', 'mpg', 'engineSize', 'price']
    corr_matrix = df[corr_cols].corr().round(2)
    fig2 = px.imshow(corr_matrix,
                     title='🔥 Correlation Heatmap',
                     color_continuous_scale='RdBu_r',
                     zmin=-1, zmax=1,
                     text_auto=True)
    fig2.update_layout(template='plotly_dark')
    st.plotly_chart(fig2, use_container_width=True)

col3, col4 = st.columns(2)

# Chart 3: Transmission Type vs Average Price
with col3:
    if 'transmission' in df.columns:
        trans_avg = df.groupby('transmission')['price'].mean().reset_index()
        trans_avg['price_converted'] = trans_avg['price'] * rate
        fig3 = px.bar(trans_avg, x='transmission', y='price_converted',
                      title='🚗 Transmission Type vs Avg Price',
                      labels={'transmission': 'Transmission', 'price_converted': f'Avg Price ({symbol})'},
                      color='price_converted',
                      color_continuous_scale='Blues')
        fig3.update_layout(template='plotly_dark')
        st.plotly_chart(fig3, use_container_width=True)

# Chart 4: Fuel Type Distribution (Pie Chart)
with col4:
    if 'fuelType' in df.columns:
        fuel_counts = df['fuelType'].value_counts().reset_index()
        fuel_counts.columns = ['fuelType', 'count']
        fig4 = px.pie(fuel_counts, values='count', names='fuelType',
                      title='🥧 Fuel Type Distribution',
                      color_discrete_sequence=px.colors.qualitative.Set2)
        fig4.update_layout(template='plotly_dark')
        st.plotly_chart(fig4, use_container_width=True)