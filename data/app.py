# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pickle
import numpy as np

# ----------------------------------
# Load Dataset
# ----------------------------------
load_df = pd.read_csv('load_raw.csv')

# Ensure year, month, day are numeric
load_df['year'] = pd.to_numeric(load_df['year'], errors='coerce')
load_df['month'] = pd.to_numeric(load_df['month'], errors='coerce')
load_df['day'] = pd.to_numeric(load_df['day'], errors='coerce')

# Create proper datetime column
load_df['Date'] = pd.to_datetime(
    dict(year=load_df['year'], month=load_df['month'], day=load_df['day']),
    errors='coerce'
)

# Drop invalid dates and normalize
load_df = load_df.dropna(subset=['Date'])
load_df['Date'] = load_df['Date'].dt.normalize()
load_df = load_df.sort_values('Date')

# Hour columns
hour_cols = [f'h{i}' for i in range(1, 25)]
load_df[hour_cols] = load_df[hour_cols].apply(pd.to_numeric, errors='coerce')

# Daily total load
load_df['Daily_Load'] = load_df[hour_cols].sum(axis=1)

# Aggregate daily load for plotting
daily_df = load_df.groupby('Date', as_index=False)['Daily_Load'].sum()
daily_df = daily_df.sort_values('Date')

# ----------------------------------
# Load Trained Model
# ----------------------------------
with open('rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

# ----------------------------------
# App Title
# ----------------------------------
st.title("⚡ Electricity Load Forecasting Web App")

# ----------------------------------
# Sidebar: Historical Date Range Selection
# ----------------------------------
st.sidebar.header("Select Historical Date Range")
start_date = st.sidebar.date_input(
    "Start Date",
    daily_df['Date'].min().date(),
    key='start_date_picker'
)
end_date = st.sidebar.date_input(
    "End Date",
    daily_df['Date'].max().date(),
    key='end_date_picker'
)

start = pd.to_datetime(start_date)
end = pd.to_datetime(end_date)

if start > end:
    st.error("Start date must be before end date")
    st.stop()

# ----------------------------------
# Sidebar: Hour Selection
# ----------------------------------
st.sidebar.header("Select Hours to Plot")
selected_hours = st.sidebar.multiselect(
    "Select hours",
    hour_cols,
    default=hour_cols,
    key='hour_select_picker'
)

# ----------------------------------
# Historical Data filtered by selected hours
# ----------------------------------
if selected_hours:
    filtered_hourly_sum = load_df[(load_df['Date'] >= start) & (load_df['Date'] <= end)][['Date'] + selected_hours].copy()
    filtered_hourly_sum['Daily_Load'] = filtered_hourly_sum[selected_hours].sum(axis=1)
else:
    filtered_hourly_sum = load_df[(load_df['Date'] >= start) & (load_df['Date'] <= end)][['Date']].copy()
    filtered_hourly_sum['Daily_Load'] = 0

# ----------------------------------
# Graph 1: Historical Daily Load
# ----------------------------------
st.subheader("Historical Daily Load (Sum of Selected Hours)")
if not filtered_hourly_sum.empty:
    fig_hist, ax_hist = plt.subplots(figsize=(12,5))
    ax_hist.plot(filtered_hourly_sum['Date'], filtered_hourly_sum['Daily_Load'], label='Historical', color='blue')
    ax_hist.set_xlabel("Date")
    ax_hist.set_ylabel("Load (MW)")
    ax_hist.set_title("Historical Electricity Load")
    ax_hist.legend()
    # Dynamic x-axis
    locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
    ax_hist.xaxis.set_major_locator(locator)
    ax_hist.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig_hist.autofmt_xdate()
    st.pyplot(fig_hist)
else:
    st.warning("No historical data available for selected date range or selected hours.")

# ----------------------------------
# Forecast Settings
# ----------------------------------
st.sidebar.header("Forecast Settings")
days_to_forecast = st.sidebar.slider(
    "Number of Days to Forecast",
    1, 365, 30,
    key='forecast_days_slider'
)
st.subheader(f"{days_to_forecast}-Day Load Forecast")

# Forecast beyond last historical date
last_row = load_df.iloc[-1]
forecast_dates = pd.date_range(start=last_row['Date'] + pd.Timedelta(days=1), periods=days_to_forecast)

forecast_values = []
load_df_copy = load_df.copy()

for date in forecast_dates:
    features = pd.DataFrame({
        'DayOfWeek':[date.weekday()],
        'Month':[date.month],
        'Is_Weekend':[date.weekday() >=5],
        'Lag_1':[load_df_copy['Daily_Load'].iloc[-1]],
        'Lag_7':[load_df_copy['Daily_Load'].iloc[-7]],
        'Lag_14':[load_df_copy['Daily_Load'].iloc[-14]],
        'Rolling_7':[load_df_copy['Daily_Load'].rolling(7).mean().iloc[-1]],
        'Rolling_14':[load_df_copy['Daily_Load'].rolling(14).mean().iloc[-1]]
    })
    pred = rf_model.predict(features)[0]
    forecast_values.append(pred)
    new_row = pd.DataFrame({'Date':[date],'Daily_Load':[pred]})
    load_df_copy = pd.concat([load_df_copy,new_row],ignore_index=True)

forecast_df = pd.DataFrame({'Date':forecast_dates,'Predicted Load':forecast_values})

# ----------------------------------
# Graph 2: Forecasted Daily Load
# ----------------------------------
st.subheader("Forecasted Daily Load")
fig_forecast, ax_forecast = plt.subplots(figsize=(12,5))
ax_forecast.plot(
    forecast_df['Date'], 
    forecast_df['Predicted Load'], 
    label='Forecast', 
    color='red', 
    linestyle='--'
)
ax_forecast.set_xlabel("Date")
ax_forecast.set_ylabel("Load (MW)")
ax_forecast.set_title("Forecasted Electricity Load (Beyond Historical Data)")
ax_forecast.legend()
# Force x-axis to show only forecast dates
ax_forecast.set_xlim(forecast_df['Date'].min(), forecast_df['Date'].max())
locator_fore = mdates.AutoDateLocator(minticks=5, maxticks=10)
ax_forecast.xaxis.set_major_locator(locator_fore)
ax_forecast.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
fig_forecast.autofmt_xdate()
st.pyplot(fig_forecast)

# ----------------------------------
# Download Forecast
# ----------------------------------
st.subheader("Download Forecast")
st.download_button(
    label="Download Forecast as CSV",
    data=forecast_df.to_csv(index=False),
    file_name="forecast.csv",
    mime="text/csv"
)

# ----------------------------------
# Feature Importance
# ----------------------------------
st.subheader("Feature Importance (Random Forest)")
feature_importance = pd.Series(
    rf_model.feature_importances_,
    index=['DayOfWeek','Month','Is_Weekend','Lag_1','Lag_7','Lag_14','Rolling_7','Rolling_14']
)
fig_fi, ax_fi = plt.subplots(figsize=(10,4))
feature_importance.sort_values().plot(kind='barh',ax=ax_fi)
ax_fi.set_xlabel("Importance Score")
st.pyplot(fig_fi)
