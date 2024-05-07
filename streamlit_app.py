%%writefile app.py
import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import openpyxl
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import plotly.graph_objs as go
import warnings
warnings.filterwarnings('ignore')

@st.cache_data
def load_data():
    data = pd.read_csv("https://raw.githubusercontent.com/Pkun-og/Stream_weather/main/datasets/DailyDelhiClimateTrain_delhi_2017.csv")
    data["date"] = pd.to_datetime(data["date"], format='%Y-%m-%d')
    data['year'] = data['date'].dt.year
    data["month"] = data["date"].dt.month
    return data

@st.cache_data
def load_train():
    train=pd.read_excel("https://github.com/Pkun-og/Stream_weather/raw/main/datasets/AQI_ttnagar_Epics.xlsx", engine='openpyxl')
    train['Date'] = pd.to_datetime(train['Date'])
    train = train.drop(["AQI Status"], axis = 1)
    train = train.drop(["Benzene (µg/m3)"], axis = 1)
    train = train.dropna()
    return train

def page1(data):
    st.header("Historical Weather Plots (Delhi)")
    st.subheader("This Page contains the plots containing historical weather data from Delhi.")
    st.markdown("<hr>", unsafe_allow_html=True)

    # Mean temperature plot
    st.subheader("Mean Temperature Over the Years:")
    mean_temp_overyears(data)

    # Humidity plot
    st.subheader("Humidity Over the Years:")
    humidity_overyears(data)

    # Wind speed plot
    st.subheader("Wind Speed Over the Years:")
    wind_speed_overyears(data)

def page2():
    st.header("AQI Historical Plot(TT Nagar Bhopal)")
    st.subheader("This Page contains the historical Observations/plots of the AQI of TT Nagar, Bhopal.")
    st.markdown("<hr>", unsafe_allow_html=True)

    # AQI over time
    st.subheader("AQI Overtime")
    aqi_otm(train)

    # AQI over the years
    st.subheader("AQI Over the years")
    aqi_oty(train)

def page3():
    st.header("AQI Predictor")
    st.subheader("Users can use this page to get predictions of AQI for specific dates.")
    st.markdown("<hr>", unsafe_allow_html=True)

    predictions_aqi = aqi_prediction(train)
    aqi_pred_specific_date(predictions_aqi)

def page4():
  st.header("Weather Predictor (Delhi)")
  st.subheader("Users can use this page to get predicitons of various measures of weather namely, Temperature, Humidity and Wind Speed.")
  st.markdown("<hr>", unsafe_allow_html=True)

  predictions_meantemp = weather_mt_prediction(data)
  specific_date_prediction_mt(predictions_meantemp)

  predictions_windspeed = weather_ws_prediction(data)
  specific_date_prediction_ws(predictions_windspeed)

  prediction_humidity = weather_hd_prediction(data)
  specific_date_prediction_hd(prediction_humidity)


#Weather Hist plots*************************************************************
@st.cache_data
def mean_temp_overyears(data):
    plt.style.use('fivethirtyeight')
    fig, ax = plt.subplots(figsize=(15, 10))
    plt.title("Temperature Change in Delhi Over the Years")
    sns.set(rc={'text.color': 'white'})
    sns.lineplot(data=data, x='month', y='meantemp', hue='year')
    st.pyplot(fig)

@st.cache_data
def wind_speed_overyears(data):
    plt.style.use('fivethirtyeight')
    fig, ax = plt.subplots(figsize=(15, 10))
    plt.title("Humidity Change in Delhi Over the Years")
    sns.set(rc={'text.color': 'white'})
    sns.lineplot(data=data, x='month', y='humidity', hue='year')
    st.pyplot(fig)

@st.cache_data
def humidity_overyears(data):
    plt.style.use('fivethirtyeight')
    fig, ax = plt.subplots(figsize=(15, 10))
    plt.title("Wind speed Change in Delhi Over the Years")
    sns.set(rc={'text.color': 'white'})
    sns.lineplot(data=data, x='month', y='wind_speed', hue='year')
    st.pyplot(fig)

#Weather Prediction*************************************************************

#Meantemp***********************************
def weather_mt_prediction(data):
    forecast_data_meanTemp = data.rename(columns={"date": "ds", "meantemp": "y"})
    model = Prophet()
    model.fit(forecast_data_meanTemp)
    forecasts_meanTemp = model.make_future_dataframe(periods=365)
    predictions_meantemp = model.predict(forecasts_meanTemp)
    fig = plot_plotly(model, predictions_meantemp)
    fig.add_trace(go.Scatter(x=forecast_data_meanTemp['ds'], y=forecast_data_meanTemp['y'],
                             mode='lines', name='Observed'))
    fig.add_trace(go.Scatter(x=predictions_meantemp['ds'], y=predictions_meantemp['yhat'],
                             mode = 'lines', name = 'Prediction'))

    # Customize layout
    fig.update_layout(title='Mean Temperature Forecast',
                      xaxis_title='Date',
                      yaxis_title='Mean Temperature')
    st.plotly_chart(fig)
    return predictions_meantemp


def specific_date_prediction_mt(predictions_meantemp):
    specific_date_input = st.text_input("Enter the date (YYYY-MM-DD):")
    if specific_date_input:
        specific_date = pd.to_datetime(specific_date_input)
        specific_date_prediction = predictions_meantemp[predictions_meantemp['ds'] == specific_date]

        # Check if specific_date_prediction is not None before accessing its attributes
        if specific_date_prediction is not None and not specific_date_prediction.empty:
            st.write("Mean Temperature Prediction for {} is : {}".format(specific_date, specific_date_prediction['yhat'].values[0]))
        else:
            st.write("No prediction available for the specified date.")
#*******************************************

#Wind Speed
def weather_ws_prediction(data):
  forecast_data_windspeed = data.rename(columns = {"date": 'ds', "wind_speed": 'y'})
  model = Prophet()
  model.fit(forecast_data_windspeed)
  forecasts_windspeed = model.make_future_dataframe(periods=365)
  predictions_windspeed = model.predict(forecasts_windspeed)
  fig = plot_plotly(model, predictions_windspeed)
  fig.add_trace(go.Scatter(x=forecast_data_windspeed['ds'], y=forecast_data_windspeed['y'],
                             mode='lines', name='Observed'))
  fig.add_trace(go.Scatter(x=predictions_windspeed['ds'], y=predictions_windspeed['yhat'],
                             mode = 'lines', name = 'Prediction'))

    # Customize layout
  fig.update_layout(title='Wind Speed Forecast',
                      xaxis_title='Date',
                      yaxis_title='Wind Speed')
  st.plotly_chart(fig)
  return predictions_windspeed

def specific_date_prediction_ws(predictions_windspeed):
  specific_date_input = st.text_input("Enter the date (YYYY-MM-DD):", key="specific_date_input_ws")
  if specific_date_input:
        specific_date = pd.to_datetime(specific_date_input)
        specific_date_prediction = predictions_windspeed[predictions_windspeed['ds'] == specific_date]

        # Check if specific_date_prediction is not None before accessing its attributes
        if specific_date_prediction is not None and not specific_date_prediction.empty:
            st.write("Wind Speed Prediction for {} is : {}".format(specific_date, specific_date_prediction['yhat'].values[0]))
        else:
            st.write("No prediction available for the specified date.")

#Humidity***********************************************************************
def weather_hd_prediction(data):
  forecast_data_humidity = data.rename(columns = {"date": 'ds', "humidity": 'y'})
  model = Prophet()
  model.fit(forecast_data_humidity)
  forecasts_humidity = model.make_future_dataframe(periods=365)
  predictions_humidity = model.predict(forecasts_humidity)
  fig = plot_plotly(model, predictions_humidity)
  fig.add_trace(go.Scatter(x=forecast_data_humidity['ds'], y=forecast_data_humidity['y'],
                             mode='lines', name='Observed'))
  fig.add_trace(go.Scatter(x=predictions_humidity['ds'], y=predictions_humidity['yhat'],
                             mode = 'lines', name = 'Prediction'))

    # Customize layout
  fig.update_layout(title='Humidity Forecast',
                      xaxis_title='Date',
                      yaxis_title='Humidity')
  st.plotly_chart(fig)
  return predictions_humidity

def specific_date_prediction_hd(predictions_humidity):
  specific_date_input = st.text_input("Enter the date (YYYY-MM-DD):", key="specific_date_input_hd")
  if specific_date_input:
        specific_date = pd.to_datetime(specific_date_input)
        specific_date_prediction = predictions_humidity[predictions_humidity['ds'] == specific_date]

        # Check if specific_date_prediction is not None before accessing its attributes
        if specific_date_prediction is not None and not specific_date_prediction.empty:
            st.write("Humidity Prediction for {} is : {}".format(specific_date, specific_date_prediction['yhat'].values[0]))
        else:
            st.write("No prediction available for the specified date.")



#AQI Hist plots*****************************************************************

@st.cache_data
def aqi_otm(train):

    x = train["Date"].tolist()  # Convert Series to list
    y = train["AQI No."].tolist()
    figure = px.line(data, x, y, title='AQI in TT Nagar Over the Years')
    st.plotly_chart(figure)

@st.cache_data
def aqi_oty(train):
    train["Date"] = pd.to_datetime(train["Date"], format = '%Y-%m-%d')
    train['year'] = train['Date'].dt.year
    train["month"] = train["Date"].dt.month
    plt.style.use('fivethirtyeight')
    fig, ax = plt.subplots(figsize=(15, 10))
    plt.title("AQI Change in Bhopal(TT Nagar) Over the Years")
    sns.lineplot(data=train, x='month', y='AQI No.', hue='year')
    st.pyplot(fig)


#AQI Prediction*****************************************************************

@st.cache_data
def aqi_prediction(train):
    forecast_AQI = train.rename(columns={"Date": "ds", "AQI No.": "y"})
    model = Prophet()
    model.fit(forecast_AQI)
    forecasts_aqi = model.make_future_dataframe(periods=365)
    predictions_aqi = model.predict(forecasts_aqi)

    # Plotting observed data and predictions
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast_AQI['ds'], y=forecast_AQI['y'], mode='lines', name='Observed'))
    fig.add_trace(go.Scatter(x=predictions_aqi['ds'], y=predictions_aqi['yhat'], mode='lines', name='Predictions'))

    fig.update_layout(title='AQI Prediction',
                      xaxis_title='Date',
                      yaxis_title='AQI')

    # Show the Plotly figure using st.plotly_chart()
    st.plotly_chart(fig)

    return predictions_aqi


def aqi_pred_specific_date(predictions_aqi):
    # Obtain prediction for a specific date
    st.write("Enter the date (YYYY-MM-DD): ")
    specific_date_input = st.text_input("")

    if specific_date_input:
        specific_date = pd.to_datetime(specific_date_input)
        specific_date_prediction = predictions_aqi[predictions_aqi['ds'] == specific_date]

        # Print the prediction for the specific date
        if not specific_date_prediction.empty:
            st.write("AQI Prediction for {} is : {}".format(specific_date, specific_date_prediction['yhat'].values[0]))
        else:
            st.write("No prediction available for the specified date.")




# Load data*********************************************************************
data = load_data()
train = load_train()

# Add image to the sidebar
st.sidebar.image("/content/drive/MyDrive/images/aero purity.jpg", use_column_width=True)

# Sidebar navigation
page = st.sidebar.selectbox("Select Page", ["Historical Weather Data (Delhi)", "Historical AQI Data (TT Nagar Bhopal)", "AQI Predictor", "Weather Predictor(Currently only Delhi)"])

def add_footer():
    st.markdown("<hr>", unsafe_allow_html=True)
    st.write("Made by Priyansh")
# Render selected page
if page == "Historical Weather Data (Delhi)":
    page1(data)
    add_footer()
elif page == "Historical AQI Data (TT Nagar Bhopal)":
    page2()
    add_footer()
elif page == "AQI Predictor":
    page3()
    add_footer()
elif page == "Weather Predictor(Currently only Delhi)":
    page4()
    add_footer()
