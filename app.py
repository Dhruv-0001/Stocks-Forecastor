import streamlit as st
import datetime
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from datetime import date, timedelta
import pandas as pd

today = date.today()

TODAY = today.strftime("%Y-%m-%d")
d2 = date.today() - timedelta(days=3000)
START = d2.strftime("%Y-%m-%d")

st.markdown(f"<h1 style='text-align: center; color: red;'>STOCKS FORECASTOR</h1>", unsafe_allow_html=True)
st.image("https://www.pngall.com/wp-content/uploads/13/Stock-Market-PNG-File.png")

df= pd.read_csv("C:/PROGRAMMING/PYTHON LANGUAGE/projects/Stock Price Predictor/nasdaq_screener_1665400144860.csv")
stocks_list = df['Symbol']

stocks = (stocks_list)
st.markdown(f"<h3 style='text-align: center; color: black;'>SELECT THE STOCK</h3>", unsafe_allow_html=True)
st.text("")

selected_stock = st.selectbox('Select dataset for prediction', stocks)

Company_row=df.loc[df['Symbol']== selected_stock]
Company_name = Company_row.values.tolist()[0][1]

st.text("")
st.markdown(f"<h4 style='text-align: center; color: black;'>COMPANY NAME</h4>", unsafe_allow_html=True)
st.markdown(f"<h2 style='text-align: center; color: BlueViolet;'>{Company_name}</h2>", unsafe_allow_html=True)


n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365

@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

	
data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig,use_container_width=True)
	
plot_raw_data()

# Predict forecast with Prophet.
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

df_train['ds'] = df_train['ds'].dt.tz_convert(None)

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast

st.markdown(f"<h2 style='text-align: center; color: BlueViolet;'>FORECASTED DATA</h2>", unsafe_allow_html=True)
st.write(forecast.tail())
    
st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1,use_container_width=True)

st.markdown(f"<h2 style='text-align: center; color: BlueViolet;'>FORECAST COMPONENTS</h2>", unsafe_allow_html=True)
fig2 = m.plot_components(forecast)
st.write(fig2)
st.write("")
st.markdown("<h4 style='text-align: center; color: BlueVoilet;'>--DEVELOPED BY @DHRUV TYAGI</h4>", unsafe_allow_html=True)
