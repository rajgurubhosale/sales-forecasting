import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import datetime

st.title('Sales Forecasting')

uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.write("Dataset Preview:", df.head())

    # Get user input for store_no and family
    store_no = st.selectbox('Select Store Number', sorted(df['store_nbr'].unique()))


    family = st.selectbox('Select Family',['All']+sorted(df['family'].unique()))

    # Filter data based on user input
    if family == 'All':
        filt = (df['store_nbr'] == store_no)
        filtered_data = df.loc[filt].groupby('date')['sales'].sum().reset_index()
    else:
        filtered_data = df[(df['store_nbr'] == store_no) & (df['family'] == family)]

    # Button to trigger forecast
    # Button to trigger forecast
    if st.button('Generate Forecast'):
        # Check if filtered data has sufficient rows for forecasting
        if len(filtered_data) >= 2:
            # Prepare the data for Prophet
            filtered_data = filtered_data[['date', 'sales']]
            filtered_data['date'] = pd.to_datetime(filtered_data['date'])
            filtered_data.rename(columns={'date': 'ds', 'sales': 'y'}, inplace=True)

            # Initialize and fit Prophet model
            model = Prophet(seasonality_prior_scale=5)
            model.fit(filtered_data)

            # Make a future dataframe for predictions
            future = model.make_future_dataframe(freq='D', periods=365)  # Forecast for the next 365 days
            forecast = model.predict(future)

            # Plot the historical data
            fig, ax = plt.subplots(figsize=(10, 6))

            # Plot the historical data (in blue)
            ax.plot(filtered_data['ds'], filtered_data['y'], label='Historical Data', color='blue')

            # Plot the forecasted data (in red), only from the point where forecasting starts
            forecast_start_date = filtered_data['ds'].max()  # The last date in the historical data
            forecasted_data = forecast[forecast['ds'] > forecast_start_date]

            ax.plot(forecasted_data['ds'], forecasted_data['yhat'], label='Forecasted Data', color='red')

            # Plot the confidence interval (shaded area)
            ax.fill_between(forecasted_data['ds'], forecasted_data['yhat_lower'], forecasted_data['yhat_upper'],
                            color='red', alpha=0.2)

            # Set labels and title
            ax.set_xlabel('Date')
            ax.set_ylabel('Sales')
            ax.set_title(f'Sales Forecast for Store {store_no} and Family {family}')
            ax.legend()

            st.write("Forecasted Sales for the 365  days (Graph):")
            st.pyplot(fig)

            # Filter forecast to only show the future predictions
            future_forecast = forecast[forecast['ds'] > filtered_data['ds'].max()]

            # Show the forecasted sales data (only future predictions)
            forecasted_sales = future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            forecasted_sales.rename(columns={'ds': 'Date', 'yhat': 'Predicted Sales', 'yhat_lower': 'Lower Bound',
                                             'yhat_upper': 'Upper Bound'}, inplace=True)

            total_sales = forecasted_sales['Predicted Sales'].sum()
            st.write(f'Total sales for next days: {total_sales}')

            st.write("Forecasted Sales Data (Table for Future Predictions):")
            st.dataframe(forecasted_sales)
            fig2 = model.plot_components(forecast)
            st.write("Prophet Model Components:")
            st.pyplot(fig2)


        else:
            st.warning("Not enough data for forecasting. Please select a different store or family.")
