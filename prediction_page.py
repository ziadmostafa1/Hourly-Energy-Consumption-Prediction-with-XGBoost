# prediction_page.py
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from pandas.tseries.offsets import DateOffset
import joblib
import plotly.express as px
from Data_Exploration import create_features, add_lags
import plotly.io as pio
pio.templates.default = 'plotly_dark'

# Load model
model = joblib.load('model.pkl')

def prediction_page(df):
    st.title('Predict Future Energy Consumption')

    # User input for prediction
    start_date = st.date_input('Select start date for prediction:', value=pd.to_datetime(df.index.max()))
    end_date = st.date_input('Select end date for prediction:',value = pd.to_datetime(df.index.max()) + DateOffset(years=3))

    # Create future dates
    future_dates = pd.date_range(start=start_date, end=end_date, freq='1h')
    future_df = pd.DataFrame(index=future_dates)
    future_df['isFuture'] = True
    df['isFuture'] = False
    df_and_future = pd.concat([df, future_df])

    # Create features and add lags
    df_and_future = create_features(df_and_future)
    df_and_future = add_lags(df_and_future)

    features = ['hour', 'dayofweek', 'quarter', 'month', 'year',
            'dayofyear', 'dayofmonth', 'weekofyear', 'lag1', 'lag2', 'lag3']

    # Predict the future
    future_with_features = df_and_future.query('isFuture').copy()
    future_with_features['pred'] = model.predict(future_with_features[features])

    # Plot the prediction
    fig = px.line(future_with_features, x=future_with_features.index, y='pred', title='Predicted PJME_MW Energy Consumption')
    st.plotly_chart(fig, use_container_width=True)

    # read feature_importances.csv and plot it
    feature_importances = pd.read_csv('feature_importance.csv', index_col=0)
    fig = px.bar(feature_importances.sort_values(by='Importance', ascending=True), 
                x='Importance', 
                y=feature_importances.index, 
                orientation='h', 
                title='Feature Importances')
    st.plotly_chart(fig, use_container_width=True)