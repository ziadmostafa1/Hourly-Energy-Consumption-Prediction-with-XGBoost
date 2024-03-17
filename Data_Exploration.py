# Data_Exploration.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import plotly.express as px
import plotly.io as pio
pio.templates.default = 'plotly_dark'



def load_data():
    df = pd.read_csv('PJME_hourly.csv')
    df.set_index('Datetime', inplace=True)
    df.index = pd.to_datetime(df.index)
    return df

df = load_data()

# Feature Creation
def create_features(df): 
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df

target_map = df['PJME_MW'].to_dict()

def add_lags(df):
    df['lag1'] = (df.index - pd.Timedelta('364 days')).map(target_map)
    df['lag2'] = (df.index - pd.Timedelta('728 days')).map(target_map)
    df['lag3'] = (df.index - pd.Timedelta('1092 days')).map(target_map)

    return df

def data_exploration_page(df):
    st.title('Data Exploration and Feature Creation')

    # Plotting
    fig = px.scatter(df, x=df.index, y='PJME_MW', title='PJM Energy use in MegaWatts')
    st.plotly_chart(fig, use_container_width=True)

    # Filter the dataframe to include only data from '2017-01-01' to '2017-12-31'
    df_2017 = df.loc['2017-01-01':'2017-12-31']

    # Create the line plot
    fig = px.line(df_2017, x=df_2017.index, y='PJME_MW', title='PJM Energy use in MegaWatts (01-01-2017 to 12-31-2017)')
    fig.update_xaxes(
        tickvals=pd.date_range('2017-01-01','2018-01-01', freq='M'),
        tickformat="%b-%Y",
        title_text='Month'
    )
    st.plotly_chart(fig, use_container_width=True)

    fig = px.histogram(df, x='PJME_MW', nbins=500, title='PJME_MW Energy Consumption')
    st.plotly_chart(fig, use_container_width=True)

    # remove outliers
    df_filtered = df.query('PJME_MW < 19_000')['PJME_MW'].reset_index()
    fig = px.scatter(df_filtered, x=df_filtered.index, y='PJME_MW', title='outliers')
    st.plotly_chart(fig, use_container_width=True)
    st.write('The outliers are removed')
    st.write('')

    df = df.query('PJME_MW > 19_000')

    # show fig.png
    st.subheader('Cross Validation Folds:')
    st.image('folds.png')

    # show df.head()
    st.subheader('Dataframe:')
    st.write(df.head())

    # Feature Creation
    df = create_features(df)

    # show df.head()
    st.subheader('Dataframe with Features:')
    st.write(df.head())

    # Create Lag Features
    df = add_lags(df)

    # show df.head()
    st.subheader('Dataframe Tail with Lag Features:')
    st.write(df.tail())

    # PJME MegaWatts by Hour of Day
    fig = px.box(df, x='hour', y='PJME_MW', title='PJME MegaWatts by Hour of Day')
    st.plotly_chart(fig, use_container_width=True)

    # PJME MegaWatts by Day of Week
    fig = px.box(df, x='month', y='PJME_MW', color='month', title='PJME MegaWatts by Month of Year')
    st.plotly_chart(fig, use_container_width=True)

    return df