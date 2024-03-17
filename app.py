# app.py
import streamlit as st
from Data_Exploration import load_data
from prediction_page import prediction_page
from Data_Exploration import load_data, data_exploration_page

# Set page config
st.set_page_config(page_title='Energy Consumption Forecast', page_icon=':zap:', layout='wide', initial_sidebar_state='expanded')
# Main app
def main():
    df = load_data()

    # Navigation
    page = st.sidebar.radio("Choose a page", ["Data Exploration", "Predict Future"])

    if page == "Data Exploration":
        data_exploration_page(df)
    elif page == "Predict Future":
        prediction_page(df)

if __name__ == "__main__":
    main()