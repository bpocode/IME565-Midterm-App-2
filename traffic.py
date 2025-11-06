# Import necessary libraries
import streamlit as st
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

traffic_df = pd.read_csv('Traffic_Volume.csv')
traffic_df['date_time'] = pd.to_datetime(traffic_df['date_time'])
traffic_df['month'] = traffic_df['date_time'].dt.month_name()
traffic_df['weekday'] = traffic_df['date_time'].dt.day_name()     # or .dt.weekday for numeric (Mon=0)
traffic_df['hour'] = traffic_df['date_time'].dt.hour

mapie_pickle = open('mapie.pickle', 'rb') 
mapie = pickle.load(mapie_pickle) 

# Set up the title and description of the app
st.title('Traffic Volume Predictor') 
st.write("Utilize our advanced Machine Learning application to predict traffic volume.")

st.image('traffic_image.gif', width = 600)

a = st.slider('Select alpha value for prediction interval',min_value=0.01, max_value=0.50, step=0.01)

# Create a sidebar for input collection
st.sidebar.image('traffic_sidebar.jpg', width = 400)
st.sidebar.header('**Input Features**')
st.sidebar.write('You can either upload your data file or manually enter input features.')


with st.sidebar.expander('Option 1:Upload CSV File'):
    upload = st.file_uploader("Choose a file")

if upload is not None:
    upload_df = pd.read_csv(upload)
    both = pd.concat([traffic_df,upload_df],axis=0)
    both = both[['holiday','temp','rain_1h','snow_1h','clouds_all','weather_main','month','weekday','hour']]
    upload_dummies = pd.get_dummies(both)
    upload_dummies = upload_dummies.iloc[len(traffic_df):,:]
    pred, pred_int = mapie.predict(upload_dummies, alpha = a)
    upload_df["lower_bound"] = pred_int[:, 0]
    upload_df["predicted"] = pred
    upload_df["upper_bound"] = pred_int[:, 1]
    st.dataframe(upload_df)

with st.sidebar.expander('Option 2:Fill Out Form'):
    st.write('Enter the traffic details manually using the form below.')
    holiday = st.selectbox('Choose whether today is a designated holiday or not', options = traffic_df['holiday'].unique().tolist())
    temp = st.number_input('Average temperature in Kelvin', value=281.21,step=1.0)
    rain_1h = st.number_input('Amount in mm of rain that occurred in the hour', value=0.33,step=0.01)
    snow_1h = st.number_input('Amount in mm of snow that occurred in the hour', value=0.0,step=0.01)
    clouds_all = st.number_input('Percentage of cloud cover', value=49, min_value=0, max_value=100,step=1)
    weather_main = st.selectbox('Choose the current weather', options = traffic_df['weather_main'].unique().tolist())
    month = st.selectbox('Choose month', options = traffic_df['month'].unique().tolist())
    weekday = st.selectbox('Choose day of the week', options = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
    hour = st.number_input('Choose hour', value=13, min_value=0, max_value=23,step=1)

    #---------------- Utilized generative AI to create encodings below ----------------#
    (   holiday_Christmas_Day,
        holiday_Columbus_Day,
        holiday_Independence_Day,
        holiday_Labor_Day,
        holiday_Martin_Luther_King_Jr_Day,
        holiday_Memorial_Day,
        holiday_New_Years_Day,
        holiday_State_Fair,
        holiday_Thanksgiving_Day,
        holiday_Veterans_Day,
        holiday_Washingtons_Birthday) = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    
    if holiday == 'Christmas Day':
        holiday_Christmas_Day = 1
    elif holiday == 'Columbus Day':
        holiday_Columbus_Day = 1
    elif holiday == 'Independence Day':
        holiday_Independence_Day = 1
    elif holiday == 'Labor Day':
        holiday_Labor_Day = 1
    elif holiday == 'Martin Luther King Jr Day':
        holiday_Martin_Luther_King_Jr_Day = 1
    elif holiday == 'Memorial Day':
        holiday_Memorial_Day = 1
    elif holiday == 'New Years Day':
        holiday_New_Years_Day = 1
    elif holiday == 'State Fair':
        holiday_State_Fair = 1
    elif holiday == 'Thanksgiving Day':
        holiday_Thanksgiving_Day = 1
    elif holiday == 'Veterans Day':
        holiday_Veterans_Day = 1
    elif holiday == 'Washingtons Birthday':
        holiday_Washingtons_Birthday = 1

    (   weather_main_Clear,
        weather_main_Clouds,
        weather_main_Drizzle,
        weather_main_Fog,
        weather_main_Haze,
        weather_main_Mist,
        weather_main_Rain,
        weather_main_Smoke,
        weather_main_Snow,
        weather_main_Squall,
        weather_main_Thunderstorm) = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    if weather_main == 'Clear':
        weather_main_Clear = 1
    elif weather_main == 'Clouds':
        weather_main_Clouds = 1
    elif weather_main == 'Drizzle':
        weather_main_Drizzle = 1
    elif weather_main == 'Fog':
        weather_main_Fog = 1
    elif weather_main == 'Haze':
        weather_main_Haze = 1
    elif weather_main == 'Mist':
        weather_main_Mist = 1
    elif weather_main == 'Rain':
        weather_main_Rain = 1
    elif weather_main == 'Smoke':
        weather_main_Smoke = 1
    elif weather_main == 'Snow':
        weather_main_Snow = 1
    elif weather_main == 'Squall':
        weather_main_Squall = 1
    elif weather_main == 'Thunderstorm':
        weather_main_Thunderstorm = 1

    (
        month_January,
        month_February,
        month_March,
        month_April,
        month_May,
        month_June,
        month_July,
        month_August,
        month_September,
        month_October,
        month_November,
        month_December,
        weekday_Monday,
        weekday_Tuesday,
        weekday_Wednesday,
        weekday_Thursday,
        weekday_Friday,
        weekday_Saturday,
        weekday_Sunday,
    ) = [0] * 19

    # --- Encode current month ---
    if month == 'January':
        month_January = 1
    elif month == 'February':
        month_February = 1
    elif month == 'March':
        month_March = 1
    elif month == 'April':
        month_April = 1
    elif month == 'May':
        month_May = 1
    elif month == 'June':
        month_June = 1
    elif month == 'July':
        month_July = 1
    elif month == 'August':
        month_August = 1
    elif month == 'September':
        month_September = 1
    elif month == 'October':
        month_October = 1
    elif month == 'November':
        month_November = 1
    elif month == 'December':
        month_December = 1

    # --- Encode current weekday ---
    if weekday == 'Monday':
        weekday_Monday = 1
    elif weekday == 'Tuesday':
        weekday_Tuesday = 1
    elif weekday == 'Wednesday':
        weekday_Wednesday = 1
    elif weekday == 'Thursday':
        weekday_Thursday = 1
    elif weekday == 'Friday':
        weekday_Friday = 1
    elif weekday == 'Saturday':
        weekday_Saturday = 1
    elif weekday == 'Sunday':
        weekday_Sunday = 1

    pred, pred_int = mapie.predict([[temp, rain_1h, snow_1h, clouds_all, hour,
    holiday_Christmas_Day, holiday_Columbus_Day, holiday_Independence_Day, holiday_Labor_Day,
    holiday_Martin_Luther_King_Jr_Day, holiday_Memorial_Day, holiday_New_Years_Day, holiday_State_Fair,
    holiday_Thanksgiving_Day, holiday_Veterans_Day, holiday_Washingtons_Birthday, weather_main_Clear,
    weather_main_Clouds, weather_main_Drizzle, weather_main_Fog, weather_main_Haze, weather_main_Mist,
    weather_main_Rain, weather_main_Smoke, weather_main_Snow, weather_main_Squall, weather_main_Thunderstorm,
    month_April, month_August, month_December, month_February, month_January, month_July, month_June,
    month_March, month_May, month_November, month_October, month_September, weekday_Friday, weekday_Monday,
    weekday_Saturday, weekday_Sunday, weekday_Thursday, weekday_Tuesday, weekday_Wednesday]], alpha = a)

    prediction = pred[0]
    lower_bound = round(float(pred_int[0, 0]),2)
    upper_bound = round(float(pred_int[0, 1]),2)

    submit_button = st.button("Submit Form Data")

if submit_button or (upload is not None):
    confidence = round((1 - a)*100,2)
    st.subheader("Predicting Traffic Volume...")
    st.write('Predicted Traffic Volume:',prediction)
    st.write('Prediction Interval('+ str(confidence)+'%):','['+str(lower_bound)+','+str(upper_bound)+']')

    st.title('Model Performance and Inference')
    tab1, tab2, tab3, tab4 = st.tabs(["Feature Importance", "Histogram of Residuals", "Predicted Vs. Actual", "Coverage Plot"])

    # Tab 1: Feature Importance
    with tab1:
        st.image('feat_imp.png')

    # Tab 2: Histogram of Residuals
    with tab2:
        st.image('residuals.png')

    # Tab 3: Predicted Vs. Actual
    with tab3:
        st.image('pred_vs_act.png')

    # Tab 4: Coverage Plot
    with tab4:
        st.image('coverage.png')
