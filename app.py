from ast import Str
# from turtle import onclick
import joblib
# from sqlalchemy import true
import streamlit as st
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px
import os
import pickle as pkl
import bcrypt
from joblib import load
from PIL import Image
from joblib import dump, load
from datetime import datetime, timedelta
import numpy as np


import random

from torch import threshold


st.set_page_config(layout="wide")


def sign_true():
    st.session_state.sign = True

def sign_false():
    st.session_state.sign = False

def logout():
    st.session_state.clear()

def create_new_user():
    if os.path.exists('accounts.pkl'):
        accounts = pd.read_pickle('accounts.pkl')
    else:
        accounts = pd.DataFrame(columns=['name', 'hash'])
    st.session_state.no_sname = False
    st.session_state.no_password = False
    st.session_state.no_match = False
    st.session_state.name_taken = False
    st.write(st.session_state.sname in accounts.name.values)
    if st.session_state.sname == '':
        st.session_state.no_sname = True
    if st.session_state.password == '':
        st.session_state.no_password == True
    elif st.session_state.password != st.session_state.password_conf:
        st.session_state.no_match = True
    elif st.session_state.sname in accounts.name.values:
        st.session_state.name_taken = True
    else:
        newdata = pd.DataFrame([[st.session_state.sname, bcrypt.hashpw(st.session_state.password.encode(), bcrypt.gensalt())]], columns=['name','hash'])
        accounts = pd.concat([accounts, newdata], ignore_index=True)
        accounts.to_pickle('accounts.pkl')
        st.session_state.login_name = st.session_state.sname
        st.session_state.sign = False


def validation():
    st.session_state.no_name = False
    st.session_state.incorrect = False
    if os.path.exists('accounts.pkl'):
        accounts = pd.read_pickle('accounts.pkl')
    else:
        accounts = pd.DataFrame(columns=['name', 'hash'])
    if st.session_state.name == "":
        st.session_state.no_name = True
    if st.session_state.name in accounts.name.values:
        if bcrypt.checkpw(st.session_state.password.encode(), accounts[accounts['name']==st.session_state.name]['hash'].values[0]) == True:
            st.session_state.login_name = st.session_state.name
        else:
            st.session_state.incorrect = True
    else:
        st.session_state.incorrect = True


def login():
    col1,col2,col3 = st.columns(3)
    col2.write("## Smart Healthcare Lab")
    col2.write("__Login to your account__")
    form = col2.form(key='loginForm')
    name = form.text_input('Name', key='name')
    password = form.text_input('Password', type='password', key='password')
    form.form_submit_button('Login', on_click = validation)
    if 'no_name' in st.session_state:
        if st.session_state.no_name == True:
            col2.error("Please fill in your username")
    if 'incorrect' in st.session_state:
        if st.session_state.incorrect == True:
            col2.error("Incorrect username and/or password")
    col2.write("No account yet?")
    col2.button('Sign Up', on_click = sign_true)
        
        

def signup():
    col1,col2,col3 = st.columns(3)
    col2.write("## Smart Healthcare Lab")
    col2.write("__Login to your account__")
    if os.path.exists('accounts.pkl'):
        accounts = pd.read_pickle('accounts.pkl')
    else:
        accounts = pd.DataFrame(columns=['name', 'hash'])
        
    formSU = col2.form(key='SignUpForm')
    name = formSU.text_input('Name', key='sname')
    password = formSU.text_input('Password', type='password', key='password')
    password_conf = formSU.text_input('Confirm password', type='password', key='password_conf')
    formSU.form_submit_button('Sign Up', on_click=create_new_user)
    if 'no_sname' in st.session_state:
        if st.session_state.no_sname == True:
            col2.error("Please fill in a username")
    if 'no_password' in st.session_state:
        if st.session_state.no_password == True:
            col2.error("Please fill in a password")
    if 'no_match' in st.session_state:
        if st.session_state.no_match == True:
            col2.error("Passwords do not match")
    if 'name_taken' in st.session_state:
        if st.session_state.name_taken == True:
            col2.error("Username is already taken")
    col2.button('Back', on_click = sign_false)



def predict_vulnerability(df, model_path, thresh, below=False):
    model = load(model_path)
    predictions = model.predict(df[['TotalSteps', 'VeryActiveMinutes', 'SedentaryMinutes', 'TotalMinutesAsleep', 'TotalMinutesAwake']])
    if sum(predictions[:10]) >= thresh:
        return True
    else:
        return False

def create_gauge(df, col, title):
    # create gauge chart
    gauge_fig = go.Figure(go.Indicator(
    domain = {'x': [0, 1], 'y': [0, 1]},
    value = df[col][:10].mean(),
    mode = "gauge+number+delta",
    title = {'text': title},
    delta = {'reference': df[col][:50].mean()},
    gauge = {'axis': {'range': [None, df[col].max()]},
             'steps' : [
                 {'range': [0, df[col].mean()-df[col].std()], 'color': "lightgray"},
                 {'range': [df[col].mean()-df[col].std(), df[col].mean()+df.TotalSteps.std()], 'color': "gray"}],
             'threshold' : {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': df[col].mean()},
             'bar': {'color': "#E42621"},}))
    return gauge_fig.update_layout(width=400, height=400)

def create_line(df,col, title, y_title):
    fig = px.line(df, x="ActivityDate", y=col, width=500, height=400, labels={"ActivityDate": 'Activity date', col: y_title}, title=title)
    fig['data'][0]['line']['color']='#FFFFFF'
    fig.update_layout(
        xaxis=dict(
            showgrid=False
        ),
        yaxis=dict(
            showgrid=False
        )
    )
    return fig

def create_bar(df,col, thresh, patient, title, y_title):
    fig = px.bar(df, x="ActivityDate", y=col, width=500, height=400, labels={"ActivityDate": 'Activity date', col: y_title}, title=title)
    fig['data'][0]['marker']['color']='#FFFFFF'
    # fig.update_traces(marker_color='#FFFFFF')#, marker_line_color='rgb(8,48,107)',marker_line_width=1.5, opacity=0.6)
    fig.add_hline(y=thresh, line={'color':'#E42621'})
    fig.update_layout(
        xaxis=dict(
            showgrid=False
        ),
        yaxis=dict(
            showgrid=False
        )
    )
    return fig

def advise(col, thresh_met, thresh, below=False):
    vinkje = Image.open('images/vinkje.jpg')
    uitroepteken = Image.open('images/uitroepteken.jpg')
    if thresh_met < thresh and below == False or thresh_met > thresh and below == True:
        col.image(uitroepteken, caption=None, width=100, use_column_width=True, clamp=False, channels="RGB", output_format="auto")
    else:
        col.image(vinkje, caption=None, width=100, use_column_width=True, clamp=False, channels="RGB", output_format="auto")

def forecast_data(df, model_path, id):
    loaded_model = load(model_path)

    #data preprocessing for time series and feature engineering steps
    df['Date'] = df['ActivityDate']
    df = df.sort_values(by="Date")
    df['Day'] = df['Date'].dt.dayofyear

    data_totalsteps = df[['Id','Date','Day','TotalSteps']]

    melt2 = data_totalsteps.copy()
    melt2['Yesterday'] = melt2.groupby(['Id'])['TotalSteps'].shift()
    melt2['Yesterday_Diff'] = melt2.groupby(['Id'])['Yesterday'].diff()
    # melt2['Yesterday-1'] = melt2.groupby(['Id'])['Calories'].shift(2)
    # melt2['Yesterday-1_Diff'] = melt2.groupby(['Id'])['Yesterday-1'].diff()
    melt2 = melt2.dropna()

    # df = melt2
    #train/test data split
    # df = df.loc[df['Id'] == id]
    mean_error = []

    # train, val = train_test_split(df, test_size=0.3, shuffle=False)
    X = melt2.drop(['TotalSteps', 'Day','Date'], axis=1)
    y = melt2['TotalSteps'].values

    #train the model
    loaded_model.fit(X, y)
    forecast = loaded_model.predict(X)


    # loaded_model.fit(df)
    # p = loaded_model.predict()
    return forecast


#function to visualise data
def plot_fc(df,forecast):
    df = df[['ActivityDate','TotalSteps']]
    dt = df.ActivityDate[-1:].values[0]
    future_dates = pd.Series([dt+np.timedelta64(td+1, 'D') for td in range(len(forecast))])
    forecast = pd.DataFrame({"ActivityDate":future_dates, 'forecast':forecast})
    forecast['ActivityDate'] = future_dates
    df = df.merge(forecast, how='outer')
    fig = px.line(df, x="ActivityDate", y=['TotalSteps', 'forecast'], width=1000,labels={"ActivityDate": 'Activity date', 'value': "Total number of steps"}, title="Total number of steps with forecast")
    fig['data'][0]['line']['color']='#FFFFFF'
    fig['data'][1]['line']['color']='#000000'
    fig.update_layout(
        xaxis=dict(
            showgrid=False
        ),
        yaxis=dict(
            showgrid=False
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor='#E42621'
        )
    )
    return fig


def dashboard():
    logo = Image.open('images/smart_health.png')
    st.sidebar.image(logo, caption=None, width=100, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

    st.sidebar.write('# Hello {}!'.format(st.session_state['login_name']))

    patient = st.sidebar.selectbox(
        "Choose a patient",
        ("Patient 1", "Patient 2", "Patient 3")
    )


    patients = {"Patient 1": 6962181067, "Patient 2":5553957443, "Patient 3": 4319703577}

    # thresholds
    TotalSteps_thresh = 4900  
    VeryActiveMinutes_thresh =  20
    SedentaryMinutes_thresh = 390
    TotalMinutesAsleep_thresh = 360
    TotalMinutesAwake_thresh = 60

    # patients details
    st.sidebar.write("# {}'s details".format(patient))
    st.sidebar.write("Age: {}".format(random.randint(20,70)))
    st.sidebar.write("Address: {} Orange street".format(random.randint(1,200)))
    st.sidebar.write("e-Mail: [{}@google.com]({}@google.com)".format(patient.replace(" ",""),patient.replace(" ","")))
    number = random.randint(10000000,99999999)
    st.sidebar.write("Tel: [06-{}](06-{})".format(number,number))
    st.sidebar.write("Patient ID: [{}]({})".format(patients[patient],patients[patient]))

    
    st.sidebar.button('Logout', on_click=logout)

    df = pd.read_csv("dailyActivity_selection.csv")
    df = df[df['Id']==patients[patient]]
    df['ActivityDate'] = pd.to_datetime(df.ActivityDate)

    st.write("# {}'s dashboard".format(patient))



    # Vulnerability classification
    vulnerable = predict_vulnerability(df,'models/LogReg.joblib', 3)
    if vulnerable == True:
        st.write("{} is classified as __vulnerable__ according to his/her lifestyle".format(patient))
    else:
        st.write("{} is classified as __healthy__ according to his/her lifestyle".format(patient))


    # personalized advice
    st.write("## {}'s stats".format(patient))
    c1,c2,c3,c4,c5 = st.columns(5)

    thresh = 5
    # TotalSteps
    c1.write("##### Steps")
    thresh_met = (df[:10]['TotalSteps']>TotalSteps_thresh).sum()
    advise(c1,thresh_met, thresh)
    c1.write("The threshold of {} steps was reached {} times in the past 10 days".format(TotalSteps_thresh,thresh_met))

    # VeryActiveMinutes
    c2.write("##### Very Active Time")
    thresh_met = (df[:10]['VeryActiveMinutes']>VeryActiveMinutes_thresh).sum()
    advise(c2,thresh_met, thresh)
    c2.write("The threshold of {} minutes was reached {} times in the past 10 days".format(VeryActiveMinutes_thresh, thresh_met))

    # SedentaryMinutes
    c3.write("##### Sedentary Time")
    thresh_met = (df[:10]['SedentaryMinutes']>SedentaryMinutes_thresh).sum()
    advise(c3,thresh_met, thresh, True)
    c3.write("The threshold of {} minutes was reached {} times in the past 10 days".format(SedentaryMinutes_thresh,thresh_met))

    # TotalMinutesAsleep
    c4.write("##### Time Asleep")
    thresh_met = (df[:10]['TotalMinutesAsleep']>TotalMinutesAsleep_thresh).sum()
    advise(c4,thresh_met, thresh)
    c4.write("The threshold of {} minutes was reached {} times in the past 10 days".format(TotalMinutesAsleep_thresh,thresh_met))

    # TotalMinutesAwake
    c5.write("##### Time Awake")
    thresh_met = (df[:10]['TotalMinutesAwake']>TotalMinutesAwake_thresh).sum()
    advise(c5,thresh_met, thresh, True)
    c5.write("The threshold of {} minutes was reached {} times in the past 10 days".format(TotalMinutesAwake_thresh, thresh_met))
    
    # Do forecasting
    forecast = forecast_data(df, "models/model.joblib", patient)
    fig = plot_fc(df, forecast)
    st.plotly_chart(fig)


    c1,c2 = st.columns(2)

    # ['TotalSteps', 'VeryActiveMinutes','SedentaryMinutes','TotalMinutesAsleep','TotalMinutesAwake']
    # total steps
    c1.plotly_chart(create_gauge(df, 'TotalSteps', "Total Steps"))
    c2.plotly_chart(create_bar(df,'TotalSteps', TotalSteps_thresh, patients[patient], title="Total Steps", y_title="Total number of steps"))

    # VeryActiveMinutes
    c1.plotly_chart(create_gauge(df, 'VeryActiveMinutes', "Active Time"))
    c2.plotly_chart(create_bar(df,'VeryActiveMinutes',VeryActiveMinutes_thresh, patients[patient], title="Active Time", y_title="Active time in minutes"))

    # SedentaryMinutes
    c1.plotly_chart(create_gauge(df, 'SedentaryMinutes', "Sedentary Time"))
    c2.plotly_chart(create_bar(df,'SedentaryMinutes', SedentaryMinutes_thresh, patients[patient], title="Sedentary Time", y_title="Sedentary time in minutes"))

    # TotalMinutesAsleep
    c1.plotly_chart(create_gauge(df, 'TotalMinutesAsleep', "Time Asleep"))
    c2.plotly_chart(create_bar(df,'TotalMinutesAsleep', TotalMinutesAsleep_thresh, patients[patient], title="Time Asleep", y_title="Time asleep in minutes"))

    # SedentaryMinutes
    c1.plotly_chart(create_gauge(df, 'TotalMinutesAwake', "Time Awake In Bed"))
    c2.plotly_chart(create_bar(df,'TotalMinutesAwake', TotalMinutesAwake_thresh, patients[patient], title="Time Awake In Bed", y_title="Time awake in bed in minutes"))




if __name__ == "__main__":
    # st.write(st.session_state)      # remove this line
    if 'login_name' in st.session_state:
            with st.spinner("loading"):
                dashboard()
    elif 'sign' in st.session_state:
        if st.session_state['sign'] == True:
            signup()
        elif st.session_state['sign'] == False and 'login_success' not in st.session_state:
            login()
        else:
            login()
    else: 
        login()