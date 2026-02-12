
import streamlit as st
import pandas as pd
df = pd.read_csv('fast_data.csv')
st.title("🤖 IBM Project Dashboard")
st.metric("Avg Response Time", f"{df['response_time_hours'].mean():.1f} hrs")
platform = st.selectbox("Platform", df['platform'].unique())
st.bar_chart(df[df['platform']==platform]['sentiment'].value_counts())
if st.button("AI Response"):
    st.success("Drafting... Done! 'We apologize for the inconvenience.'")
        