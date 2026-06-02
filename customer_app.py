import pandas as pd
import numpy as np 
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import joblib

# Configuration of page
st.set_page_config(
    page_title= "Customer segmentation",     
    layout="wide"
) 


# custom css
st.markdown("""

<style>
.main {padding: 0rem 1rem;}
h1 {color:#9b59b6; padding.bottom: 1rem;}
</style>
""", unsafe_allow_html= True)

# loading model and data
@st.cache_resource
def load_model_and_data():
    try:  
        kmeans = joblib.load("kmeans_model.pkl")
        scaler = joblib.load("scaler.pkl")

        data = pd.read_csv("Mall_Customers.csv")
        x = data.iloc[:, [3, 4]].values
        data['Cluster'] = kmeans.predict(scaler.transform(x))
        return kmeans, scaler, data
    except FileNotFoundError:
        return None, None, None
kmeans, scaler, data = load_model_and_data()

income = st.number_input("Annual Income (k$)", min_value=0, max_value=200, value=50)
score = st.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=50)

if st.button("Predict Cluster"):

    customer_scaled = scaler.transform([[income, score]])

    predicted_cluster = kmeans.predict(customer_scaled)[0]

    st.write(f"Customer belongs to Cluster {predicted_cluster}")

# this shows scatter_plot
# this shows scatter_plot
if data is not None:
    fig = px.scatter(
        data,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        color="Cluster",
        title="Customer Segmentation Result"
    )
    st.plotly_chart(fig)
