import streamlit as st
import numpy as np
import pickle

# Load model and scaler
@st.cache_resource
def load_model():
    with open("k_mean.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

# Prediction logic
def get_cluster_message(cluster_id):
    messages = {
        0: "Cluster 0: Average income and spending.",
        1: "Cluster 1: High income, high spending — Premium Customers.",
        2: "Cluster 2: Low income, low spending — Budget Customers.",
        3: "Cluster 3: High income, low spending — Cautious Customers.",
        4: "Cluster 4: Low income, high spending — Impulsive Customers.",
    }
    return messages.get(cluster_id, "Unknown cluster")

def predict_cluster(age, income, score):
    input_data = np.array([[age, income, score]])
    input_scaled = scaler.transform(input_data)
    cluster = model.predict(input_scaled)[0]
    return cluster, get_cluster_message(cluster)

# Streamlit UI
st.title("🧠 Customer Segmentation using K-Means")
st.write("Enter customer details to predict their segment.")

age = st.number_input("Customer Age", min_value=10, max_value=100, step=1)
income = st.number_input("Annual Income (k$)", min_value=0.0)
score = st.number_input("Spending Score (1–100)", min_value=0.0, max_value=100.0)

if st.button("Predict Segment"):
    cluster_id, message = predict_cluster(age, income, score)
    st.success(f"Predicted Cluster: {cluster_id}")
    st.info(message)
