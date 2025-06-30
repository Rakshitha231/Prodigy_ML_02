import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pickle

# Load your dataset
df = pd.read_csv("Mall_Customers.csv")

# Select relevant features
features = df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]

# Scale the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(scaled_features)

# Save the model and scaler
with open("k_mean.pkl", "wb") as f:
    pickle.dump(kmeans, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Model and scaler saved as 'k_mean.pkl' and 'scaler.pkl'")
