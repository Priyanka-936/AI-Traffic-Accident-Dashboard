import pandas as pd
from sklearn.cluster import KMeans

# Load data
df = pd.read_csv(
    "data/US_Accidents_March23.csv",
    nrows=5000
)

# Coordinates
coords = df[["Start_Lat", "Start_Lng"]].dropna()

# KMeans clustering
kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

kmeans.fit(coords)

# Cluster centers
centers = kmeans.cluster_centers_

print("\nAccident Hotspot Centers:")
print(centers)