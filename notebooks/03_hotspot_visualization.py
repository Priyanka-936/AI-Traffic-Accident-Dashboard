import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load data
df = pd.read_csv(
    "data/US_Accidents_March23.csv",
    nrows=5000
)

coords = df[["Start_Lat", "Start_Lng"]].dropna()

# Train KMeans
kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(coords)

# Plot clusters
plt.figure(figsize=(10, 6))

plt.scatter(
    coords["Start_Lng"],
    coords["Start_Lat"],
    c=clusters,
    s=8
)

# Plot cluster centers
plt.scatter(
    kmeans.cluster_centers_[:, 1],
    kmeans.cluster_centers_[:, 0],
    marker="X",
    s=250
)

plt.title("Accident Hotspot Detection")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.tight_layout()

plt.savefig("screenshots/hotspot_detection.png")

plt.show()