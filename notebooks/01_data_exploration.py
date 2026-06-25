import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv(
    "data/US_Accidents_March23.csv",
    nrows=5000
)

# Convert Start_Time to datetime
df["Start_Time"] = pd.to_datetime(df["Start_Time"])

# Extract hour
df["Hour"] = df["Start_Time"].dt.hour

# Count accidents per hour
hourly_counts = df["Hour"].value_counts().sort_index()

print(hourly_counts)

# Plot
plt.figure(figsize=(10,5))
hourly_counts.plot(kind="line", marker="o")

plt.title("Accidents by Hour of Day")
plt.xlabel("Hour")
plt.ylabel("Number of Accidents")

plt.grid(True)

plt.tight_layout()

plt.savefig("screenshots/accidents_by_hour.png")

plt.show()