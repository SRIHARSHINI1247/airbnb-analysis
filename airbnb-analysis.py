"""
Airbnb Hotel Booking Analysis
This script is ready to paste directly into Google Colab
"""

# ---------------------------
# Step 1: Install required packages
# ---------------------------
import sys
import subprocess

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

packages = ["pandas", "matplotlib", "seaborn", "plotly", "openpyxl"]
for pkg in packages:
    try:
        __import__(pkg)
    except ImportError:
        install(pkg)

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from google.colab import files
import io

print("All packages installed and imported!")

# ---------------------------
# Step 2: Upload Dataset
# ---------------------------
print("\nPlease upload your Airbnb Excel/CSV dataset.")
uploaded = files.upload()

file_name = list(uploaded.keys())[0]
if file_name.endswith('.xlsx'):
    df = pd.read_excel(io.BytesIO(uploaded[file_name]))
else:
    df = pd.read_csv(io.BytesIO(uploaded[file_name]))

print("Dataset loaded! Shape:", df.shape)
display(df.head())

# ---------------------------
# Step 3: Data Cleaning & Column Mapping
# ---------------------------
# Map dataset columns to expected names
column_mapping = {
    'last review': 'booking_date',
    'review rate number': 'guest_rating',
    'neighbourhood': 'location',
    'price': 'price'  # adjust if your dataset uses a different column for price
}
df.rename(columns=column_mapping, inplace=True)

# Convert types
df['booking_date'] = pd.to_datetime(df['booking_date'], errors='coerce')
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df['guest_rating'] = pd.to_numeric(df['guest_rating'], errors='coerce')

# Drop rows with missing critical values
df.dropna(subset=['booking_date', 'location', 'price', 'guest_rating'], inplace=True)
print("Data cleaned! Remaining rows:", len(df))

# ---------------------------
# Step 4: Booking Patterns
# ---------------------------
df['month'] = df['booking_date'].dt.month
monthly_bookings = df.groupby('month').size().reset_index(name='bookings')

plt.figure(figsize=(10,5))
sns.lineplot(data=monthly_bookings, x='month', y='bookings', marker='o')
plt.title('Monthly Booking Trend')
plt.xlabel('Month')
plt.ylabel('Number of Bookings')
plt.show()

# Location-wise bookings
location_counts = df['location'].value_counts().reset_index()
location_counts.columns = ['location', 'bookings']

plt.figure(figsize=(12,6))
sns.barplot(data=location_counts.head(10), x='location', y='bookings', palette='viridis')
plt.title('Top 10 Locations by Bookings')
plt.xticks(rotation=45)
plt.show()

# ---------------------------
# Step 5: Pricing Analysis
# ---------------------------
plt.figure(figsize=(10,5))
sns.boxplot(data=df, x='location', y='price')
plt.title('Price Distribution by Location')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10,5))
sns.scatterplot(data=df, x='price', y='guest_rating', alpha=0.6)
plt.title('Price vs Guest Rating')
plt.xlabel('Price')
plt.ylabel('Guest Rating')
plt.show()

# ---------------------------
# Step 6: Guest Preferences
# ---------------------------
if 'amenities' in df.columns:
    all_amenities = df['amenities'].dropna().str.split(',').explode()
    top_amenities = all_amenities.value_counts().head(10)

    plt.figure(figsize=(10,5))
    sns.barplot(x=top_amenities.values, y=top_amenities.index, palette='coolwarm')
    plt.title('Top 10 Most Desired Amenities')
    plt.xlabel('Count')
    plt.ylabel('Amenity')
    plt.show()

# ---------------------------
# Step 7: Host Performance
# ---------------------------
if 'host_response_time' in df.columns:
    plt.figure(figsize=(10,5))
    sns.countplot(y='host_response_time', data=df, order=df['host_response_time'].value_counts().index)
    plt.title('Host Response Time Distribution')
    plt.show()

if 'guest_complaints' in df.columns:
    complaints = df['guest_complaints'].dropna().str.split(',').explode().value_counts().head(10)
    plt.figure(figsize=(10,5))
    sns.barplot(x=complaints.values, y=complaints.index, palette='magma')
    plt.title('Top 10 Guest Complaints')
    plt.show()

# ---------------------------
# Step 8: Interactive Plotly Chart
# ---------------------------
fig = px.scatter(df, x='price', y='guest_rating', color='location', hover_data=['booking_date'])
fig.update_layout(title='Interactive Price vs Guest Rating by Location')
fig.show()

print("\nAirbnb analysis completed successfully!")
