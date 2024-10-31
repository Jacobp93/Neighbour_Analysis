import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np
from geopy.geocoders import OpenCage  # Use OpenCage instead of Nominatim
from sqlalchemy import create_engine
import os
import pyodbc as db
# Load your amended CSV file (adjust the file path accordingly)
#df = pd.read_csv(r"C:\Users\Jacob\OneDrive - Jigsaw PSHE Ltd\Documents\Python\Neighbour_Analysis\HS_PSHE_RE_DATA_with_lat_lon_MASTER.csv")



SQL_SERVER = st.secrets["sql"]["server"]
SQL_DATABASE = st.secrets["sql"]["database"]
SQL_UID = st.secrets["sql"]["user"]
SQL_PASS = st.secrets["sql"]["password"]
OPENCAGE_API_KEY = st.secrets["api_keys"]["opencage"]  # Example assuming OpenCage API key is also in secrets
driver = '{ODBC Driver 17 for SQL Server}'

try:
    conn = db.connect(
        f'DRIVER={driver};'
        f'SERVER={SQL_SERVER};'
        f'DATABASE={SQL_DATABASE};'
        f'UID={SQL_UID};'
        f'PWD={SQL_PASS};'  # Include password for authentication
        'Trusted_Connection=no;'
    )
    st.write("Connection established successfully")
except db.Error as e:
    st.error(f"Error connecting to database: {e}")
    st.stop()


geolocator = OpenCage(api_key="OPENCAGE_API_KEY")


# Caching function for geocoding to avoid redundant API calls
@st.cache_data
def get_geocode(postcode):
    return geolocator.geocode(postcode)

# Load and clean data from SQL and separate into datasets
@st.cache_data
def load_data():
    engine = create_engine(driver)  # Update with your connection string
    
    # SQL query to load data into DataFrame
    query = """
    SELECT 
        id AS [Record ID],
        property_name AS [Company name], 
        property_post_code AS [Postcode], 
        property_customer_type AS [Customer type - Primary], 
        property_customer_type_re AS [Customer type - RE], 
        property_longitude AS [longitude], 
        property_latitude AS [latitude]
    FROM 
        _hubspot.company;
    """
    
    # Load data into DataFrame
    df = pd.read_sql_query(query, engine)

    # Drop rows with missing coordinates
    df = df.dropna(subset=['latitude', 'longitude'])
    
    # Convert coordinates to numeric to handle non-numeric entries gracefully
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df = df.dropna(subset=['latitude', 'longitude'])  # Drop any rows that couldn't be converted
    
    # Separate into primary and RE SaaS and Legacy datasets
    primary_saas = df[df['Customer type - Primary'] == 'SaaS'].copy()
    primary_legacy = df[df['Customer type - Primary'] == 'Legacy'].copy()
    re_saas = df[df['Customer type - RE'] == 'SaaS'].copy()
    re_legacy = df[df['Customer type - RE'] == 'Legacy'].copy()
    
    return primary_saas, primary_legacy, re_saas, re_legacy

# Load data into respective DataFrames
primary_saas, primary_legacy, re_saas, re_legacy = load_data()

# Function to find nearest SaaS locations for a given Legacy dataset within a radius
def find_nearest_locations(target_location, data, radius=10):
    if data.empty:
        return pd.DataFrame()

    # Convert target location and dataset coordinates to radians for haversine calculations
    target_coords = np.radians([[target_location.latitude, target_location.longitude]])
    data_coords = np.radians(data[['latitude', 'longitude']].values.astype(float))
    
    # Initialize NearestNeighbors with radius in meters
    nbrs = NearestNeighbors(radius=radius * 1609.34, algorithm='ball_tree', metric='haversine')
    nbrs.fit(data_coords)
    
    # Find neighbors within the radius
    distances, indices = nbrs.radius_neighbors(target_coords)
    
    # Convert distances from radians to miles and filter within radius
    results = []
    for dist_list, idx_list in zip(distances, indices):
        dist_miles = dist_list * 6371 * 0.621371
        within_radius = [(dist, idx) for dist, idx in zip(dist_miles, idx_list) if dist <= radius]
        
        # Select only the top 5 closest results
        top_5_within_radius = sorted(within_radius, key=lambda x: x[0])[:5]
        
        # Retrieve SaaS details and distances
        nearest_data = data.iloc[[idx for _, idx in top_5_within_radius]][['Record ID', 'Company name', 'Customer type - Primary', 'Customer type - RE']].copy()
        nearest_data['Distance (miles)'] = [dist for dist, _ in top_5_within_radius]
        results.append(nearest_data)
    
    if results:
        return pd.concat(results, ignore_index=True)
    else:
        return pd.DataFrame()

# Streamlit app layout
st.title("School Nearest Neighbor Finder")
st.write("Search for the top 5 closest schools by entering a postcode.")

# Postcode input
postcode = st.text_input("Enter a postcode:", "")

# Checkbox for selecting Primary Legacy or Jigsaw RE
search_primary_legacy = st.checkbox("Search Primary PSHE", value=True)
search_jigsaw_re = st.checkbox("Search Jigsaw RE", value=False)

# Set a search radius
radius = st.slider("Set Search Radius (in miles)", min_value=1, max_value=50, value=10)

# Find the nearest neighbors based on user selection
if st.button("Search"):
    if postcode:
        location = get_geocode(postcode)
        if location is None:
            st.error("Could not locate the postcode.")
        else:
            selected_datasets = []

            if search_primary_legacy:
                st.subheader("Searching in Primary Legacy dataset...")
                selected_datasets.append(primary_saas)  # Use `primary_saas` for finding nearest SaaS

            if search_jigsaw_re:
                st.subheader("Searching in Jigsaw RE dataset...")
                selected_datasets.append(re_saas)  # Use `re_saas` for Jigsaw RE data

            # Concatenate selected datasets if any are selected
            combined_data = pd.concat(selected_datasets, ignore_index=True) if selected_datasets else pd.DataFrame()

            if not combined_data.empty:
                nearest_schools = find_nearest_locations(location, combined_data, radius=radius)
                
                # Display the results
                if not nearest_schools.empty:
                    st.write(f"Top 5 closest schools within {radius} miles:")
                    st.table(nearest_schools)
                else:
                    st.error("No schools found within the specified radius.")
            else:
                st.error("No datasets were selected for the search.")
    else:
        st.error("Please enter a postcode.")