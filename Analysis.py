import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sqlalchemy import create_engine
# Load your amended CSV file (adjust the file path accordingly)
#df = pd.read_csv(r"C:\Users\Jacob\OneDrive - Jigsaw PSHE Ltd\Documents\Python\Neighbour_Analysis\HS_PSHE_RE_DATA_with_lat_lon_MASTER.csv")



server = ''
database = ''
username = ''
password = ''
connection_string = f'mssql+pyodbc://{username}:{password}@{server}/{database}?driver=SQL+Server'

# Create an engine
engine = create_engine(connection_string)

# Write your SQL query to load data into the DataFrame
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


df = df.dropna(subset=['latitude', 'longitude'])


# Separate Primary SaaS and Primary Legacy customers
primary_saas = df[df['Customer type - Primary'] == 'SaaS'].copy()
primary_legacy = df[df['Customer type - Primary'] == 'Legacy'].copy()

# Separate RE SaaS and RE Legacy customers
re_saas = df[df['Customer type - RE'] == 'SaaS'].copy()
re_legacy = df[df['Customer type - RE'] == 'Legacy'].copy()

# Function to find the top 5 nearest SaaS for a given Legacy dataset
def find_nearest_legacy_saas(legacy_df, saas_df):
    # Filter out rows where latitude or longitude is 'Blank'
    valid_legacy = legacy_df[legacy_df['latitude'] != 'Blank']
    valid_saas = saas_df[saas_df['latitude'] != 'Blank']
    
    if valid_legacy.empty or valid_saas.empty:
        # If no valid coordinates, return empty DataFrame and empty array
        return pd.DataFrame(), np.array([]), []

    legacy_coords = valid_legacy[['latitude', 'longitude']].values.astype(float)
    saas_coords = valid_saas[['latitude', 'longitude']].values.astype(float)

    # Use NearestNeighbors to find the top 5 closest SaaS for each Legacy
    nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree', metric='haversine').fit(np.radians(saas_coords))
    distances, indices = nbrs.kneighbors(np.radians(legacy_coords))

    # Convert distances from radians to kilometers and then to miles
    distances_in_miles = distances * 6371 * 0.621371

    # Get the closest SaaS customer information, including Record ID, Company name, and Customer type
    closest_saas = valid_saas.iloc[indices.flatten()][['Record ID', 'Company name', 'Customer type - Primary', 'Customer type - RE']].reset_index(drop=True)

    # Reshape the distances and school names for easy assignment later
    closest_saas = closest_saas.groupby(np.arange(len(closest_saas)) // 5).apply(lambda x: x.reset_index(drop=True))
    
    return closest_saas, distances_in_miles, valid_legacy.index

# Find the nearest Primary SaaS for each Primary Legacy
closest_primary_saas, distances_to_primary_saas, valid_legacy_indices = find_nearest_legacy_saas(primary_legacy, primary_saas)

# Initialize new columns for the 5 schools, distances, types, and Record IDs
for i in range(1, 6):
    primary_legacy[f'School {i}'] = "Blank"
    primary_legacy[f'Distance to School {i} (miles)'] = "Blank"
    primary_legacy[f'School {i} Type'] = "Blank"  # To store whether SaaS or Legacy
    primary_legacy[f'School {i} Record ID'] = "Blank"  # To store Record ID

# Assign the top 5 closest Primary SaaS, distances, types, and Record IDs to the valid rows (iterating row by row)
for i, idx in enumerate(valid_legacy_indices):
    # Extract the top 5 schools, distances, types, and Record IDs for the current row
    top_5_saas = closest_primary_saas.iloc[i * 5:(i + 1) * 5][['Company name', 'Customer type - Primary', 'Record ID']].values  # Extract the 5 schools, types, and Record IDs
    top_5_distances = distances_to_primary_saas[i]  # Extract the 5 distances for this record
    
    # Assign each school, distance, type, and Record ID to its respective column
    for j in range(5):
        primary_legacy.loc[idx, f'School {j+1}'] = top_5_saas[j][0]  # School name
        primary_legacy.loc[idx, f'Distance to School {j+1} (miles)'] = top_5_distances[j]  # Distance
        primary_legacy.loc[idx, f'School {j+1} Type'] = top_5_saas[j][1]  # Type (SaaS or Legacy)
        primary_legacy.loc[idx, f'School {j+1} Record ID'] = top_5_saas[j][2]  # Record ID

# Find the nearest RE SaaS for each RE Legacy
closest_re_saas, distances_to_re_saas, valid_legacy_indices_re = find_nearest_legacy_saas(re_legacy, re_saas)

# Initialize new columns for the 5 schools, distances, types, and Record IDs for RE
for i in range(1, 6):
    re_legacy[f'School {i}'] = "Blank"
    re_legacy[f'Distance to School {i} (miles)'] = "Blank"
    re_legacy[f'School {i} Type'] = "Blank"  # To store whether SaaS or Legacy
    re_legacy[f'School {i} Record ID'] = "Blank"  # To store Record ID

# Assign the top 5 closest RE SaaS, distances, types, and Record IDs to the valid rows (iterating row by row)
for i, idx in enumerate(valid_legacy_indices_re):
    # Extract the top 5 schools, distances, types, and Record IDs for the current row
    top_5_saas = closest_re_saas.iloc[i * 5:(i + 1) * 5][['Company name', 'Customer type - RE', 'Record ID']].values  # Extract the 5 schools, types, and Record IDs
    top_5_distances = distances_to_re_saas[i]  # Extract the 5 distances for this record
    
    # Assign each school, distance, type, and Record ID to its respective column
    for j in range(5):
        re_legacy.loc[idx, f'School {j+1}'] = top_5_saas[j][0]  # School name
        re_legacy.loc[idx, f'Distance to School {j+1} (miles)'] = top_5_distances[j]  # Distance
        re_legacy.loc[idx, f'School {j+1} Type'] = top_5_saas[j][1]  # Type (SaaS or Legacy)
        re_legacy.loc[idx, f'School {j+1} Record ID'] = top_5_saas[j][2]  # Record ID




#print("Primary Legacy DataFrame with Closest SaaS Schools:")
#print(primary_legacy[['Record ID', 'Postcode', 'Customer type - Primary', 'School 1', 'Distance to School 1 (miles)', 'School 1 Type', 'School 1 Record ID']].head())

#primary_legacy.to_csv(r"C:\Users\Jacob\Downloads\Primary_Legacy_with_Closest_SaaS.csv", index=False)

