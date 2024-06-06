import geopandas as gpd
import fiona
import numpy as np
import os
from shapely import geometry, Point
from matplotlib import pyplot as plt
from scipy.spatial import cKDTree
import tqdm as tqdm
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Set pandas display option to avoid scientific notation
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# Load shapefiles
bathymetry = gpd.read_file(r"C:\Users\funkt\OneDrive\Desktop\176cproject\bathymety_vector_clipped\bathymety_vector_clipped.shp")
windspeed = gpd.read_file(r"C:\Users\funkt\OneDrive\Desktop\176cproject\windspeed\windspeedclipped.shp")
transmission = gpd.read_file(r"C:\Users\funkt\OneDrive\Desktop\176cproject\Transmission_Line_7304850911266214096\TransmissionLine_CEC.shp")
suitable_area = gpd.read_file(r"C:\Users\funkt\OneDrive\Desktop\176cproject\drive-download-20240517T231941Z-001\suitablearea.shp")
ports = gpd.read_file(r"C:\Users\funkt\OneDrive\Desktop\176cproject\Ports\RR_Ports.shp")  # Update this path with the actual path to your ports shapefile

# Set the CRS to California State Plane Zone 6 (EPSG:3310) - units in feet
crs = 'EPSG:3310'

# Reproject all GeoDataFrames to the new CRS
bathymetry = bathymetry.to_crs(crs)
windspeed = windspeed.to_crs(crs)
transmission = transmission.to_crs(crs)
suitable_area = suitable_area.to_crs(crs)
ports = ports.to_crs(crs)



parcels = bathymetry.copy()
parcels = gpd.sjoin(parcels, windspeed, how='left', predicate='intersects')
parcels = parcels[['OBJECTID', 'geometry', 'gridcode', 'value']]
parcels.rename(columns={'value': 'windspeed', 'gridcode': 'bathymetry'}, inplace=True)
#parcels = parcels.head(100)  # Use a subset for quicker debugging

# Verify the CRS of the parcels GeoDataFrame
print("CRS of parcels GeoDataFrame:", parcels.crs)

# Calculate distances from parcel centroids to nearest transmission linestring and port, and store distances in new columns
def distance_to_features(parcels, features):
    distances = []
    for index, parcel in tqdm.tqdm(parcels.iterrows(), total=parcels.shape[0]):
        centroid = parcel['geometry'].centroid
        nearest_feature = features.geometry.distance(centroid).min()
        distances.append(nearest_feature)
    return distances

parcels['distances'] = distance_to_features(parcels, transmission)
parcels['distance_to_port'] = distance_to_features(parcels, ports)

# Calculate cost for each parcel
def calculate_cost(parcels, cost_factors):
    parcels['cost'] = (
        parcels['bathymetry'].abs() * cost_factors['depth'] + 
        parcels['distances'] * cost_factors['distance_to_transmission'] + 
        parcels['distance_to_port'] * cost_factors['distance_to_port']
    )
    return parcels

# Calculate KWH for each parcel
def calculate_kwh(parcels, power_curve_factor, hours_per_year=175200):
    parcels['kwh'] = parcels['windspeed'] * power_curve_factor * hours_per_year
    return parcels

# Calculate KWH per cost for each parcel
def calculate_kwh_per_cost(parcels):
    parcels['kwh_per_cost'] = parcels['kwh'] / (parcels['cost'] + 1e-6)
    return parcels

# Define cost factors and power curve factor
cost_factors = {
    'depth': 100,  # Cost per meter depth
    'distance_to_transmission': 50,  # Cost per kilometer distance to transmission
    'distance_to_port': 75  # Cost per kilometer distance to port
}
power_curve_factor = 500  # Example factor converting wind speed to KWH

parcels = calculate_cost(parcels, cost_factors)
parcels = calculate_kwh(parcels, power_curve_factor)
parcels = calculate_kwh_per_cost(parcels)

# Normalize KWH per cost to a score between 1 and 10
def normalize_scores(parcels):
    min_score = 1
    max_score = 10
    scaler = MinMaxScaler(feature_range=(min_score, max_score))
    parcels['score'] = scaler.fit_transform(parcels[['kwh_per_cost']])
    return parcels

parcels = normalize_scores(parcels)
print(parcels.head())  # Print a sample of the parcels DataFrame to check the results

# Sort parcels based on the score to determine the best locations
sorted_parcels = parcels.sort_values('score', ascending=False)

sorted_parcels.to_file(r"C:\Users\funkt\OneDrive\Desktop\176cproject\updated_parcels.shp")
