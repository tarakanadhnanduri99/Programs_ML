# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd

# Load the dataset
df = pd.read_csv('d:/restaurant_data.csv')

# Group by city to count the number of restaurants
city_restaurant_count = df.groupby('City')['Restaurant Name'].count().reset_index()
city_restaurant_count.columns = ['City', 'Restaurant Count']

# Identify the city with the highest number of restaurants
city_max_restaurants = city_restaurant_count.loc[city_restaurant_count['Restaurant Count'].idxmax()]

# Group by city and calculate the average rating
city_avg_rating = df.groupby('City')['Rating'].mean().reset_index()
city_avg_rating.columns = ['City', 'Average Rating']

# Identify the city with the highest average rating
city_max_avg_rating = city_avg_rating.loc[city_avg_rating['Average Rating'].idxmax()]

# Print the results
print("City with the highest number of restaurants:", city_max_restaurants['City'])
print("City with the highest average rating:", city_max_avg_rating['City'])

