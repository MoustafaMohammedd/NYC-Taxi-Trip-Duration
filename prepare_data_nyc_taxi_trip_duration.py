import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

from geopy import distance
from geopy.point import Point
import math

df_train =pd.read_csv(r"/content/drive/MyDrive/NYK/train.csv")
df_val =pd.read_csv(r"/content/drive/MyDrive/NYK/val.csv")
df_test=pd.read_csv(r"/content/drive/MyDrive/NYK/test.csv")
pd.set_option("display.max_columns",50)

def prepare_data(df):

    def range_months(x):
        if 1<=x<=2 :
            return 0  # winter
        elif 3<=x<=5:
            return 1  # spring
        else:
            return 2  # summer      #we know that data for 6 months from 1 to 6

    def range_hours(x):
        if 5<=x<12 :
            return 0  # Morning
        elif 12<=x<17:
            return 1  # Afternoon
        else:
            return 2  # Night

    def range_minutes(x):
        if 1<=x<=15 :
            return 0   #Q1
        elif 16<=x<=30:
            return 1   #Q2
        elif 31<=x<=45:
            return 2   #Q3
        else:
            return 3   #Q4

    def haversine_distance(row):
        pick = Point(row['pickup_latitude'], row['pickup_longitude'])
        drop = Point(row['dropoff_latitude'], row['dropoff_longitude'])
        dist = distance.geodesic(pick, drop)
        return dist.km

    def calculate_direction(row):
        pickup_coordinates =  Point(row['pickup_latitude'], row['pickup_longitude'])
        dropoff_coordinates = Point(row['dropoff_latitude'], row['dropoff_longitude'])
        # Calculate the difference in longitudes
        delta_longitude = dropoff_coordinates[1] - pickup_coordinates[1]
        # Calculate the bearing (direction) using trigonometry
        y = math.sin(math.radians(delta_longitude)) * math.cos(math.radians(dropoff_coordinates[0]))
        x = math.cos(math.radians(pickup_coordinates[0])) * math.sin(math.radians(dropoff_coordinates[0])) - \
            math.sin(math.radians(pickup_coordinates[0])) * math.cos(math.radians(dropoff_coordinates[0])) * \
            math.cos(math.radians(delta_longitude))
        # Calculate the bearing in degrees
        bearing = math.atan2(y, x)
        bearing = math.degrees(bearing)
        # Adjust the bearing to be in the range [0, 360)
        bearing = (bearing + 360) % 360

        return bearing

    def manhattan_distance(row):

        lat_distance = abs(row['pickup_latitude'] - row['dropoff_latitude']) * 111  # approx 111 km per degree latitude
        lon_distance = abs(row['pickup_longitude'] - row['dropoff_longitude']) * 111 * math.cos(math.radians(row['pickup_latitude']))  # adjust for latitude

        return lat_distance + lon_distance

    # datetime
    df["pickup_datetime"]=pd.to_datetime(df["pickup_datetime"])
    df["pickup_hour"]=df["pickup_datetime"].dt.hour
    df["pickup_month"]=df["pickup_datetime"].dt.month
    df["pickup_day"]=df["pickup_datetime"].dt.day
    df["pickup_dayofweek"]=df["pickup_datetime"].dt.day_name()
    df["pickup_minute"]=df["pickup_datetime"].dt.minute

    df["log_trip_duration"]=np.log1p(df["trip_duration"])

    df["is_weekend"]=((df["pickup_dayofweek"]=="Sunday") | (df["pickup_dayofweek"]=="Saturday")).astype(int)
    df["range_minutes"]=df["pickup_minute"].apply(lambda x:range_minutes(x))
    df["range_hours"]=df["pickup_hour"].apply(lambda x:range_hours(x))
    df["range_months"]=df["pickup_month"].apply(lambda x:range_months(x))

    df['distance_haversine'] = df.apply(haversine_distance, axis=1)
    df['direction'] = df.apply(calculate_direction, axis=1)
    df['distance_manhattan'] = df.apply(manhattan_distance, axis=1)

    df.drop(columns=['id','pickup_datetime','trip_duration','pickup_minute'],inplace=True)

    return df

cat_features=['vendor_id', 'passenger_count','store_and_fwd_flag','pickup_hour', 'pickup_month', 'pickup_day', 'pickup_dayofweek', 'is_weekend', 'range_minutes','range_hours', 'range_months']
num_features =[ 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude','dropoff_latitude','distance_haversine', 'direction','distance_manhattan']

df_train=prepare_data(df_train)
df_val=prepare_data(df_val)
df_test=prepare_data(df_test)

