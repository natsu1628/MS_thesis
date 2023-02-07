import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import datetime

# import leafmap.leafmap as leafmap

data_path = "/data/data_location_10152022.csv"

st.title("Location data visualization")
st.markdown("This app provides a visualization to the location data collected as part of the thesis work")


def haversine_distance(lon1, lat1, lon2, lat2, **kwargs):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    All args must be of equal length.
    
    Sourced from https://stackoverflow.com/a/29546836/11637704
    
    Thanks to derricw!
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km


@st.cache(persist=True)
def load_data():
	data_df = pd.read_csv(data_path)

	# Group the data by user
	group_sort_data = data_df.groupby("user").apply(pd.DataFrame.sort_values, "timestamp").reset_index(drop=True)

	# Remove user_4 data from the grouped data
	group_sort_data = group_sort_data[group_sort_data["user"]!="user_4"]

	# Convert the timestamp to datetime object
	group_sort_data["dt"] = pd.to_datetime(group_sort_data["timestamp"])

	# add a date column and timezone column in the data
	group_sort_data["tz"] = group_sort_data["dt"].apply(lambda x: datetime.timedelta(seconds=x.utcoffset().total_seconds()))
	group_sort_data["date"] = group_sort_data["dt"].apply(lambda x: x.strftime("%Y-%m-%d"))

	# Create a tuple of (latitude, longitude)
	group_sort_data["location"] = group_sort_data[["latitude", "longitude"]].apply(tuple, axis=1)

	# Get only the date and hour in the timestamp
	group_sort_data["datetime"] = group_sort_data["dt"].apply(lambda x: x.strftime("%Y-%m-%dT%H:00:00"))

	# Get ready for the clustering of the data
	cluster_group_data = group_sort_data[["user", "latitude", "longitude", "location", "timestamp", "dt", "datetime"]]	

	return group_sort_data

loc_df = load_data()

all_users = sorted(loc_df["user"].unique())

st.header("Mapping of location data based on filter conditions: user and dates")
user = st.selectbox(
	"Enter the user", options=all_users)
user_data = loc_df[loc_df["user"]==user]
date_select = st.selectbox("Enter the date travelled by user", options=user_data["date"].unique())
time_user_data = user_data.query("date in @date_select")

color = [10]*time_user_data.shape[0]
color[0] = 40
color[-1] = 20

size = [x//10 for x in color]

# Plot 1
# st.map(time_user_data[["latitude", "longitude"]], zoom=8)
fig1 = px.scatter_mapbox(time_user_data, lat="latitude", lon="longitude", color=color, size=size, hover_data=["latitude", "longitude", "timestamp", "charging_type", "wifi_status"], zoom=12)
fig1.update_layout(mapbox_style="carto-darkmatter")
fig1.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

st.plotly_chart(fig1, use_container_width=False)


# Plot 2
# time_user_data["color"] = color

time_user_data.sort_values(by=["timestamp"], inplace=True)
fig2 = px.scatter_mapbox(time_user_data, lat="latitude", lon="longitude", color=color, size=size, hover_data=["latitude", "longitude", "timestamp", "charging_type", "wifi_status"], zoom=12)
fig2.update_layout(mapbox_style="open-street-map")
fig2.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

st.plotly_chart(fig2, use_container_width=False)

st_time = time_user_data["timestamp"].iloc[0]
en_time = time_user_data["timestamp"].iloc[time_user_data.shape[0]-1]
st.markdown(f"Start time: {st_time}")
st.markdown(f"End Time: {en_time}")

# Plot 3
st.markdown("Heat map of user based on the locations visited")
fig3 = px.density_mapbox(time_user_data, lat="latitude", lon="longitude", z="location", mapbox_style="carto-darkmatter", zoom=12)
fig3.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
st.plotly_chart(fig3, use_container_width=False)

# Plot 4
# fig4 = px.line_geo(time_user_data, lat="latitude", lon="longitude", color=color, markers=True, projection="mercator")
# fig4.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
# st.plotly_chart(fig4, use_container_width=False)