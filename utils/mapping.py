import folium
from folium.plugins import HeatMap, MarkerCluster
import pandas as pd
import numpy as np

from utils.data_loader import get_project_root

import json
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
from geopandas import GeoDataFrame
import geopandas as gpd


def generate_base_map(default_location=[12.9716, 77.5946], default_zoom_start=12):
    base_map = folium.Map(
        location=default_location,
        zoom_start=default_zoom_start,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    bangalore_geojson_path = get_project_root() / "data/layers/bangalore.geojson"
    forest_geojson_path = get_project_root() / "data/layers/forest.geojson"

    if bangalore_geojson_path.exists():
        try:
            with open(bangalore_geojson_path, "r") as f:
                bangalore_geojson_data = json.load(f)

            # Add original Bangalore boundary layer
            folium.GeoJson(
                bangalore_geojson_data,
                name="Bangalore Wards Boundary",
                style_function=lambda x: {
                    "fillColor": "#228B22",
                    "color": "#000000",
                    "weight": 0.75,
                    "fillOpacity": 0.15,
                },
            ).add_to(base_map)

            if forest_geojson_path.exists():
                with open(forest_geojson_path, "r") as f:
                    forest_geojson_data = json.load(f)

                # Convert GeoJSON to GeoDataFrame
                bangalore_gdf = GeoDataFrame.from_features(
                    bangalore_geojson_data["features"]
                )
                forest_gdf = GeoDataFrame.from_features(forest_geojson_data["features"])

                # Subtract forest areas from Bangalore
                bangalore_geometry = unary_union(bangalore_gdf.geometry)
                forest_geometry = unary_union(forest_gdf.geometry)
                updated_geometry = bangalore_geometry.difference(forest_geometry)

                # Convert back to GeoJSON
                updated_geojson = mapping(updated_geometry)
                forest_subtracted_geojson_data = {
                    "type": "FeatureCollection",
                    "features": [{"type": "Feature", "geometry": updated_geojson}],
                }

                # Add forest-subtracted boundary layer
                folium.GeoJson(
                    forest_subtracted_geojson_data,
                    name="Bangalore Without Forest",
                    style_function=lambda x: {
                        "fillColor": "#FF4500",
                        "color": "#000000",
                        "weight": 0.75,
                        "fillOpacity": 0.15,
                    },
                ).add_to(base_map)

        except Exception as e:
            print(f"Error loading GeoJSON: {e}")
    else:
        print(f"Warning: GeoJSON file not found at {bangalore_geojson_path}")

    # Dictionary of additional GeoJSON layers
    additional_layers = {
        "Roads": get_project_root() / "data/layers/road.geojson",
        "Educational Institutes": get_project_root()
        / "data/layers/educational-institute.geojson",
        "Tourist": get_project_root()
        / "data/layers/tourist.geojson",
    }

    # Add additional layers to the map
    for layer_name, layer_path in additional_layers.items():
        if layer_path.exists():
            try:
                with open(layer_path, "r") as f:
                    layer_data = json.load(f)

                folium.GeoJson(
                    layer_data,
                    name=layer_name,
                    style_function=lambda x: {
                        "fillColor": "#6baed6",
                        "color": "#08519c",
                        "weight": 0.5,
                        "fillOpacity": 0.3,
                    },
                ).add_to(base_map)
            except Exception as e:
                print(f"Error loading {layer_name} GeoJSON: {e}")
        else:
            print(f"Warning: {layer_name} GeoJSON file not found at {layer_path}")

    return base_map


def create_heatmap(base_map, data, name="Heatmap", radius=15):
    # Create a copy of the base map
    map_with_heatmap = base_map

    # Extract just the numeric data for the heatmap (Latitude, Longitude, and value)
    heatmap_data = data[["Latitude", "Longitude"]]
    if len(data.columns) > 2:
        # Add the third column (intensity value) if it exists
        value_col = [
            col
            for col in data.columns
            if col not in ["Latitude", "Longitude", "Name", "Location"]
        ][0]
        heatmap_data[value_col] = data[value_col]
        # Convert to numpy array of floats for heatmap
        heatmap_array = heatmap_data.values.astype(float)
    else:
        # If no value column, just use lat/lon
        heatmap_array = heatmap_data.values.astype(float)

    # Add heatmap layer with default colors
    HeatMap(
        data=heatmap_array,
        radius=radius,
        max_zoom=15,
        name=name,
        control=True,
        show=True,
    ).add_to(map_with_heatmap)

    # Create a marker cluster layer
    marker_cluster = MarkerCluster(name=f"{name} Markers").add_to(map_with_heatmap)

    # Add markers for each location with counts
    for idx, row in data.iterrows():
        # Assuming the data has Name and Count columns or can use the lat/lon as labels
        popup_text = ""
        if "Name" in data.columns and "Count" in data.columns:
            popup_text = f"{row['Name']}: {row['Count']} restaurants"
        elif "Location" in data.columns and len(data.columns) >= 3:
            value_col = [
                col
                for col in data.columns
                if col not in ["Latitude", "Longitude", "Location"]
            ][0]
            popup_text = f"{row['Location']}: {row[value_col]}"
        else:
            # Use the third column as the value to display (after lat/lon)
            value_col = data.columns[2] if len(data.columns) > 2 else None
            if value_col:
                popup_text = f"Value: {row[value_col]}"
            else:
                popup_text = (
                    f"Location: ({row['Latitude']:.4f}, {row['Longitude']:.4f})"
                )

        # Get tooltip from Name column if it exists, otherwise use location name if it exists
        tooltip_text = "Click for details"
        if "Name" in data.columns:
            tooltip_text = row["Name"]
        elif "Location" in data.columns:
            tooltip_text = row["Location"]

        folium.Marker(
            location=[row["Latitude"], row["Longitude"]],
            popup=popup_text,
            tooltip=tooltip_text,
        ).add_to(marker_cluster)

    # Add layer control
    folium.LayerControl().add_to(map_with_heatmap)

    return map_with_heatmap


def create_marker_cluster_map(base_map, locations_data):
    # Create marker cluster
    marker_cluster = MarkerCluster(name="Restaurant Clusters").add_to(base_map)

    # Add markers for each location
    for idx, row in locations_data.iterrows():
        folium.Marker(
            location=[row["Latitude"], row["Longitude"]],
            popup=f"{row['Name']}: {row['Count']} restaurants",
            tooltip=row["Name"],
        ).add_to(marker_cluster)

    # Add layer control
    folium.LayerControl().add_to(base_map)

    return base_map


def add_markers_to_map(map_obj, recommendations):
    # Create a FeatureGroup for restaurants
    restaurant_group = folium.FeatureGroup(name="Recommended Restaurants")

    # Add markers for each restaurant
    for idx, rest in recommendations.iterrows():
        # Create popup content
        popup_content = f"""
        <strong>{rest['name']}</strong><br>
        Rating: {rest['rating']}/5 ({rest['votes']} votes)<br>
        Cuisines: {rest['cuisines']}<br>
        Cost: ₹{rest['approx_cost']} for two<br>
        Distance: {rest['distance_km']:.2f} km
        """

        # Add marker
        folium.Marker(
            location=[rest["latitude"], rest["longitude"]],
            popup=folium.Popup(popup_content, max_width=300),
            tooltip=rest["name"],
            icon=folium.Icon(color="red", icon="cutlery", prefix="fa"),
        ).add_to(restaurant_group)

    # Add the feature group to the map
    restaurant_group.add_to(map_obj)

    # Add layer control if it doesn't exist
    if not any(
        isinstance(child, folium.LayerControl) for child in map_obj._children.values()
    ):
        folium.LayerControl().add_to(map_obj)

    return map_obj


def create_cuisine_heatmap(base_map, df_zomato, cuisine):
    # Filter for the selected cuisine
    df_cuisine = df_zomato[df_zomato["cuisines"] == cuisine]

    # Group by location and count
    df_cuisine = (
        df_cuisine.groupby("location (Processed)")["name"].agg("count").reset_index()
    )
    df_cuisine.columns = ["Name", "Count"]

    # Sort by count
    df_cuisine = df_cuisine.sort_values(by="Count", ascending=False)

    # Merge with location coordinates
    df_cuisine = df_cuisine.merge(
        right=df_zomato[
            ["location (Processed)", "latitude", "longitude"]
        ].drop_duplicates(),
        left_on="Name",
        right_on="location (Processed)",
        how="left",
    ).dropna()

    # Create the heatmap
    heatmap_data = df_cuisine[["latitude", "longitude", "Count"]]

    return create_heatmap(base_map, heatmap_data, name=f"{cuisine} Restaurant Density")


def create_folium_map(center_lat, center_lon, zoom_start=12):
    m = folium.Map(
        location=[center_lat, center_lon], zoom_start=zoom_start, tiles="OpenStreetMap"
    )

    return m
