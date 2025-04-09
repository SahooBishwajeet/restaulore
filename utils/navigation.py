import folium
import numpy as np
import pandas as pd
from math import radians
from sklearn.metrics.pairwise import haversine_distances
import requests


def generate_route_with_api(
    user_lat, user_lon, restaurant_lat, restaurant_lon, restaurant_name
):
    try:
        from polyline import decode
    except ImportError:
        print("Warning: 'polyline' package not installed. Using direct route only.")
        return generate_direct_route(
            user_lat, user_lon, restaurant_lat, restaurant_lon, restaurant_name
        )

    # Create base map
    center_lat = (user_lat + restaurant_lat) / 2
    center_lon = (user_lon + restaurant_lon) / 2
    route_map = folium.Map(location=[center_lat, center_lon], zoom_start=14)

    # Add markers for user location and restaurant
    folium.Marker(
        [user_lat, user_lon],
        popup="Your Location",
        tooltip="Your Location",
        icon=folium.Icon(color="blue", icon="user", prefix="fa"),
    ).add_to(route_map)

    folium.Marker(
        [restaurant_lat, restaurant_lon],
        popup=restaurant_name,
        tooltip=restaurant_name,
        icon=folium.Icon(color="red", icon="cutlery", prefix="fa"),
    ).add_to(route_map)

    # Try OSRM API first
    try:
        print("Attempting to get route from OSRM API...")
        url = f"http://router.project-osrm.org/route/v1/driving/{user_lon},{user_lat};{restaurant_lon},{restaurant_lat}?overview=full&geometries=polyline"
        response = requests.get(url, timeout=5)

        if response.status_code == 200:
            route_data = response.json()

            if route_data["code"] == "Ok" and len(route_data["routes"]) > 0:
                # Extract route geometry (in encoded polyline format)
                geometry = route_data["routes"][0]["geometry"]

                # Decode the polyline
                route_coords = decode(geometry)

                # Extract route information
                distance = route_data["routes"][0]["distance"] / 1000  # Convert to km
                duration = (
                    route_data["routes"][0]["duration"] / 60
                )  # Convert to minutes

                # Add route to map
                folium.PolyLine(
                    locations=route_coords,
                    color="green",
                    weight=5,
                    opacity=0.7,
                    popup=f"Distance: {distance:.2f} km, Duration: {duration:.1f} min",
                ).add_to(route_map)

                # Add info box
                info_html = f"""
                <div style="padding: 10px; background-color: white; border-radius: 5px; box-shadow: 0 0 10px rgba(0,0,0,0.1);">
                    <h4>Route Information</h4>
                    <b>Restaurant:</b> {restaurant_name}<br>
                    <b>Distance:</b> {distance:.2f} km<br>
                    <b>Est. travel time:</b> {duration:.1f} min<br>
                    <i>(Based on actual road network from OSRM)</i>
                </div>
                """

                folium.Marker(
                    [user_lat, user_lon],
                    popup=folium.Popup(info_html, max_width=300),
                    icon=folium.DivIcon(
                        html=""
                    ),  # Invisible marker, just for the popup
                ).add_to(route_map)

                print("Successfully generated route using OSRM API")
                return route_map
    except Exception as e:
        print(f"OSRM API error: {str(e)}")

    # Fallback to direct route
    print("Falling back to direct route...")
    return generate_direct_route(
        user_lat, user_lon, restaurant_lat, restaurant_lon, restaurant_name
    )


def generate_direct_route(
    user_lat, user_lon, restaurant_lat, restaurant_lon, restaurant_name
):
    # Create map
    center_lat = (user_lat + restaurant_lat) / 2
    center_lon = (user_lon + restaurant_lon) / 2
    route_map = folium.Map(location=[center_lat, center_lon], zoom_start=14)

    # Add markers for user location and restaurant
    folium.Marker(
        [user_lat, user_lon],
        popup="Your Location",
        tooltip="Your Location",
        icon=folium.Icon(color="blue", icon="user", prefix="fa"),
    ).add_to(route_map)

    folium.Marker(
        [restaurant_lat, restaurant_lon],
        popup=restaurant_name,
        tooltip=restaurant_name,
        icon=folium.Icon(color="red", icon="cutlery", prefix="fa"),
    ).add_to(route_map)

    # Calculate direct distance using Haversine formula
    user_coords = np.array([[radians(user_lat), radians(user_lon)]])
    rest_coords = np.array([[radians(restaurant_lat), radians(restaurant_lon)]])
    direct_distance_km = haversine_distances(user_coords, rest_coords)[0][0] * 6371

    # Draw direct line
    folium.PolyLine(
        locations=[[user_lat, user_lon], [restaurant_lat, restaurant_lon]],
        color="red",
        weight=5,
        opacity=0.7,
        popup=f"Direct distance: {direct_distance_km:.2f} km",
        dash_array="10,10",  # Dashed line to indicate it's not a real route
    ).add_to(route_map)

    # Add estimated time (assuming 20 km/h in urban traffic)
    estimated_time_min = direct_distance_km / 20 * 60

    # Add warning box about direct route
    info_html = f"""
    <div style="padding: 10px; background-color: white; border-radius: 5px; box-shadow: 0 0 10px rgba(0,0,0,0.1); border-left: 4px solid orange;">
        <h4>⚠️ Direct Route Only</h4>
        <b>Restaurant:</b> {restaurant_name}<br>
        <b>Direct distance:</b> {direct_distance_km:.2f} km<br>
        <b>Est. travel time:</b> {estimated_time_min:.1f} min<br>
        <i>Note: This is a straight-line distance, not following roads.<br>
        Actual travel distance and time will be longer.</i>
    </div>
    """

    folium.Marker(
        [user_lat, user_lon],
        popup=folium.Popup(info_html, max_width=300),
        icon=folium.DivIcon(html=""),  # Invisible marker, just for the popup
    ).add_to(route_map)

    return route_map
