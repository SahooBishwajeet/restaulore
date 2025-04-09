import pandas as pd
import numpy as np
from math import radians
from sklearn.metrics.pairwise import haversine_distances
from pathlib import Path


def get_precise_location_from_regions(
    restaurant_name, location_processed, df_original=None
):
    """
    Get precise location coordinates from region files if available

    Args:
        restaurant_name: Name of the restaurant to look up
        location_processed: Processed location name to determine which region file to check
        df_original: Original dataframe containing the restaurant record

    Returns:
        tuple: (latitude, longitude) if found, otherwise (None, None)
    """
    # Import here to avoid circular imports
    from utils.data_loader import get_project_root, load_region_data

    try:
        # Try to find region file based on location
        if location_processed:
            region_name = location_processed.strip()
            region_data = load_region_data(region_name)

            if region_data is not None and isinstance(region_data, pd.DataFrame):
                # Check if restaurant exists in region data
                restaurant_records = region_data[
                    region_data["name"].str.contains(
                        restaurant_name, case=False, na=False
                    )
                ]
                if not restaurant_records.empty:
                    # Get first match coordinates
                    record = restaurant_records.iloc[0]
                    if "latitude" in record and "longitude" in record:
                        return record["latitude"], record["longitude"]

        # Return original coordinates if available in the original dataframe
        if df_original is not None:
            restaurant_record = df_original[df_original["name"] == restaurant_name]
            if (
                not restaurant_record.empty
                and "latitude" in restaurant_record
                and "longitude" in restaurant_record
            ):
                return (
                    restaurant_record.iloc[0]["latitude"],
                    restaurant_record.iloc[0]["longitude"],
                )

    except Exception as e:
        print(f"Error retrieving precise location: {str(e)}")

    return None, None


def recommend_restaurants(
    user_lat,
    user_lon,
    cuisine_preference=None,
    min_rating=3.5,
    min_votes=50,
    max_distance=5,
    top_n=5,
    df=None,
):
    print(f"Finding restaurants near ({user_lat}, {user_lon})...")

    # Ensure we have data
    if df is None or df.empty:
        print("Error: No restaurant data available!")
        return pd.DataFrame()

    # Make a copy to avoid modifying the original
    working_df = df.copy()

    # Filter restaurants with valid coordinates
    working_df = working_df.dropna(subset=["latitude", "longitude"])
    print(f"Found {working_df.shape[0]} restaurants with valid coordinates")

    # Filter by minimum rating and votes
    filtered_df = working_df[
        (working_df["rating"] >= min_rating) & (working_df["votes"] >= min_votes)
    ]
    print(f"After rating/votes filter: {filtered_df.shape[0]} restaurants")

    # Filter by cuisine if specified
    if cuisine_preference:
        # Handle multiple cuisine preferences (comma-separated)
        if isinstance(cuisine_preference, str) and "," in cuisine_preference:
            cuisine_list = [c.strip() for c in cuisine_preference.split(",")]
            cuisine_mask = filtered_df["cuisines"].str.contains(
                "|".join(cuisine_list), case=False, na=False
            )
            filtered_df = filtered_df[cuisine_mask]
        else:
            filtered_df = filtered_df[
                filtered_df["cuisines"].str.contains(
                    cuisine_preference, case=False, na=False
                )
            ]
        print(f"After cuisine filter: {filtered_df.shape[0]} restaurants")

    if filtered_df.empty:
        print("No restaurants match the criteria! Relaxing constraints...")
        # Try with more relaxed criteria
        filtered_df = working_df[
            (working_df["rating"] >= min_rating - 0.5)
            & (working_df["votes"] >= min_votes // 2)
        ]
        print(f"After relaxed rating/votes filter: {filtered_df.shape[0]} restaurants")

        if cuisine_preference and not filtered_df.empty:
            if isinstance(cuisine_preference, str) and "," in cuisine_preference:
                cuisine_list = [c.strip() for c in cuisine_preference.split(",")]
                cuisine_mask = filtered_df["cuisines"].str.contains(
                    "|".join(cuisine_list), case=False, na=False
                )
                filtered_df = filtered_df[cuisine_mask]
            else:
                filtered_df = filtered_df[
                    filtered_df["cuisines"].str.contains(
                        cuisine_preference, case=False, na=False
                    )
                ]
            print(f"After relaxed cuisine filter: {filtered_df.shape[0]} restaurants")

    if filtered_df.empty:
        print("No restaurants match even the relaxed criteria!")
        return pd.DataFrame()

    # Calculate distance from user location
    user_coords = np.array([[radians(user_lat), radians(user_lon)]])

    # Convert restaurant coordinates to radians
    rest_coords = np.array(
        [
            [radians(lat), radians(lon)]
            for lat, lon in zip(filtered_df["latitude"], filtered_df["longitude"])
        ]
    )

    # Calculate distances
    distances = (
        haversine_distances(user_coords, rest_coords) * 6371
    )  # Earth radius in km
    filtered_df["distance_km"] = distances.flatten()

    # Filter by maximum distance
    distance_filtered = filtered_df[filtered_df["distance_km"] <= max_distance]
    print(
        f"After distance filter: {distance_filtered.shape[0]} restaurants within {max_distance}km"
    )

    if distance_filtered.empty:
        print(
            f"No restaurants within {max_distance}km. Trying with increased distance..."
        )
        distance_filtered = filtered_df[
            filtered_df["distance_km"] <= max_distance * 1.5
        ]
        print(
            f"After increased distance filter: {distance_filtered.shape[0]} restaurants within {max_distance * 1.5}km"
        )

        if distance_filtered.empty:
            print("No restaurants within reasonable distance!")
            return pd.DataFrame()

    # Create a weighted score for ranking
    max_votes = distance_filtered["votes"].max()
    max_distance_value = distance_filtered["distance_km"].max()

    distance_filtered["score"] = (
        distance_filtered["rating"] / 5 * 0.5  # 50% weight for rating
        + distance_filtered["votes"]
        / max(1, max_votes)
        * 0.3  # 30% weight for popularity
        + (1 - distance_filtered["distance_km"] / max(1, max_distance_value))
        * 0.2  # 20% weight for proximity
    )

    # Ensure we're not getting duplicate restaurants
    # Create a unique identifier based on name and location
    distance_filtered["unique_id"] = (
        distance_filtered["name"]
        + "_"
        + distance_filtered["latitude"].astype(str)
        + "_"
        + distance_filtered["longitude"].astype(str)
    )

    # Get top N recommendations with unique restaurants
    unique_filtered = distance_filtered.drop_duplicates(subset=["unique_id"])
    recommendations = unique_filtered.sort_values(by="score", ascending=False).head(
        top_n
    )

    # Check for more precise location data for plotting and navigation
    if "location (Processed)" in recommendations.columns:
        # Create copies of latitude and longitude columns to preserve original values
        recommendations["original_latitude"] = recommendations["latitude"].copy()
        recommendations["original_longitude"] = recommendations["longitude"].copy()

        # Try to get precise locations from region files for each restaurant
        for idx, row in recommendations.iterrows():
            lat, lon = get_precise_location_from_regions(
                row["name"], row["location (Processed)"], df_original=df
            )

            # Update only if precise location found
            if lat is not None and lon is not None:
                recommendations.at[idx, "latitude"] = lat
                recommendations.at[idx, "longitude"] = lon

    # Select and reorder columns for display
    result = recommendations[
        [
            "name",
            "address",
            "cuisines",
            "rating",
            "votes",
            "approx_cost",
            "latitude",
            "longitude",
            "distance_km",
            "score",
        ]
    ]

    print(f"Found {len(result)} unique recommendations!")

    return result
