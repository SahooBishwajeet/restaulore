import pandas as pd
import os
import numpy as np
from pathlib import Path


def get_project_root():
    current_file = Path(__file__)
    project_root = current_file.parent.parent
    return project_root


def load_census_data():
    project_root = get_project_root()
    file_path = project_root / "data/bangalore-ward-level-census-2011.csv"

    if os.path.exists(file_path):
        # Read the census data
        census_df = pd.read_csv(file_path)
        return census_df
    else:
        print(f"Warning: Census data file not found at {file_path}")
        return None


def load_processed_zomato_data():
    project_root = get_project_root()
    file_path = project_root / "data/zomato_processed_with_geo.csv"

    if not os.path.exists(file_path):
        print(f"Warning: Processed file not found at {file_path}")
        print("Falling back to raw data processing...")
        return load_and_preprocess_raw_data()

    return pd.read_csv(file_path)


def load_and_preprocess_raw_data():
    """Load and preprocess the raw Zomato data"""
    project_root = get_project_root()
    file_path = project_root / "data/zomato.csv"

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Zomato data file not found at {file_path}")

    df_zomato = pd.read_csv(file_path)

    # Process the data
    print("Processing raw Zomato data...")

    # Drop missing values from location
    df_zomato.dropna(axis="index", subset=["location"], inplace=True)

    # Location processing
    df_zomato["location (Processed)"] = df_zomato["location"].apply(
        lambda x: x.replace("Bangalore", "").strip()
    )
    df_zomato["location (Full)"] = df_zomato["location (Processed)"] + " Bangalore"

    # Process rating
    df_zomato.dropna(axis="index", subset=["rate"], inplace=True)
    df_zomato["rating"] = df_zomato["rate"].apply(
        lambda x: x.split("/")[0] if isinstance(x, str) else x
    )
    df_zomato["rating"] = df_zomato["rating"].replace("NEW", "0")
    df_zomato["rating"] = df_zomato["rating"].replace("-", "0")
    df_zomato["rating"] = pd.to_numeric(df_zomato["rating"], errors="coerce")

    # Clean cost data
    df_zomato["approx_cost"] = pd.to_numeric(
        df_zomato["approx_cost(for two people)"], errors="coerce"
    )

    # Process cuisines
    df_zomato["cuisines"] = df_zomato["cuisines"].fillna("")

    # Process votes - convert to numeric
    df_zomato["votes"] = pd.to_numeric(df_zomato["votes"], errors="coerce")
    df_zomato["votes"] = df_zomato["votes"].fillna(0).astype(int)

    return df_zomato


def load_restaurant_locations():
    project_root = get_project_root()
    file_path = project_root / "data/Restaurant_Locations.csv"

    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        print(f"Warning: Restaurant_Locations.csv not found at {file_path}")
        print("Generating restaurant locations data...")

        return None


def load_restaurant_ratings():
    project_root = get_project_root()
    file_path = project_root / "data/Restaurant_Ratings.csv"

    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        print(f"Warning: Restaurant_Ratings.csv not found at {file_path}")
        print("Generating restaurant ratings data...")

        return None


def load_locations_data():
    """Load locations data with coordinates"""
    project_root = get_project_root()
    file_path = project_root / "data/zomato_locations.csv"

    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        print(f"Warning: zomato_locations.csv not found at {file_path}")

        return None


def load_cuisine_data(cuisine_name=None):
    project_root = get_project_root()
    cuisines_dir = project_root / "data/cuisines"

    # Load cuisine counts
    cuisine_counts_path = cuisines_dir / "cuisine_counts.csv"
    if os.path.exists(cuisine_counts_path):
        cuisine_counts = pd.read_csv(cuisine_counts_path)
    else:
        print(f"Warning: cuisine_counts.csv not found at {cuisine_counts_path}")
        cuisine_counts = None

    # If no specific cuisine requested, return counts data
    if cuisine_name is None:
        return {"counts": cuisine_counts}

    # Load specific cuisine data
    cuisine_file = cuisines_dir / f"{cuisine_name}_restaurants.csv"
    if os.path.exists(cuisine_file):
        cuisine_data = pd.read_csv(cuisine_file)
        return cuisine_data
    else:
        print(f"Warning: {cuisine_name}_restaurants.csv not found at {cuisine_file}")
        return None


def load_region_data(region_name=None):
    project_root = get_project_root()
    regions_dir = project_root / "data/regions"

    # Load region metadata
    region_metadata_path = regions_dir / "region_metadata.csv"
    if os.path.exists(region_metadata_path):
        region_metadata = pd.read_csv(region_metadata_path)
    else:
        print(f"Warning: region_metadata.csv not found at {region_metadata_path}")
        # If metadata file doesn't exist, create list from directory contents
        if os.path.exists(regions_dir):
            region_files = [
                f for f in os.listdir(regions_dir) if f.endswith("_restaurants.csv")
            ]
            region_names = [f.replace("_restaurants.csv", "") for f in region_files]
            region_metadata = pd.DataFrame({"region": region_names})
        else:
            region_metadata = None

    # If no specific region requested, return metadata only
    if region_name is None:
        return {"metadata": region_metadata}

    # Load specific region data
    region_file = region_name.replace(" ", "_")
    region_file_path = regions_dir / f"{region_file}_restaurants.csv"
    if os.path.exists(region_file_path):
        region_data = pd.read_csv(region_file_path)
        return region_data
    else:
        print(f"Warning: {region_file}_restaurants.csv not found at {region_file_path}")
        return None


def get_precise_coordinates(location_name):
    """Get precise coordinates for a restaurant location from region files."""
    project_root = get_project_root()
    regions_dir = project_root / "data/regions"
    
    # Clean location name for file matching
    location_file = location_name.replace(" ", "_").lower()
    region_file_path = regions_dir / f"{location_file}_restaurants.csv"
    
    if os.path.exists(region_file_path):
        region_df = pd.read_csv(region_file_path)
        if "latitude" in region_df.columns and "longitude" in region_df.columns:
            return region_df[["latitude", "longitude"]]
    return None


def load_all_data():
    df_zomato = load_processed_zomato_data()
    locations_df = load_locations_data()
    restaurant_locations = load_restaurant_locations()
    restaurant_ratings = load_restaurant_ratings()
    cuisine_data = load_cuisine_data()
    region_data = load_region_data()

    return (
        df_zomato,
        locations_df,
        restaurant_locations,
        restaurant_ratings,
        cuisine_data,
        region_data,
    )
