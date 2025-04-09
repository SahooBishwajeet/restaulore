import streamlit as st
import folium
import pandas as pd
import numpy as np
from streamlit_folium import folium_static
import os
import sys
import time
from folium.plugins import HeatMap, MarkerCluster

# Add the app directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utility modules
from utils.data_loader import load_all_data, load_cuisine_data, load_region_data
from utils.mapping import (
    generate_base_map,
    create_heatmap,
    create_marker_cluster_map,
    add_markers_to_map,
    create_folium_map,
)
from utils.recommendation import recommend_restaurants
from utils.navigation import generate_route_with_api
from utils.business_intelligence import (
    analyze_restaurant_feasibility,
    generate_feasibility_map,
)

from config.settings import *

# Page configuration
st.set_page_config(
    page_title="Bangalore Restaurant GIS",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state for persistence
if "recommendations" not in st.session_state:
    st.session_state.recommendations = None

if "user_coords" not in st.session_state:
    st.session_state.user_coords = [
        DEFAULT_LATITUDE,
        DEFAULT_LONGITUDE,
    ]  # Default: Bangalore center

if "selected_restaurant" not in st.session_state:
    st.session_state.selected_restaurant = None

if "nav_map" not in st.session_state:
    st.session_state.nav_map = None

if "current_view" not in st.session_state:
    st.session_state.current_view = "Restaurant Recommendations & Navigation"

if "map" not in st.session_state:
    st.session_state.map = None

if "optimal_locations_map" not in st.session_state:
    st.session_state.optimal_locations_map = None

# Load all data
(
    df_zomato,
    locations_df,
    restaurant_locations,
    restaurant_ratings,
    cuisine_data,
    region_data,
) = load_all_data()

# Title and introduction
st.title("🍽️ Bangalore Restaurant GIS")
st.markdown(
    """
Find the best restaurants in Bangalore based on your preferences, location, and favorite cuisines.
This interactive application helps you discover top-rated restaurants and navigate to them.
"""
)

# Sidebar
st.sidebar.title("Options")

# View selection
view_options = [
    "Restaurant Recommendations & Navigation",
    "Restaurant Distribution Heatmap",
    "Restaurant Ratings Heatmap",
    "Cuisine Distribution Heatmap",
    "Business Intelligence - Restaurant Feasibility",
    "Business Intelligence - Optimal Location Finder",  # Add the new BI option
]

selected_view = st.sidebar.selectbox(
    "Select View",
    options=view_options,
    index=(
        view_options.index(st.session_state.current_view)
        if st.session_state.current_view in view_options
        else 0
    ),
    key="view_selector",
)

# Update the current view in session state
st.session_state.current_view = selected_view

# Add user preferences to sidebar when in recommendation view
if selected_view == "Restaurant Recommendations & Navigation":
    st.sidebar.subheader("Your Preferences")

    # Get user location
    location_options = ["Custom Location"] + sorted(
        locations_df["location_name"].unique().tolist()
    )
    location_choice = st.sidebar.selectbox("Choose Location", options=location_options)

    if location_choice == "Custom Location":
        col1, col2 = st.sidebar.columns(2)
        user_lat = col1.number_input(
            "Latitude", value=DEFAULT_LATITUDE, format="%.6f", step=0.001
        )
        user_lon = col2.number_input(
            "Longitude", value=DEFAULT_LONGITUDE, format="%.6f", step=0.001
        )
    else:
        selected_location = locations_df[
            locations_df["location_name"] == location_choice
        ].iloc[0]
        user_lat = selected_location["latitude"]
        user_lon = selected_location["longitude"]

    # Cuisine preference
    all_cuisines_option = "All Cuisines"

    # Add cuisine list selector
    cuisine_list_type = st.sidebar.radio(
        "Cuisine List Type",
        options=["Popular Cuisines", "All Available Cuisines"],
        index=0,
    )

    # Set cuisine options based on selection
    if cuisine_list_type == "Popular Cuisines":
        cuisine_options = [all_cuisines_option] + TOP_CUISINES
    else:
        cuisine_options = [all_cuisines_option] + sorted(list(set(ALL_CUISINES)))

    cuisine_choice = st.sidebar.selectbox("Cuisine Preference", options=cuisine_options)
    cuisine_preference = (
        None if cuisine_choice == all_cuisines_option else cuisine_choice
    )

    # Additional filters
    st.sidebar.subheader("Additional Filters")
    min_rating = st.sidebar.slider("Minimum Rating", 1.0, 5.0, 3.5, 0.1)
    min_votes = st.sidebar.slider("Minimum Votes", 100, 5000, 500, 50)
    max_distance = st.sidebar.slider("Maximum Distance (km)", 1, 30, 5, 1)
    top_n = st.sidebar.slider("Number of Recommendations", 3, 30, 5, 1)

    # Generate recommendations button
    if st.sidebar.button("Find Restaurants"):
        with st.spinner("Finding the best restaurants for you..."):
            recommendations = recommend_restaurants(
                user_lat=user_lat,
                user_lon=user_lon,
                cuisine_preference=cuisine_preference,
                min_rating=min_rating,
                min_votes=min_votes,
                max_distance=max_distance,
                top_n=top_n,
                df=df_zomato,
            )
            st.session_state.recommendations = recommendations

# Business Intelligence - Restaurant Feasibility view
elif selected_view == "Business Intelligence - Restaurant Feasibility":
    st.header("Restaurant Feasibility Analysis")
    st.markdown(
        """
    Analyze the feasibility of opening a restaurant at a specific location in Bangalore.
    This tool helps restaurant owners evaluate potential locations based on:
    - Local competition analysis
    - Population demographics
    - Market potential
    - Investment returns
    """
    )

    st.sidebar.subheader("Location Selection")

    # Location selection options
    loc_selection_method = st.sidebar.radio(
        "Select location by:",
        options=["Map", "Coordinates", "Existing Area"],
        index=0,
    )

    # Initialize location variables
    location_lat = DEFAULT_LATITUDE
    location_lon = DEFAULT_LONGITUDE
    location_name = None

    if loc_selection_method == "Map":
        # Show a map where user can click to select location
        if "selected_location" not in st.session_state:
            st.session_state.selected_location = [DEFAULT_LATITUDE, DEFAULT_LONGITUDE]

        # Create a simple map for location selection
        map_selection = folium.Map(
            location=[DEFAULT_LATITUDE, DEFAULT_LONGITUDE], zoom_start=12
        )

        # Add a marker for the currently selected location
        folium.Marker(
            location=st.session_state.selected_location,
            popup="Selected Location",
            icon=folium.Icon(color="green", icon="info-sign"),
        ).add_to(map_selection)

        folium_static(map_selection)

        # Provide text fields for manual coordinate input alongside the map
        col1, col2 = st.columns(2)
        with col1:
            location_lat = st.number_input(
                "Latitude", value=st.session_state.selected_location[0], format="%.6f"
            )
        with col2:
            location_lon = st.number_input(
                "Longitude", value=st.session_state.selected_location[1], format="%.6f"
            )

        st.session_state.selected_location = [location_lat, location_lon]

    elif loc_selection_method == "Coordinates":
        # Direct coordinate input
        col1, col2 = st.columns(2)
        with col1:
            location_lat = st.number_input(
                "Latitude", value=DEFAULT_LATITUDE, format="%.6f"
            )
        with col2:
            location_lon = st.number_input(
                "Longitude", value=DEFAULT_LONGITUDE, format="%.6f"
            )

    elif loc_selection_method == "Existing Area":
        # Choose from existing locations
        if locations_df is not None:
            location_options = sorted(locations_df["location_name"].unique().tolist())
            selected_area = st.sidebar.selectbox(
                "Select Area", options=location_options
            )

            selected_location_data = locations_df[
                locations_df["location_name"] == selected_area
            ].iloc[0]
            location_lat = selected_location_data["latitude"]
            location_lon = selected_location_data["longitude"]
            location_name = selected_area
        else:
            st.error(
                "Location data not available. Please use coordinate input instead."
            )

    st.sidebar.subheader("Restaurant Parameters")

    # Cuisine selection with multi-select
    all_cuisines_option = "All Cuisines"
    cuisine_list_type = st.sidebar.radio(
        "Cuisine List Type",
        options=["Popular Cuisines", "All Available Cuisines"],
        index=0,
    )

    if cuisine_list_type == "Popular Cuisines":
        cuisine_options = [all_cuisines_option] + TOP_CUISINES
    else:
        cuisine_options = [all_cuisines_option] + sorted(list(set(ALL_CUISINES)))

    selected_cuisines = st.sidebar.multiselect(
        "Select Cuisines",
        options=cuisine_options,
        default=[all_cuisines_option],
    )

    # Handle 'All Cuisines' selection
    if all_cuisines_option in selected_cuisines:
        selected_cuisines = None

    radius_km = st.sidebar.slider("Analysis Radius (km)", 0.5, 5.0, 2.0, 0.5)

    # Optional parameters
    with st.sidebar.expander("Advanced Parameters"):
        target_customers = st.number_input(
            "Target Daily Customers",
            min_value=0,
            max_value=1000,
            value=0,
            step=10,
            help="Your target number of customers per day (optional)",
        )
        target_customers = None if target_customers == 0 else target_customers

        avg_price_per_person = st.number_input(
            "Average Price Per Person (₹)",
            min_value=0,
            max_value=5000,
            value=0,
            step=50,
            help="Average amount each person will spend (optional)",
        )
        avg_price_per_person = (
            None if avg_price_per_person == 0 else avg_price_per_person
        )

        investment = st.number_input(
            "Investment Budget (Lakhs ₹)",
            min_value=0.0,
            max_value=1000.0,
            value=0.0,
            step=5.0,
            help="Your investment budget in lakhs of rupees (optional)",
        )
        investment = None if investment == 0.0 else investment

    # Run analysis button
    if st.sidebar.button("Run Feasibility Analysis"):
        with st.spinner("Analyzing restaurant feasibility..."):
            # Run the feasibility analysis
            feasibility_result = analyze_restaurant_feasibility(
                (location_lat, location_lon),
                cuisine_type=selected_cuisines,
                target_customers=target_customers,
                radius_km=radius_km,
                investment=investment,
                avg_price_per_person=avg_price_per_person,
            )

            # Store the result in the session state
            st.session_state.feasibility_result = feasibility_result

            # Generate the map for visualization
            st.session_state.feasibility_map = generate_feasibility_map(
                (location_lat, location_lon),
                radius_km=radius_km,
                cuisine_type=selected_cuisines,
            )

    # Display results if available
    if "feasibility_result" in st.session_state:
        result = st.session_state.feasibility_result

        # Display the map
        st.subheader("Analysis Area")
        folium_static(st.session_state.feasibility_map)

        # Display feasibility score prominently
        st.subheader("Feasibility Score")
        score = result["feasibility_score"]

        # Color the score based on value
        color = "red"
        if score >= 80:
            color = "green"
        elif score >= 60:
            color = "lightgreen"
        elif score >= 40:
            color = "orange"

        st.markdown(
            f"""
            <div style="background-color: {color}; padding: 20px; border-radius: 10px;">
                <h1 style="text-align: center; color: white;">{score:.1f}/100</h1>
            </div>
            <p style="text-align: center; font-size: 18px; margin-top: 10px;">
                <b>{result["recommendation"]}</b>
            </p>
            """,
            unsafe_allow_html=True,
        )

        # Display detailed metrics
        st.subheader("Detailed Analysis")

        # Create 3 columns for metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### Competition")
            st.metric(
                label="Nearby Restaurants",
                value=result["competition_metrics"]["total_restaurants"],
            )
            if selected_cuisines:
                st.metric(
                    label=f"{selected_cuisines} Restaurants",
                    value=result["competition_metrics"]["cuisine_restaurants"],
                )
            st.metric(
                label="Restaurant Density",
                value=f"{result['competition_metrics']['restaurant_density']:.1f}/km²",
            )
            st.metric(
                label="Average Rating",
                value=f"{result['competition_metrics']['avg_rating']:.1f}/5",
            )

        with col2:
            st.markdown("### Demographics")
            st.metric(
                label="Estimated Population",
                value=f"{result['population_metrics']['estimated_population']:,}",
            )
            st.metric(
                label="Population Density",
                value=f"{result['population_metrics']['population_density']}/km²",
            )
            st.metric(
                label="Spending Power Index",
                value=f"{result['population_metrics']['spending_power_index']}/100",
            )

        with col3:
            st.markdown("### Market Potential")
            st.metric(
                label="Est. Daily Customers",
                value=f"{result['market_potential']['estimated_daily_customers']:,}",
            )
            st.metric(
                label="Est. Daily Revenue",
                value=f"₹{result['market_potential']['estimated_daily_revenue']:,}",
            )
            st.metric(
                label="Est. Monthly Revenue",
                value=f"₹{result['market_potential']['estimated_monthly_revenue']:,}",
            )

        # If investment data is provided, show ROI information
        if "investment_metrics" in result:
            st.subheader("Investment Analysis")
            inv_col1, inv_col2 = st.columns(2)

            with inv_col1:
                st.metric(
                    label="Monthly Profit",
                    value=f"₹{result['investment_metrics']['monthly_profit']:,}",
                )
                st.metric(
                    label="Annual Profit",
                    value=f"₹{result['investment_metrics']['annual_profit']:,}",
                )

            with inv_col2:
                st.metric(
                    label="Annual ROI",
                    value=f"{result['investment_metrics']['annual_roi_percentage']:.1f}%",
                )
                st.metric(
                    label="Payback Period",
                    value=f"{result['investment_metrics']['payback_period_months']:.1f} months",
                )

        # Add explanation of the analysis
        with st.expander("How this analysis works"):
            st.markdown(
                """
            This feasibility analysis evaluates four key factors:

            1. **Competition**: Analyzes nearby restaurants, their density, and ratings to assess market saturation.

            2. **Demographics**: Estimates the population and spending power in the vicinity to determine potential customer base.

            3. **Market Potential**: Calculates projected customers and revenue based on competition and demographics.

            4. **Investment Return**: If investment data is provided, calculates ROI and payback period.

            The overall feasibility score is a weighted average of these factors, with competition and market potential having the highest weights.

            **Note**: This analysis is based on available data and mathematical models. Use it as one of many inputs in your business decision-making process.
            """
            )

# Business Intelligence - Optimal Location Finder view
elif selected_view == "Business Intelligence - Optimal Location Finder":
    st.header("Optimal Restaurant Location Finder")
    st.markdown(
        """
    Find the best locations in Bangalore to open a restaurant based on your business requirements.
    This tool analyzes various factors to recommend optimal locations:
    - Local competition analysis
    - Population demographics
    - Market potential
    - Investment returns
    """
    )

    st.sidebar.subheader("Restaurant Parameters")

    # Cuisine selection with multi-select
    all_cuisines_option = "All Cuisines"
    cuisine_list_type = st.sidebar.radio(
        "Cuisine List Type",
        options=["Popular Cuisines", "All Available Cuisines"],
        index=0,
    )

    if cuisine_list_type == "Popular Cuisines":
        cuisine_options = [all_cuisines_option] + TOP_CUISINES
    else:
        cuisine_options = [all_cuisines_option] + sorted(list(set(ALL_CUISINES)))

    selected_cuisines = st.sidebar.multiselect(
        "Select Cuisines",
        options=cuisine_options,
        default=[all_cuisines_option],
    )

    # Handle 'All Cuisines' selection
    if all_cuisines_option in selected_cuisines:
        selected_cuisines = None

    # Optional parameters
    with st.sidebar.expander("Business Requirements"):
        target_customers = st.number_input(
            "Target Daily Customers",
            min_value=0,
            max_value=1000,
            value=0,
            step=10,
            help="Your target number of customers per day (optional)",
        )
        target_customers = None if target_customers == 0 else target_customers

        investment = st.number_input(
            "Investment Budget (Lakhs ₹)",
            min_value=0.0,
            max_value=1000.0,
            value=0.0,
            step=5.0,
            help="Your investment budget in lakhs of rupees (optional)",
        )
        investment = None if investment == 0.0 else investment

        avg_price_per_person = st.number_input(
            "Average Price Per Person (₹)",
            min_value=0,
            max_value=5000,
            value=0,
            step=50,
            help="Average amount each person will spend (optional)",
        )
        avg_price_per_person = (
            None if avg_price_per_person == 0 else avg_price_per_person
        )

    # Number of locations to recommend
    max_locations = st.sidebar.slider(
        "Number of locations to recommend", min_value=3, max_value=15, value=5, step=1
    )

    # Flag to track if analysis has been run
    has_results = False
    optimal_locations = None

    # Find locations button
    if st.sidebar.button("Find Optimal Locations"):
        with st.spinner("Analyzing locations across Bangalore..."):
            # This analysis can take time as it evaluates multiple locations
            from utils.business_intelligence import (
                find_optimal_locations,
                generate_optimal_locations_map,
            )

            # Run the location finder algorithm
            optimal_locations = find_optimal_locations(
                cuisine_type=selected_cuisines,
                target_customers=target_customers,
                investment=investment,
                max_locations=max_locations,
                avg_price_per_person=avg_price_per_person,
            )
            has_results = True

            # Generate the map for visualization and store it in session state
            st.session_state.optimal_locations_map = generate_optimal_locations_map(
                optimal_locations, cuisine_type=selected_cuisines
            )

    # Display results if available
    if optimal_locations:
        # Display the map
        st.subheader("Recommended Locations")

        # Display the map from session state
        folium_static(st.session_state.optimal_locations_map)

        # Display location details in a table
        st.subheader("Location Details")

        # Convert to DataFrame for display
        location_df = pd.DataFrame(optimal_locations)

        # Format scores to 1 decimal place
        location_df["feasibility_score"] = location_df["feasibility_score"].map(
            lambda x: f"{x:.1f}/100"
        )
        if "annual_roi_percentage" in location_df.columns:
            location_df["annual_roi_percentage"] = location_df[
                "annual_roi_percentage"
            ].map(lambda x: f"{x:.1f}%")
            location_df["payback_period_months"] = location_df[
                "payback_period_months"
            ].map(lambda x: f"{x:.1f} months")

        # Format revenue with commas
        location_df["estimated_monthly_revenue"] = location_df[
            "estimated_monthly_revenue"
        ].map(lambda x: f"₹{x:,}")

        # Select and rename columns for display
        display_cols = [
            "location_name",
            "feasibility_score",
            "estimated_daily_customers",
            "estimated_monthly_revenue",
        ]

        # Add ROI columns if available
        if "annual_roi_percentage" in location_df.columns:
            display_cols.extend(["annual_roi_percentage", "payback_period_months"])

        # Add competition columns
        display_cols.extend(["nearby_restaurants"])
        if selected_cuisines:
            display_cols.append("cuisine_restaurants")

        # Display the table
        display_df = location_df[display_cols].copy()

        # Rename columns for better readability
        column_names = {
            "location_name": "Location",
            "feasibility_score": "Feasibility Score",
            "estimated_daily_customers": "Est. Daily Customers",
            "estimated_monthly_revenue": "Est. Monthly Revenue",
            "annual_roi_percentage": "Annual ROI",
            "payback_period_months": "Payback Period",
            "nearby_restaurants": "Nearby Restaurants",
            "cuisine_restaurants": f"Nearby {selected_cuisines} Restaurants",
        }
        display_df.rename(columns=column_names, inplace=True)

        # Display table
        st.dataframe(display_df, use_container_width=True)

        # Location comparison
        st.subheader("Location Comparison")

        # Create a bar chart to compare feasibility scores
        scores_chart_data = location_df[["location_name", "feasibility_score"]]
        scores_chart_data["feasibility_score"] = pd.to_numeric(
            scores_chart_data["feasibility_score"].str.replace("/100", "")
        )

        bar_chart = st.bar_chart(
            data=scores_chart_data.set_index("location_name"), height=400
        )

        # Explanation of the results
        with st.expander("How these locations were selected"):
            st.markdown(
                """
            The optimal locations were selected based on a comprehensive analysis of:

            1. **Competition Analysis**: Evaluating the density and ratings of nearby restaurants, with special attention to restaurants serving similar cuisine.

            2. **Population Demographics**: Analyzing local population density and estimated spending power.

            3. **Market Potential**: Projecting the number of potential customers and expected revenue based on location-specific factors.

            4. **Investment Returns**: Calculating ROI and payback periods based on your specified investment amount.

            Locations with the highest overall feasibility scores were selected as the optimal choices for your restaurant.

            **Note**: For each location, we analyze various factors within a 1.5km radius to determine its potential for your specific restaurant type.
            """
            )
    else:
        if selected_view == "Business Intelligence - Optimal Location Finder":
            st.info(
                "Click 'Find Optimal Locations' to analyze the best areas for your restaurant."
            )

if selected_view == "Restaurant Distribution Heatmap":
    st.header("Restaurant Distribution in Bangalore")
    st.markdown(
        """
    This heatmap shows the density of restaurants across different areas of Bangalore.
    Areas with higher restaurant density appear brighter on the map.
    """
    )

    # Add region selection to sidebar
    st.sidebar.subheader("Region Selection")

    # Option to show all regions or select a specific one
    region_selection_type = st.sidebar.radio(
        "View Mode",
        options=["All Regions", "Select Specific Region"],
        index=0,
    )

    # Add option for number of areas to display in table
    num_areas = st.sidebar.slider(
        "Number of areas to display", min_value=5, max_value=30, value=10, step=5
    )

    # Create base map
    base_map = generate_base_map()

    if region_selection_type == "All Regions":
        # Create heatmap with all data
        map_with_heatmap = create_heatmap(
            base_map,
            restaurant_locations[["Latitude", "Longitude", "Count", "Name"]],
            "Restaurant Density",
        )

        # Additional analysis
        st.subheader(f"Top {num_areas} Areas by Restaurant Count")
        top_areas = restaurant_locations.sort_values(by="Count", ascending=False).head(
            num_areas
        )
        st.dataframe(top_areas[["Name", "Count"]])

        # Add statistics
        total_restaurants = restaurant_locations["Count"].sum()
        total_areas = len(restaurant_locations)
        avg_per_area = total_restaurants / total_areas if total_areas > 0 else 0
    else:
        # Get list of regions from metadata
        if (
            region_data
            and "metadata" in region_data
            and region_data["metadata"] is not None
        ):
            region_metadata = region_data["metadata"]

            # Sort regions by restaurant_count if available, otherwise by name
            if "restaurant_count" in region_metadata.columns:
                sorted_regions = region_metadata.sort_values(
                    by="restaurant_count", ascending=False
                )
                region_options = sorted_regions["region"].tolist()
            else:
                region_options = sorted(region_metadata["region"].tolist())

            selected_region = st.sidebar.selectbox(
                "Select a region",
                options=region_options,
            )

            # Load region-specific data
            with st.spinner(f"Loading {selected_region} restaurant data..."):
                region_restaurants_df = load_region_data(selected_region)

            if region_restaurants_df is not None:
                st.markdown(f"### Restaurant Distribution in {selected_region}")

                # Add cuisine filter option
                if "cuisines" in region_restaurants_df.columns:
                    # Extract unique cuisines from the region dataset
                    all_cuisines = []
                    for cuisines_str in region_restaurants_df["cuisines"].dropna():
                        if isinstance(cuisines_str, str):
                            all_cuisines.extend(
                                [c.strip() for c in cuisines_str.split(",")]
                            )

                    unique_cuisines = sorted(list(set(all_cuisines)))

                    # Add cuisine filter options to sidebar with multi-select
                    st.sidebar.subheader("Cuisine Filter")

                    # Add "All Cuisines" option as the first choice in the list
                    all_cuisines_option = "All Cuisines"
                    cuisine_options = [all_cuisines_option] + unique_cuisines

                    selected_cuisines = st.sidebar.multiselect(
                        "Select cuisines to filter",
                        options=cuisine_options,
                        default=[all_cuisines_option],
                    )

                    # Logic to handle "All Cuisines" selection
                    if all_cuisines_option in selected_cuisines:
                        if (
                            len(selected_cuisines) > 1
                        ):  # User selected All and other cuisines
                            # Just use "All Cuisines" and ignore other selections
                            st.sidebar.info(
                                "'All Cuisines' is selected. Other selections will be ignored."
                            )
                            selected_cuisines = [all_cuisines_option]
                        filtered_df = region_restaurants_df  # Show all cuisines
                    else:
                        if selected_cuisines:  # User selected specific cuisines
                            filtered_df = region_restaurants_df[
                                region_restaurants_df["cuisines"].apply(
                                    lambda x: (
                                        any(
                                            cuisine in x
                                            for cuisine in selected_cuisines
                                        )
                                        if isinstance(x, str)
                                        else False
                                    )
                                )
                            ]
                        else:  # No cuisine selected, default to all
                            selected_cuisines = [all_cuisines_option]
                            filtered_df = region_restaurants_df
                else:
                    filtered_df = region_restaurants_df

                # Check if the dataframe has the necessary location columns
                if (
                    "latitude" in filtered_df.columns
                    and "longitude" in filtered_df.columns
                ):
                    # Group by location if not already aggregated
                    if "Count" not in filtered_df.columns:
                        region_agg = (
                            filtered_df.groupby(["latitude", "longitude"])
                            .size()
                            .reset_index(name="Count")
                        )
                        if "name" in filtered_df.columns:
                            # Get a representative name for each location group
                            name_groups = (
                                filtered_df.groupby(["latitude", "longitude"])["name"]
                                .first()
                                .reset_index()
                            )
                            region_agg = region_agg.merge(
                                name_groups, on=["latitude", "longitude"]
                            )
                        else:
                            region_agg["name"] = selected_region

                        heatmap_data = region_agg.rename(
                            columns={
                                "name": "Name",
                                "latitude": "Latitude",
                                "longitude": "Longitude",
                                "Count": "Count",
                            }
                        )
                    else:
                        # Data is already aggregated
                        heatmap_data = filtered_df.rename(
                            columns={
                                "Name": "Name",
                                "Latitude": "Latitude",
                                "Longitude": "Longitude",
                                "Count": "Count",
                            }
                        )

                    # Show filter stats if filtering is applied
                    if (
                        all_cuisines_option not in selected_cuisines
                        and selected_cuisines
                    ):
                        filter_text = ", ".join(selected_cuisines)
                        st.info(
                            f"Showing {len(filtered_df)} restaurants matching cuisine filter: {filter_text}"
                        )

                    # Create region-specific heatmap
                    region_map = create_heatmap(
                        base_map,
                        heatmap_data[["Latitude", "Longitude", "Count", "Name"]],
                        f"{selected_region} Restaurant Density",
                    )

                    # Display the map
                    folium_static(region_map)

                    # Show top restaurants in this region if individual restaurant data
                    if (
                        "rating" in filtered_df.columns
                        and "name" in filtered_df.columns
                    ):
                        st.subheader(
                            f"Top {num_areas} Restaurants in {selected_region}"
                        )
                        top_restaurants = filtered_df.sort_values(
                            by="rating", ascending=False
                        ).head(num_areas)
                        display_columns = ["name", "rating", "cuisines", "approx_cost"]
                        display_columns = [
                            col
                            for col in display_columns
                            if col in top_restaurants.columns
                        ]
                        st.dataframe(top_restaurants[display_columns])

                    # Add statistics for the selected region
                    if "Count" in heatmap_data.columns:
                        total_restaurants = heatmap_data["Count"].sum()
                    else:
                        total_restaurants = len(filtered_df)

                    # Get percentage of total if we have total restaurants data
                    if "restaurant_count" in region_metadata.columns:
                        region_info = region_metadata[
                            region_metadata["region"] == selected_region
                        ]
                        region_percentage = (
                            (total_restaurants / restaurant_locations["Count"].sum())
                            * 100
                            if restaurant_locations["Count"].sum() > 0
                            else 0
                        )
                    else:
                        region_percentage = None
                else:
                    st.error(
                        f"The data for {selected_region} doesn't contain location coordinates."
                    )
                    total_restaurants = 0
                    region_percentage = None
            else:
                st.error(f"Data for {selected_region} region could not be loaded.")
                total_restaurants = 0
                region_percentage = None
        else:
            st.error("Region data could not be loaded. Please check the data files.")
            total_restaurants = 0
            region_percentage = None

    # Display the map if not already displayed
    if region_selection_type == "All Regions":
        folium_static(map_with_heatmap)

    # Show statistics in three columns
    st.subheader("Restaurant Distribution Statistics")
    col1, col2, col3 = st.columns(3)

    if region_selection_type == "All Regions":
        col1.metric("Total Restaurants", f"{int(total_restaurants):,}")
        col2.metric("Areas Covered", f"{total_areas}")
        col3.metric("Avg. Restaurants per Area", f"{avg_per_area:.1f}")

        # Additional insights for all regions
        st.subheader("Distribution Insights")

        # Top 5 areas with highest concentration
        top_concentration = restaurant_locations.sort_values(
            by="Count", ascending=False
        ).head(5)
        top_percentage = (
            (top_concentration["Count"].sum() / total_restaurants) * 100
            if total_restaurants > 0
            else 0
        )

        st.markdown(
            f"The top 5 areas account for **{top_percentage:.1f}%** of all restaurants in Bangalore."
        )

        # Density metrics
        city_area = 709  # Bangalore area in sq km (approximate)
        density = total_restaurants / city_area if city_area > 0 else 0
        st.markdown(
            f"Bangalore has approximately **{density:.1f}** restaurants per square kilometer."
        )
    else:
        col1.metric("Total Restaurants", f"{int(total_restaurants):,}")
        if region_percentage is not None:
            col2.metric("% of All Bangalore Restaurants", f"{region_percentage:.1f}%")
        else:
            col2.metric("Region Count", "1")

        # Add a relevant third metric based on what data is available
        if "rating" in filtered_df.columns:
            avg_rating = filtered_df["rating"].mean()
            col3.metric("Average Rating", f"{avg_rating:.1f}")
        else:
            col3.metric("Restaurant Density", "N/A")

        # Additional region-specific insights if available
        if filtered_df is not None and len(filtered_df) > 0:
            st.subheader(f"Insights for {selected_region}")

            # Show cuisine distribution if available
            if "cuisines" in filtered_df.columns:
                # Extract and count cuisines
                all_cuisines = []
                for cuisines in filtered_df["cuisines"].dropna():
                    if isinstance(cuisines, str):
                        all_cuisines.extend([c.strip() for c in cuisines.split(",")])

                if all_cuisines:
                    cuisine_counts = pd.Series(all_cuisines).value_counts().head(5)
                    st.markdown("#### Popular Cuisines in this Region")
                    cuisine_df = pd.DataFrame(
                        {
                            "Cuisine": cuisine_counts.index,
                            "Count": cuisine_counts.values,
                        }
                    )
                    st.dataframe(cuisine_df)

elif selected_view == "Restaurant Ratings Heatmap":
    st.header("Restaurant Ratings Across Bangalore")
    st.markdown(
        """
    This heatmap shows areas with the highest average restaurant ratings.
    Brighter areas indicate locations with better-rated restaurants on average.
    """
    )

    # Additional analysis
    st.subheader("Top Areas by Average Rating")
    top_rated_areas = restaurant_ratings.sort_values(
        by="Avg_Rating", ascending=False
    ).head(10)
    st.dataframe(top_rated_areas[["Location", "Avg_Rating"]])

    # Create base map
    base_map = generate_base_map()

    # Create heatmap for ratings
    map_with_ratings = create_heatmap(
        base_map,
        restaurant_ratings[["Latitude", "Longitude", "Avg_Rating", "Location"]],
        "Average Rating",
    )

    # Display the map
    folium_static(map_with_ratings)

elif selected_view == "Cuisine Distribution Heatmap":
    st.header("Cuisine Distribution Across Bangalore")

    # Display top cuisines from the preloaded cuisine counts data
    if cuisine_data and "counts" in cuisine_data and cuisine_data["counts"] is not None:
        cuisine_counts_df = cuisine_data["counts"]

        st.subheader("Top Cuisines in Bangalore")
        st.dataframe(cuisine_counts_df.head(15))

        # Add cuisine selection to sidebar
        st.sidebar.subheader("Cuisine Selection")

        # Add cuisine list selector
        cuisine_list_type = st.sidebar.radio(
            "Cuisine List Type",
            options=["Popular Cuisines", "All Available Cuisines"],
            index=0,
        )

        if cuisine_list_type == "Popular Cuisines":
            cuisine_options = cuisine_counts_df.head(15)["cuisine"].tolist()
        else:
            cuisine_options = cuisine_counts_df["cuisine"].tolist()

        selected_cuisines = st.sidebar.multiselect(
            "Select Cuisines",
            options=cuisine_options,
            default=cuisine_options[:5],
        )

        # Proceed only if cuisines are selected
        if selected_cuisines:
            # Initialize a list to store DataFrames for each cuisine
            cuisine_dfs = []

            # Load and process data for each selected cuisine
            for cuisine in selected_cuisines:
                cuisine_file = cuisine.replace(" ", "_")
                with st.spinner(f"Loading {cuisine} restaurant data..."):
                    cuisine_data = load_cuisine_data(cuisine_file)
                    if cuisine_data is not None:
                        # Add cuisine name to the DataFrame
                        cuisine_data["cuisine_type"] = cuisine
                        cuisine_dfs.append(cuisine_data)

            if cuisine_dfs:
                # Combine all cuisine DataFrames
                combined_df = pd.concat(cuisine_dfs, ignore_index=True)

                # Group by location to get aggregated counts
                heatmap_data = (
                    combined_df.groupby(["Name", "Latitude", "Longitude"])
                    .agg(
                        {
                            "Count": "sum",
                            "cuisine_type": lambda x: ", ".join(sorted(set(x))),
                        }
                    )
                    .reset_index()
                )

                # Create base map
                base_map = generate_base_map()

                # Create cuisine heatmap
                cuisine_map = create_heatmap(
                    base_map,
                    heatmap_data[["Latitude", "Longitude", "Count", "Name"]],
                    f"Selected Cuisines Restaurant Density",
                )

                # Display the map
                folium_static(cuisine_map)

                # Show top areas for selected cuisines
                num_areas = 10
                st.subheader(f"Top {num_areas} Areas for Selected Cuisines")
                top_areas = heatmap_data.sort_values(by="Count", ascending=False).head(
                    num_areas
                )

                # Create a more detailed display DataFrame
                display_df = top_areas[["Name", "Count", "cuisine_type"]].rename(
                    columns={"cuisine_type": "Available Cuisines"}
                )
                st.dataframe(display_df)

                # Show aggregated statistics
                total_restaurants = heatmap_data["Count"].sum()
                total_areas = len(heatmap_data)
                avg_per_area = total_restaurants / total_areas if total_areas > 0 else 0

                # Display metrics in columns
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Restaurants", f"{total_restaurants}")
                col2.metric("Areas with Selected Cuisines", f"{total_areas}")
                col3.metric("Avg. Restaurants per Area", f"{avg_per_area:.1f}")

                # Add distribution analysis
                st.subheader("Cuisine Distribution Analysis")
                cuisine_distribution = (
                    combined_df.groupby("cuisine_type")["Count"]
                    .sum()
                    .sort_values(ascending=False)
                )
                st.bar_chart(cuisine_distribution)
            else:
                st.error("Could not load data for any of the selected cuisines.")
        else:
            st.info("Please select at least one cuisine to view its distribution.")
    else:
        st.error("Cuisine data could not be loaded. Please check the data files.")

elif selected_view == "Restaurant Recommendations & Navigation":
    st.header("Restaurant Recommendations & Navigation")

    # Map display area first (full width) - Always on top
    st.subheader("Interactive Map")

    # Decide which map to show
    if st.session_state.nav_map is not None:
        # Show navigation map
        folium_static(st.session_state.nav_map)
    elif (
        st.session_state.recommendations is not None
        and not st.session_state.recommendations.empty
    ):
        # Show map with multiple restaurants and dotted lines from current location
        with st.spinner("Generating map..."):
            user_location = (user_lat, user_lon)
            map_with_restaurants = create_folium_map(
                center_lat=user_lat, center_lon=user_lon, zoom_start=13
            )

            # Add user location marker
            folium.Marker(
                location=user_location,
                popup="Your Location",
                icon=folium.Icon(color="blue", icon="user", prefix="fa"),
            ).add_to(map_with_restaurants)

            # Add recommended restaurants and dotted lines
            for _, restaurant in st.session_state.recommendations.iterrows():
                # Add restaurant marker
                restaurant_location = (restaurant["latitude"], restaurant["longitude"])
                folium.Marker(
                    location=restaurant_location,
                    popup=folium.Popup(
                        f"""
                        <div style="width: 300px; font-family: Arial, sans-serif;">
                            <h4 style="color: #d32f2f; margin-bottom: 5px;">{restaurant['name']}</h4>
                            <div style="display: flex; margin-bottom: 8px;">
                                <div style="font-weight: bold; color: #d32f2f;">⭐ {restaurant['rating']}/5</div>
                                <div style="margin-left: 10px; color: #666;">({restaurant['votes']} votes)</div>
                            </div>
                            <div style="margin-bottom: 8px;">
                                <span style="font-weight: bold;">🍽️ Cuisines:</span> {restaurant['cuisines']}
                            </div>
                            <div style="margin-bottom: 8px;">
                                <span style="font-weight: bold;">💰 Approx Cost:</span> {"₹" + str(restaurant['approx_cost']) if pd.notna(restaurant['approx_cost']) else "Cost data not available"}
                            </div>
                            <div style="margin-bottom: 8px;">
                                <span style="font-weight: bold;">📍 Distance:</span> {restaurant['distance_km']:.2f} km
                            </div>
                            <div style="margin-bottom: 8px;">
                                <span style="font-weight: bold;">📍 Address:</span> {restaurant['address']}
                            </div>
                            <hr style="border-top: 1px solid #eee; margin: 8px 0;">
                        </div>
                        """,
                        max_width=300,
                    ),
                    tooltip=restaurant["name"],
                    icon=folium.Icon(color="red", icon="cutlery", prefix="fa"),
                ).add_to(map_with_restaurants)

                # Add dotted line from user to restaurant
                folium.PolyLine(
                    locations=[user_location, restaurant_location],
                    color="red",
                    weight=2,
                    opacity=0.7,
                    dash_array="5",
                    popup=f"Distance: {restaurant['distance_km']:.2f} km",
                ).add_to(map_with_restaurants)

            folium_static(map_with_restaurants)
    else:
        # Show default map
        default_map = create_folium_map(
            center_lat=DEFAULT_LATITUDE, center_lon=DEFAULT_LONGITUDE, zoom_start=12
        )
        folium_static(default_map)

    # Recommendations display area (below the map)
    if (
        st.session_state.recommendations is not None
        and not st.session_state.recommendations.empty
    ):
        st.subheader("Your Recommended Restaurants")

        for idx, restaurant in st.session_state.recommendations.iterrows():
            with st.container():
                col1, col2 = st.columns([3, 1])

                with col1:
                    # Restaurant name as a header
                    st.markdown(f"### {restaurant['name']}")
                    st.markdown(f"**Location:** {restaurant['address']}")

                    # Create three columns for the restaurant details
                    info_col1, info_col2, info_col3 = st.columns(3)

                    with info_col1:
                        st.markdown("**Rating**")
                        st.markdown(f"⭐ {restaurant['rating']}/5")
                        st.caption(f"{restaurant['votes']} votes")

                    with info_col2:
                        st.markdown("**Cost**")
                        st.markdown(
                            f"💰 ₹{restaurant['approx_cost']}"
                            if pd.notna(restaurant["approx_cost"])
                            else "Cost data not available"
                        )
                        st.caption("for two people")

                    with info_col3:
                        st.markdown("**Distance**")
                        st.markdown(f"📍 {restaurant['distance_km']:.2f} km")
                        st.caption("from your location")

                    # Cuisines in its own row
                    st.markdown("**Cuisines**")
                    st.markdown(f"{restaurant['cuisines']}")

                with col2:
                    # Center the button vertically
                    st.write("")
                    st.write("")
                    if st.button(
                        "🗺️ Navigate", key=f"select_{idx}", use_container_width=True
                    ):
                        st.session_state.selected_restaurant = restaurant
                        # Generate route map - now using precise coordinates from recommendation engine
                        with st.spinner(
                            "Generating route with precise location data..."
                        ):
                            route_map = generate_route_with_api(
                                user_lat=user_lat,
                                user_lon=user_lon,
                                restaurant_lat=restaurant["latitude"],
                                restaurant_lon=restaurant["longitude"],
                                restaurant_name=restaurant["name"],
                            )
                            st.session_state.nav_map = route_map
                            # st.experimental_rerun()

                # Divider between restaurants
                st.divider()
    else:
        if selected_view == "Restaurant Recommendations & Navigation":
            st.info(
                "Please set your preferences in the sidebar and click 'Find Restaurants' to get recommendations."
            )

# Footer
st.markdown("---")
st.markdown("© 2025 RestauLore | Powered by Folium, Streamlit & Zomato Data")
