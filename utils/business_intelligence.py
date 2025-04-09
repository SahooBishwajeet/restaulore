import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap, MarkerCluster
import os
from pathlib import Path
from math import radians
from sklearn.metrics.pairwise import haversine_distances

from utils.data_loader import (
    load_census_data,
    get_project_root,
    load_processed_zomato_data,
    load_locations_data,
)

from config.settings import TOP_CUISINES
from utils.mapping import generate_base_map


def analyze_restaurant_feasibility(
    location,
    cuisine_type=None,
    target_customers=None,
    radius_km=2,
    investment=None,
    avg_price_per_person=None,
):
    # Load required data
    df_zomato = load_processed_zomato_data()
    census_data = load_census_data()

    if df_zomato is None:
        return {"error": "Restaurant data not available"}

    # Extract location coordinates
    lat, lon = location

    # 1. Calculate competition metrics
    competition_metrics = analyze_competition(
        df_zomato, lat, lon, radius_km, cuisine_type
    )

    # 2. Calculate population metrics (based on census data)
    population_metrics = analyze_population(census_data, lat, lon, radius_km)

    # 3. Calculate market potential
    market_potential = calculate_market_potential(
        competition_metrics,
        population_metrics,
        target_customers,
        cuisine_type,
        avg_price_per_person,
    )

    # 4. Calculate investment metrics if investment is provided
    investment_metrics = {}
    if investment is not None:
        investment_metrics = calculate_investment_metrics(investment, market_potential)

    # 5. Calculate overall feasibility score (0-100)
    feasibility_score = calculate_feasibility_score(
        competition_metrics, population_metrics, market_potential, investment_metrics
    )

    # Prepare result
    result = {
        "location": {"latitude": lat, "longitude": lon},
        "competition_metrics": competition_metrics,
        "population_metrics": population_metrics,
        "market_potential": market_potential,
        "feasibility_score": feasibility_score,
        "recommendation": get_recommendation(feasibility_score),
    }

    # Add investment metrics if available
    if investment_metrics:
        result["investment_metrics"] = investment_metrics

    return result


def analyze_competition(df_zomato, lat, lon, radius_km, cuisine_type=None):
    # Convert the user location to radians for haversine calculation
    user_coords = np.array([[radians(lat), radians(lon)]])

    # Filter restaurants with valid coordinates
    df_valid = df_zomato.dropna(subset=["latitude", "longitude"])

    # Convert restaurant coordinates to radians
    rest_coords = np.array(
        [
            [radians(rlat), radians(rlon)]
            for rlat, rlon in zip(df_valid["latitude"], df_valid["longitude"])
        ]
    )

    if len(rest_coords) == 0:
        return {
            "total_restaurants": 0,
            "cuisine_restaurants": 0,
            "competition_score": 0,
        }

    # Calculate distances
    distances = (
        haversine_distances(user_coords, rest_coords)[0] * 6371
    )  # Earth radius in km

    # Add distances to the dataframe
    df_valid = df_valid.copy()
    df_valid["distance_km"] = distances

    # Filter restaurants within the specified radius
    nearby_restaurants = df_valid[df_valid["distance_km"] <= radius_km]

    # Count total restaurants in the area
    total_restaurants = len(nearby_restaurants)

    # If cuisine type is specified, count restaurants of that cuisine
    if cuisine_type and isinstance(cuisine_type, list):
        cuisine_restaurants = nearby_restaurants[
            nearby_restaurants["cuisines"].apply(
                lambda x: any(
                    cuisine in x for cuisine in cuisine_type if isinstance(x, str)
                )
            )
        ]
        cuisine_count = len(cuisine_restaurants)
    elif cuisine_type and isinstance(cuisine_type, str):
        cuisine_restaurants = nearby_restaurants[
            nearby_restaurants["cuisines"].str.contains(
                cuisine_type, na=False, case=False
            )
        ]
        cuisine_count = len(cuisine_restaurants)
    else:
        cuisine_count = 0

    # Calculate average rating of nearby restaurants
    avg_rating = nearby_restaurants["rating"].mean() if total_restaurants > 0 else 0

    # Calculate competition density (restaurants per sq km)
    area = np.pi * radius_km * radius_km
    density = total_restaurants / area if area > 0 else 0

    # Calculate competition score (inverse relationship - more competition means lower score)
    # Scale from 0-100, where 0 means extreme competition and 100 means no competition
    competition_score = max(
        0, 100 - (density * 10)
    )  # Adjust multiplier based on typical density

    # If cuisine_type is specified, adjust score based on cuisine competition
    if cuisine_type and total_restaurants > 0:
        cuisine_ratio = cuisine_count / total_restaurants
        # If many restaurants of this cuisine, decrease score (higher competition)
        # If few restaurants, increase score (opportunity gap)
        if cuisine_ratio > 0.3:  # More than 30% of restaurants are this cuisine
            competition_score *= 1 - (cuisine_ratio - 0.3)  # Decrease score
        elif cuisine_ratio < 0.1:  # Less than 10% of restaurants are this cuisine
            competition_score *= 1 + (0.1 - cuisine_ratio) * 2  # Increase score

        # Ensure score stays in 0-100 range
        competition_score = min(100, max(0, competition_score))

    return {
        "total_restaurants": total_restaurants,
        "cuisine_restaurants": cuisine_count,
        "cuisine_percentage": (
            (cuisine_count / total_restaurants * 100) if total_restaurants > 0 else 0
        ),
        "avg_rating": avg_rating,
        "restaurant_density": density,
        "competition_score": competition_score,
    }


def analyze_population(census_data, lat, lon, radius_km):
    # If census data isn't available, use estimates
    if census_data is None:
        # Default estimates for Bangalore
        return {
            "estimated_population": 75000,  # Average per ward
            "population_density": 15000,  # Average per sq km
            "spending_power_index": 65,  # Medium-high (0-100)
            "data_source": "estimated",
        }

    # Find the closest ward to the given coordinates
    # Convert census data coordinates to radians for calculation
    # Note: This is a simplified approach. In a full implementation,
    # we would determine which ward the point falls within using spatial operations

    # For this implementation, we'll use average figures from the census data
    # with some variation based on the location within the city

    # Calculate distance from city center (12.9716, 77.5946)
    city_center_coords = np.array([[radians(12.9716), radians(77.5946)]])
    location_coords = np.array([[radians(lat), radians(lon)]])

    distance_from_center = (
        haversine_distances(city_center_coords, location_coords)[0][0] * 6371
    )  # km

    # Adjust population density based on distance from center
    # Further from center typically means lower density
    avg_population = census_data["Population"].mean()
    avg_density = avg_population / 5  # Assuming average ward size of 5 sq km

    # Density decreases as we move away from center
    density_factor = max(0.5, min(1.5, 1.5 - distance_from_center / 20))
    adj_density = avg_density * density_factor

    # Calculate estimated population in the area
    area = np.pi * radius_km * radius_km
    estimated_population = adj_density * area

    # Calculate spending power index (higher near center, lower in outskirts)
    base_spending_power = 65  # Base index (0-100)
    spending_power_index = base_spending_power * max(
        0.7, min(1.3, 1.2 - distance_from_center / 25)
    )

    return {
        "estimated_population": int(estimated_population),
        "population_density": int(adj_density),
        "spending_power_index": int(spending_power_index),
        "data_source": "extrapolated",
    }


def calculate_market_potential(
    competition_metrics,
    population_metrics,
    target_customers=None,
    cuisine_type=None,
    avg_price_per_person=None,
):
    # Base potential customer pool (20% of population eats out daily)
    total_daily_diners = population_metrics["estimated_population"] * 0.20

    # Calculate capture rate based on competition
    total_restaurants = competition_metrics["total_restaurants"]

    # Calculate market share percentage based on competition
    # With no competition, capture up to 20% of dining population
    # As competition increases, capture rate decreases
    if total_restaurants == 0:
        capture_rate_max = 60.0
        capture_rate_min = 30.0
    else:
        # Exponential decay formula for market share
        capture_rate_max = max(0.5, min(60.0, 60.0 * np.exp(-0.1 * total_restaurants)))
        capture_rate_min = max(0.2, min(30.0, 30.0 * np.exp(-0.1 * total_restaurants)))

    # Adjust for cuisine type popularity
    if cuisine_type and isinstance(cuisine_type, list):
        cuisine_type = [c.lower() for c in cuisine_type]
        popular_cuisines = [c.lower() for c in TOP_CUISINES]
        cuisine_factor = 1.0
        for cuisine in cuisine_type:
            if cuisine in popular_cuisines:
                cuisine_factor += 0.3  # Popular cuisines capture more customers
            else:
                cuisine_factor -= 0.2  # Less popular cuisines capture fewer customers
        cuisine_factor = max(0.5, min(1.5, cuisine_factor))
        capture_rate_min *= cuisine_factor
        capture_rate_max *= cuisine_factor
    elif cuisine_type:
        cuisine_type = cuisine_type.lower()
        popular_cuisines = [c.lower() for c in TOP_CUISINES]
        if cuisine_type in popular_cuisines:
            cuisine_factor = 1.3  # Popular cuisines can capture more customers
        else:
            cuisine_factor = 0.8  # Less popular cuisines capture fewer customers

        capture_rate_min *= cuisine_factor
        capture_rate_max *= cuisine_factor

        # Consider cuisine-specific competition
        cuisine_count = competition_metrics["cuisine_restaurants"]
        if cuisine_count > 0:
            # Reduce capture percentage if many similar restaurants exist
            cuisine_competition_factor = max(0.5, np.exp(-0.15 * cuisine_count))
            capture_rate_min *= cuisine_competition_factor
            capture_rate_max *= cuisine_competition_factor

    # Ensure capture rates stay within reasonable bounds
    capture_rate_min = max(0.2, min(10.0, capture_rate_min))
    capture_rate_max = max(0.5, min(20.0, capture_rate_max))

    # Calculate estimated customer range
    min_customers = int(total_daily_diners * capture_rate_min / 100)
    max_customers = int(total_daily_diners * capture_rate_max / 100)
    avg_customers = (min_customers + max_customers) // 2

    # Calculate average spending per customer based on spending power index
    spending_power = population_metrics["spending_power_index"]

    # Use provided price or calculate based on spending power
    if avg_price_per_person:
        avg_spend = avg_price_per_person
    else:
        base_avg_spend = 200
        # Adjust for spending power
        avg_spend = base_avg_spend * (0.5 + spending_power / 100)

    # If target_customers is specified, calculate the feasibility of achieving that target
    target_feasibility = 100  # Default: 100% feasible
    if target_customers and target_customers > 0:
        ratio = target_customers / max(1, max_customers)
        # If ratio > 1, target is higher than estimate, which means less feasible
        if ratio > 1:
            target_feasibility = max(0, min(100, 100 / ratio))

    # Calculate daily and monthly revenue
    if target_customers is None:
        daily_customers = avg_customers
    else:
        # 30-70 weighted average favoring the lower value
        lower_value = min(target_customers, avg_customers)
        higher_value = max(target_customers, avg_customers)
        daily_customers = int(0.7 * lower_value + 0.3 * higher_value)
    daily_revenue = daily_customers * avg_spend
    monthly_revenue = daily_revenue * 30

    # Calculate operating expenses (typically 80% of revenue in restaurant industry)
    monthly_expenses = monthly_revenue * 0.8
    monthly_profit = monthly_revenue - monthly_expenses
    annual_profit = monthly_profit * 12

    # Calculate market share
    market_share = 1 / (total_restaurants + 1) if total_restaurants > 0 else 1

    return {
        "potential_customer_percentage": (capture_rate_min + capture_rate_max) / 2,
        "estimated_daily_customers_min": min_customers,
        "estimated_daily_customers_max": max_customers,
        "estimated_daily_customers": avg_customers,
        "target_daily_customers": target_customers if target_customers else None,
        "target_feasibility": target_feasibility if target_customers else None,
        "average_spend_per_customer": int(avg_spend),
        "estimated_daily_revenue": int(daily_revenue),
        "estimated_monthly_revenue": int(monthly_revenue),
        "estimated_monthly_expenses": int(monthly_expenses),
        "estimated_monthly_profit": int(monthly_profit),
        "estimated_annual_profit": int(annual_profit),
        "potential_market_share": market_share * 100,  # As percentage
    }


def calculate_investment_metrics(investment, market_potential):
    # Convert investment to rupees (1 lakh = 100,000)
    investment_rupees = investment * 100000

    # Monthly revenue from market potential
    monthly_revenue = market_potential["estimated_monthly_revenue"]

    # Assume costs are 70% of revenue in restaurant business
    monthly_profit = monthly_revenue * 0.3
    annual_profit = monthly_profit * 12

    # Calculate ROI (Return on Investment)
    roi = annual_profit / investment_rupees * 100 if investment_rupees > 0 else 0

    # Calculate payback period in months
    payback_period = (
        investment_rupees / monthly_profit if monthly_profit > 0 else float("inf")
    )

    # Calculate ROI score (0-100, higher is better)
    # Good annual ROI for restaurants is around 15-20%
    roi_score = min(100, max(0, roi * 5))  # 20% ROI = 100 score

    return {
        "investment_amount": investment_rupees,
        "monthly_profit": int(monthly_profit),
        "annual_profit": int(annual_profit),
        "annual_roi_percentage": roi,
        "payback_period_months": payback_period,
        "roi_score": roi_score,
    }


def calculate_feasibility_score(
    competition_metrics, population_metrics, market_potential, investment_metrics=None
):
    # Weights for different factors
    w_competition = 0.30
    w_population = 0.20
    w_revenue = 0.30
    w_target = 0.10
    w_roi = 0.10

    # Competition score is already 0-100
    competition_score = competition_metrics["competition_score"]

    # Population score (0-100) based on density
    # Higher density is better, but with diminishing returns
    population_density = population_metrics["population_density"]
    population_score = min(100, max(0, 100 * (1 - np.exp(-population_density / 5000))))

    # Revenue score (0-100) based on estimated monthly revenue
    monthly_revenue = market_potential["estimated_monthly_revenue"]
    revenue_score = min(100, max(0, 100 * (1 - np.exp(-monthly_revenue / 1000000))))

    # Target feasibility score if a target was specified
    # Ensure target_score is a number with a default value of 100
    target_score = market_potential.get("target_feasibility", 100)
    if target_score is None:
        target_score = 100

    # ROI score if investment metrics were provided
    roi_score = 100  # Default value
    if investment_metrics is not None:
        roi_score = investment_metrics.get("roi_score", 100)
        if roi_score is None:
            roi_score = 100

    # Calculate weighted score
    total_score = (
        w_competition * competition_score
        + w_population * population_score
        + w_revenue * revenue_score
        + w_target * target_score
        + w_roi * roi_score
    )

    return min(100, max(0, total_score))


def get_recommendation(score):
    if score >= 80:
        return "Highly Recommended - This location has excellent potential for a restaurant."
    elif score >= 60:
        return "Recommended - This location has good potential for a restaurant."
    elif score >= 40:
        return "Moderate Potential - This location has moderate potential. Consider carefully."
    elif score >= 20:
        return "Low Potential - This location has limited potential. Not recommended."
    else:
        return (
            "Not Recommended - This location has very low potential for a restaurant."
        )


def generate_feasibility_map(location, radius_km=2, cuisine_type=None):
    lat, lon = location

    # Load restaurant data
    df_zomato = load_processed_zomato_data()

    # Create a base map centered on the location
    m = generate_base_map(default_location=[lat, lon], default_zoom_start=14)

    # Add a marker for the proposed location
    folium.Marker(
        location=[lat, lon],
        popup="Proposed Restaurant Location",
        icon=folium.Icon(color="green", icon="info-sign"),
    ).add_to(m)

    # Add a circle to represent the analysis radius
    folium.Circle(
        location=[lat, lon],
        radius=radius_km * 1000,  # Convert km to meters
        color="blue",
        fill=True,
        fill_opacity=0.1,
    ).add_to(m)

    # Filter nearby restaurants
    if df_zomato is not None:
        # Calculate distances
        user_coords = np.array([[radians(lat), radians(lon)]])
        df_valid = df_zomato.dropna(subset=["latitude", "longitude"])

        if len(df_valid) > 0:
            rest_coords = np.array(
                [
                    [radians(rlat), radians(rlon)]
                    for rlat, rlon in zip(df_valid["latitude"], df_valid["longitude"])
                ]
            )

            distances = haversine_distances(user_coords, rest_coords)[0] * 6371
            df_valid = df_valid.copy()
            df_valid["distance_km"] = distances

            nearby_restaurants = df_valid[df_valid["distance_km"] <= radius_km]

            # Filter by cuisine if specified
            if cuisine_type and isinstance(cuisine_type, list):
                cuisine_restaurants = nearby_restaurants[
                    nearby_restaurants["cuisines"].apply(
                        lambda x: any(
                            cuisine in x
                            for cuisine in cuisine_type
                            if isinstance(x, str)
                        )
                    )
                ]
                # Add markers for cuisine-specific restaurants
                for idx, rest in cuisine_restaurants.iterrows():
                    folium.Marker(
                        location=[rest["latitude"], rest["longitude"]],
                        popup=f"{rest['name']} ({', '.join(cuisine_type)})",
                        tooltip=rest["name"],
                        icon=folium.Icon(color="red", icon="cutlery", prefix="fa"),
                    ).add_to(m)
            elif cuisine_type:
                cuisine_restaurants = nearby_restaurants[
                    nearby_restaurants["cuisines"].str.contains(
                        cuisine_type, na=False, case=False
                    )
                ]

                # Add markers for cuisine-specific restaurants
                for idx, rest in cuisine_restaurants.iterrows():
                    folium.Marker(
                        location=[rest["latitude"], rest["longitude"]],
                        popup=f"{rest['name']} ({cuisine_type})",
                        tooltip=rest["name"],
                        icon=folium.Icon(color="red", icon="cutlery", prefix="fa"),
                    ).add_to(m)

                # Add other restaurant markers
                other_restaurants = nearby_restaurants[
                    ~nearby_restaurants.index.isin(cuisine_restaurants.index)
                ]

                if len(other_restaurants) > 0:
                    cluster = MarkerCluster(name="Other Restaurants").add_to(m)
                    for idx, rest in other_restaurants.iterrows():
                        folium.Marker(
                            location=[rest["latitude"], rest["longitude"]],
                            popup=f"{rest['name']} ({rest['cuisines']})",
                            tooltip=rest["name"],
                            icon=folium.Icon(color="blue", icon="cutlery", prefix="fa"),
                        ).add_to(cluster)
            else:
                # Add all restaurant markers to a cluster
                cluster = MarkerCluster(name="Nearby Restaurants").add_to(m)
                for idx, rest in nearby_restaurants.iterrows():
                    folium.Marker(
                        location=[rest["latitude"], rest["longitude"]],
                        popup=f"{rest['name']} ({rest['cuisines']})",
                        tooltip=rest["name"],
                        icon=folium.Icon(color="blue", icon="cutlery", prefix="fa"),
                    ).add_to(cluster)

    # Add layer control
    folium.LayerControl().add_to(m)

    return m


def find_optimal_locations(
    cuisine_type=None,
    target_customers=None,
    investment=None,
    max_locations=5,
    avg_price_per_person=None,
):
    # Load required data
    df_zomato = load_processed_zomato_data()
    census_data = load_census_data()
    locations_df = load_locations_data()

    if df_zomato is None or locations_df is None:
        return {"error": "Required data not available"}

    # Create a list to store location scores
    location_scores = []

    # Process each location area
    for idx, location_row in locations_df.iterrows():
        location_name = location_row["location_name"]
        lat = location_row["latitude"]
        lon = location_row["longitude"]

        # Skip if coordinates are missing
        if pd.isna(lat) or pd.isna(lon):
            continue

        try:
            # Analyze this location (use a smaller radius for quicker analysis)
            radius_km = 1.5
            competition_metrics = analyze_competition(
                df_zomato, lat, lon, radius_km, cuisine_type
            )
            population_metrics = analyze_population(census_data, lat, lon, radius_km)
            market_potential = calculate_market_potential(
                competition_metrics,
                population_metrics,
                target_customers,
                cuisine_type,
                avg_price_per_person,
            )

            # Calculate investment metrics if investment is provided
            investment_metrics = {}
            if investment is not None:
                investment_metrics = calculate_investment_metrics(
                    investment, market_potential
                )

            # Calculate overall score
            feasibility_score = calculate_feasibility_score(
                competition_metrics,
                population_metrics,
                market_potential,
                investment_metrics,
            )

            # Create a summary of this location
            location_summary = {
                "location_name": location_name,
                "latitude": lat,
                "longitude": lon,
                "feasibility_score": feasibility_score,
                "competition_score": competition_metrics["competition_score"],
                "nearby_restaurants": competition_metrics["total_restaurants"],
                "cuisine_restaurants": competition_metrics.get(
                    "cuisine_restaurants", 0
                ),
                "estimated_daily_customers": market_potential[
                    "estimated_daily_customers"
                ],
                "estimated_monthly_revenue": market_potential[
                    "estimated_monthly_revenue"
                ],
                "population_density": population_metrics["population_density"],
                "spending_power_index": population_metrics["spending_power_index"],
            }

            # Add ROI information if available
            if investment_metrics:
                location_summary.update(
                    {
                        "annual_roi_percentage": investment_metrics[
                            "annual_roi_percentage"
                        ],
                        "payback_period_months": investment_metrics[
                            "payback_period_months"
                        ],
                    }
                )

            location_scores.append(location_summary)

        except Exception as e:
            print(f"Error analyzing {location_name}: {str(e)}")
            continue

    # Sort locations by feasibility score (descending)
    location_scores.sort(key=lambda x: x["feasibility_score"], reverse=True)

    # Return top locations
    top_locations = location_scores[:max_locations]

    return top_locations


def generate_optimal_locations_map(locations, cuisine_type=None):
    # Create base map centered on the first location
    if not locations or len(locations) == 0:
        # Default center if no locations
        center = [12.9716, 77.5946]  # Bangalore center
    else:
        center = [locations[0]["latitude"], locations[0]["longitude"]]

    # Create map
    m = generate_base_map(default_location=center, default_zoom_start=12)

    # Add markers for all optimal locations
    for i, location in enumerate(locations):
        # Ensure all dictionary keys are strings to avoid the split error
        location_dict = {str(k): v for k, v in location.items()}

        # Format popup content
        popup_content = f"""
        <div style="width: 300px; font-family: Arial, sans-serif;">
            <h4 style="color: #4CAF50; margin-bottom: 5px;">{location['location_name']}</h4>
            <div style="display: flex; margin-bottom: 8px;">
                <div style="font-weight: bold; color: #4CAF50;">Score: {location['feasibility_score']:.1f}/100</div>
            </div>
            <div style="margin-bottom: 8px;">
                <span style="font-weight: bold;">Estimated Daily Customers:</span> {location['estimated_daily_customers']}
            </div>
            <div style="margin-bottom: 8px;">
                <span style="font-weight: bold;">Estimated Monthly Revenue:</span> ₹{location['estimated_monthly_revenue']:,}
            </div>
            <div style="margin-bottom: 8px;">
                <span style="font-weight: bold;">Nearby Restaurants:</span> {location['nearby_restaurants']}
            </div>
        """

        # Add cuisine-specific info if provided
        if cuisine_type and "cuisine_restaurants" in location:
            popup_content += f"""
            <div style="margin-bottom: 8px;">
                <span style="font-weight: bold;">Nearby {cuisine_type} Restaurants:</span> {location['cuisine_restaurants']}
            </div>
            """

        # Add ROI info if available
        if "annual_roi_percentage" in location:
            popup_content += f"""
            <div style="margin-bottom: 8px;">
                <span style="font-weight: bold;">Annual ROI:</span> {location['annual_roi_percentage']:.1f}%
            </div>
            <div style="margin-bottom: 8px;">
                <span style="font-weight: bold;">Payback Period:</span> {location['payback_period_months']:.1f} months
            </div>
            """

        popup_content += "</div>"

        # Add the marker to the map
        folium.Marker(
            location=[location["latitude"], location["longitude"]],
            popup=folium.Popup(popup_content, max_width=350),
            tooltip=f"{i+1}. {location['location_name']} - Score: {location['feasibility_score']:.1f}",
            icon=folium.Icon(color="green", icon="star"),
        ).add_to(m)

        # Add a circle to show the analysis radius (1.5km default)
        folium.Circle(
            radius=1500,  # 1.5km radius
            location=[location["latitude"], location["longitude"]],
            color="#4CAF50",
            fill=True,
            fill_opacity=0.2,
            tooltip=f"Analysis Area: 1.5km radius",
        ).add_to(m)

    return m
