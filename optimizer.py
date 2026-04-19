import numpy as np
import pandas as pd


def estimate_drive_minutes(distance_km, traffic_level, road_condition):
    road_multiplier = {"Good": 1.0, "Average": 1.15, "Poor": 1.35}
    adjusted_speed = max(18.0, 58 - float(traffic_level) * 4.8)
    adjusted_speed = adjusted_speed / road_multiplier.get(str(road_condition), 1.15)
    return float(distance_km) / max(adjusted_speed, 12.0) * 60


def build_strategy_table(scenario, preference_weights):
    capacity = float(scenario["Battery_Capacity_kWh"])
    soc = float(scenario["State_of_Charge_%"])
    rate = max(float(scenario["Charging_Rate_kW"]), 1.0)
    distance = float(scenario["Distance_to_Destination_km"])
    consumption = max(float(scenario["Energy_Consumption_Rate_kWh/km"]), 0.1)
    queue_time = max(float(scenario["Queue_Time_mins"]), 0.0)
    traffic = float(scenario["Traffic_Data"])
    road = scenario["Road_Conditions"]
    current_energy = capacity * soc / 100
    trip_energy = distance * consumption

    strategies = [
        {"strategy": "Immediate minimal top-up", "reserve_ratio": 0.10, "comfort_weight": 0.8},
        {"strategy": "Balanced recharge", "reserve_ratio": 0.20, "comfort_weight": 1.0},
        {"strategy": "High-resilience charge", "reserve_ratio": 0.35, "comfort_weight": 1.2},
    ]

    rows = []
    for item in strategies:
        reserve_energy = capacity * item["reserve_ratio"]
        charge_needed = max(trip_energy + reserve_energy - current_energy, 0)
        charging_minutes = charge_needed / rate * 60
        drive_minutes = estimate_drive_minutes(distance, traffic, road)
        battery_risk = max((trip_energy - current_energy) / max(capacity, 1), 0) * 100
        buffer_after_trip = current_energy + charge_needed - trip_energy

        total_time_score = drive_minutes + queue_time + charging_minutes
        battery_score = max(0, 25 - buffer_after_trip) + battery_risk
        stress_score = traffic * 6 + (12 if road == "Poor" else 5 if road == "Average" else 0)

        final_score = (
            total_time_score * preference_weights["time_weight"]
            + battery_score * preference_weights["battery_weight"]
            + stress_score * preference_weights["reliability_weight"] * item["comfort_weight"]
        )

        rows.append(
            {
                "strategy": item["strategy"],
                "charge_needed_kwh": charge_needed,
                "charging_minutes": charging_minutes,
                "drive_minutes": drive_minutes,
                "queue_minutes": queue_time,
                "buffer_after_trip_kwh": buffer_after_trip,
                "optimization_score": final_score,
            }
        )

    return pd.DataFrame(rows).sort_values("optimization_score").reset_index(drop=True)


def build_route_candidates(df, scenario, top_n=5):
    candidates = df.copy()
    if candidates.empty:
        return candidates

    target_hour = float(scenario["hour"])
    target_distance = float(scenario["Distance_to_Destination_km"])
    target_traffic = float(scenario["Traffic_Data"])
    target_weather = scenario["Weather_Conditions"]
    target_road = scenario["Road_Conditions"]

    candidates["candidate_drive_mins"] = candidates.apply(
        lambda row: estimate_drive_minutes(
            row["Distance_to_Destination_km"],
            row["Traffic_Data"],
            row["Road_Conditions"],
        ),
        axis=1,
    )
    candidates["candidate_total_mins"] = (
        candidates["candidate_drive_mins"]
        + candidates["Queue_Time_mins"]
        + candidates["Time_Spent_Charging_mins"] * 0.25
    )

    candidates["similarity_score"] = (
        (candidates["hour"] - target_hour).abs() * 1.6
        + (candidates["Distance_to_Destination_km"] - target_distance).abs() * 0.5
        + (candidates["Traffic_Data"] - target_traffic).abs() * 4.5
        + np.where(candidates["Weather_Conditions"].eq(target_weather), 0, 8)
        + np.where(candidates["Road_Conditions"].eq(target_road), 0, 10)
        + candidates["Queue_Time_mins"] * 0.9
    )
    candidates = candidates.sort_values(["similarity_score", "candidate_total_mins"])
    columns = [
        col
        for col in [
            "Date_Time",
            "Distance_to_Destination_km",
            "Traffic_Data",
            "Road_Conditions",
            "Weather_Conditions",
            "Queue_Time_mins",
            "candidate_drive_mins",
            "candidate_total_mins",
        ]
        if col in candidates.columns
    ]
    return candidates[columns].head(top_n).reset_index(drop=True)


def optimizer_summary(df, scenario, forecast_load_kw=None, forecast_queue_mins=None):
    preference_weights = {
        "time_weight": float(scenario.get("time_priority", 1.0)),
        "battery_weight": float(scenario.get("battery_priority", 1.0)),
        "reliability_weight": float(scenario.get("reliability_priority", 1.0)),
    }
    strategies = build_strategy_table(scenario, preference_weights)
    best_strategy = strategies.iloc[0].to_dict()
    routes = build_route_candidates(df, scenario, top_n=5)

    current_energy = float(scenario["Battery_Capacity_kWh"]) * float(scenario["State_of_Charge_%"]) / 100
    trip_energy = float(scenario["Distance_to_Destination_km"]) * float(
        scenario["Energy_Consumption_Rate_kWh/km"]
    )
    range_km = current_energy / max(float(scenario["Energy_Consumption_Rate_kWh/km"]), 0.1)

    return {
        "current_energy_kwh": current_energy,
        "trip_energy_kwh": trip_energy,
        "estimated_range_km": range_km,
        "predicted_load_kw": forecast_load_kw,
        "predicted_queue_mins": forecast_queue_mins,
        "best_strategy": best_strategy,
        "strategy_table": strategies,
        "route_candidates": routes,
    }
