import os
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_DATASET_PATHS = [
    "ev_charging_dataset.csv",
    "Project Dataset ev fleet.csv",
    "ev fleet.csv",
    "Project Dataset.csv",
    "dataset.csv",
]

BASE_DIR = Path(__file__).resolve().parent

NUMERIC_FEATURES = [
    "Battery_Capacity_kWh",
    "State_of_Charge_%",
    "Energy_Consumption_Rate_kWh/km",
    "Distance_to_Destination_km",
    "Traffic_Data",
    "Charging_Rate_kW",
    "Queue_Time_mins",
    "Station_Capacity_EV",
    "Time_Spent_Charging_mins",
    "Energy_Drawn_kWh",
    "Fleet_Size",
    "Temperature_C",
    "Wind_Speed_m/s",
    "Precipitation_mm",
]

CATEGORICAL_FEATURES = [
    "Road_Conditions",
    "Weather_Conditions",
]

ID_COLUMNS = [
    "Vehicle_ID",
    "Charging_Station_ID",
]


def load_dataset(uploaded_file=None):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)

    for path in DEFAULT_DATASET_PATHS:
        candidate = BASE_DIR / path
        if candidate.exists():
            return pd.read_csv(candidate)

    return None


def generate_synthetic_dataset(
    days=21,
    vehicles=24,
    stations=6,
    seed=42,
    start_date="2026-01-01",
):
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(start=start_date, periods=days * 24, freq="h")

    weather_options = np.array(["Clear", "Cloudy", "Rain", "Storm"])
    road_options = np.array(["Good", "Average", "Poor"])

    station_lat = 28.60 + rng.normal(0, 0.06, stations)
    station_lon = 77.20 + rng.normal(0, 0.06, stations)
    station_capacity = rng.integers(3, 9, size=stations)
    station_base_rate = rng.uniform(45, 160, size=stations)

    records = []
    for ts in timestamps:
        hour = ts.hour
        weekday = ts.weekday()
        weekend = 1 if weekday >= 5 else 0

        traffic_bias = 2.2 if hour in {7, 8, 9, 17, 18, 19, 20} else 0.8
        rain_factor = rng.choice([0, 0.4, 0.8, 1.4], p=[0.55, 0.2, 0.18, 0.07])
        traffic_base = traffic_bias + rain_factor + (0.4 if weekend else 0)

        weather = rng.choice(weather_options, p=[0.47, 0.28, 0.18, 0.07])
        road = rng.choice(road_options, p=[0.60, 0.28, 0.12])

        for vehicle_id in range(1, vehicles + 1):
            station_idx = int(rng.integers(0, stations))
            battery_capacity = float(rng.uniform(45, 110))
            soc = float(np.clip(rng.normal(42 - traffic_bias * 3, 18), 8, 96))
            efficiency = float(np.clip(rng.normal(0.18 + rain_factor * 0.02, 0.035), 0.11, 0.31))
            distance = float(np.clip(rng.gamma(shape=2.8, scale=8.5), 1, 120))

            current_lat = float(28.61 + rng.normal(0, 0.09))
            current_lon = float(77.22 + rng.normal(0, 0.09))
            destination_lat = float(current_lat + rng.normal(0, 0.04))
            destination_lon = float(current_lon + rng.normal(0, 0.04))

            traffic_level = int(np.clip(round(rng.normal(traffic_base, 1.5)), 0, 11))
            queue_time = float(
                np.clip(
                    rng.normal(
                        4.5 + traffic_level * 1.2 + (8 if weather == "Storm" else 3 if weather == "Rain" else 0),
                        3.5,
                    ),
                    0,
                    65,
                )
            )
            charge_rate = float(np.clip(rng.normal(station_base_rate[station_idx], 12), 30, 180))
            time_spent = float(np.clip(rng.normal(32 + queue_time * 0.65 + (18 if soc < 20 else 5), 12), 12, 150))
            energy_drawn = float(
                np.clip(
                    distance * efficiency * rng.uniform(0.8, 1.25) + max(0, 30 - soc) * 0.32,
                    4,
                    90,
                )
            )
            precipitation = float(
                np.clip(rng.gamma(1.5, 1.2), 0, 22) if weather in {"Rain", "Storm"} else 0
            )
            temperature = float(np.clip(rng.normal(29 if hour >= 10 and hour <= 17 else 21, 5), 4, 43))
            wind_speed = float(np.clip(rng.normal(3.5 + rain_factor * 1.8, 1.4), 0, 15))
            fleet_size = int(rng.integers(max(vehicles, 20), max(vehicles * 5, 40)))
            fleet_schedule = int(hour in {6, 7, 8, 17, 18, 19})
            charging_pref = int(rng.random() > 0.72)

            charging_load = float(
                np.clip(
                    charge_rate * 0.23
                    + queue_time * 0.18
                    + traffic_level * 0.9
                    + fleet_size * 0.05
                    + (7 if weather == "Storm" else 3 if weather == "Rain" else 0)
                    + rng.normal(0, 4),
                    8,
                    95,
                )
            )

            records.append(
                {
                    "Date_Time": ts,
                    "Vehicle_ID": vehicle_id,
                    "Battery_Capacity_kWh": battery_capacity,
                    "State_of_Charge_%": soc,
                    "Energy_Consumption_Rate_kWh/km": efficiency,
                    "Current_Latitude": current_lat,
                    "Current_Longitude": current_lon,
                    "Destination_Latitude": destination_lat,
                    "Destination_Longitude": destination_lon,
                    "Distance_to_Destination_km": distance,
                    "Traffic_Data": traffic_level,
                    "Road_Conditions": road,
                    "Charging_Station_ID": station_idx + 1,
                    "Charging_Rate_kW": charge_rate,
                    "Queue_Time_mins": queue_time,
                    "Station_Capacity_EV": int(station_capacity[station_idx]),
                    "Time_Spent_Charging_mins": time_spent,
                    "Energy_Drawn_kWh": energy_drawn,
                    "Session_Start_Hour": hour,
                    "Fleet_Size": fleet_size,
                    "Fleet_Schedule": fleet_schedule,
                    "Temperature_C": temperature,
                    "Wind_Speed_m/s": wind_speed,
                    "Precipitation_mm": precipitation,
                    "Weekday": weekday,
                    "Charging_Preferences": charging_pref,
                    "Weather_Conditions": weather,
                    "Charging_Load_kW": charging_load,
                }
            )

    return pd.DataFrame(records)


def prepare_dataset(df):
    prepared = df.copy()
    prepared.columns = prepared.columns.str.strip()

    if "Date_Time" in prepared.columns:
        prepared["Date_Time"] = pd.to_datetime(prepared["Date_Time"], errors="coerce")
        prepared["Date_Time"] = prepared["Date_Time"].ffill().bfill()

    numeric_cols = prepared.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        prepared[col] = prepared[col].fillna(prepared[col].median())

    categorical_cols = prepared.select_dtypes(exclude=[np.number]).columns.tolist()
    for col in categorical_cols:
        if col == "Date_Time":
            continue
        mode = prepared[col].mode(dropna=True)
        prepared[col] = prepared[col].fillna(mode.iloc[0] if not mode.empty else "Unknown")

    if "Date_Time" in prepared.columns:
        prepared["hour"] = prepared["Date_Time"].dt.hour
        prepared["day"] = prepared["Date_Time"].dt.day
        prepared["month"] = prepared["Date_Time"].dt.month
        prepared["weekday_index"] = prepared["Date_Time"].dt.weekday
        prepared["is_weekend"] = prepared["weekday_index"].isin([5, 6]).astype(int)
    else:
        prepared["hour"] = prepared.get("Session_Start_Hour", 0)
        prepared["day"] = 1
        prepared["month"] = 1
        prepared["weekday_index"] = prepared.get("Weekday", 0)
        prepared["is_weekend"] = 0

    prepared["peak_hour_flag"] = prepared["hour"].isin([7, 8, 9, 17, 18, 19, 20]).astype(int)
    prepared["current_energy_kwh"] = (
        prepared["Battery_Capacity_kWh"] * prepared["State_of_Charge_%"] / 100
    )
    prepared["estimated_trip_energy_kwh"] = (
        prepared["Distance_to_Destination_km"] * prepared["Energy_Consumption_Rate_kWh/km"]
    )
    safe_consumption = prepared["Energy_Consumption_Rate_kWh/km"].replace(0, np.nan).fillna(0.18)
    prepared["estimated_range_km"] = prepared["current_energy_kwh"] / safe_consumption
    prepared["energy_buffer_kwh"] = prepared["current_energy_kwh"] - prepared["estimated_trip_energy_kwh"]
    prepared["soc_gap_to_30"] = 30 - prepared["State_of_Charge_%"]
    prepared["queue_pressure"] = prepared["Queue_Time_mins"] / prepared["Charging_Rate_kW"].clip(lower=1)
    prepared["weather_severity"] = prepared["Weather_Conditions"].map(
        {"Clear": 0, "Cloudy": 1, "Rain": 2, "Storm": 3}
    ).fillna(1)
    prepared["road_severity"] = prepared["Road_Conditions"].map(
        {"Good": 0, "Average": 1, "Poor": 2}
    ).fillna(1)
    prepared["trip_urgency_score"] = (
        prepared["Traffic_Data"] * 0.35
        + prepared["road_severity"] * 0.2
        + prepared["weather_severity"] * 0.15
        + prepared["Distance_to_Destination_km"].clip(upper=100) / 100 * 0.3
    )
    prepared["charging_need_flag"] = (prepared["energy_buffer_kwh"] < 0).astype(int)

    return prepared


def dataset_overview(df):
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "avg_load_kw": float(df["Charging_Load_kW"].mean()) if "Charging_Load_kW" in df.columns else None,
        "avg_queue_mins": float(df["Queue_Time_mins"].mean()) if "Queue_Time_mins" in df.columns else None,
        "avg_trip_km": float(df["Distance_to_Destination_km"].mean()) if "Distance_to_Destination_km" in df.columns else None,
        "charge_need_rate": float(df["charging_need_flag"].mean()) if "charging_need_flag" in df.columns else None,
    }


def build_scenario_defaults(df):
    defaults = {}
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            defaults[col] = float(df[col].median())
    defaults["Road_Conditions"] = (
        df["Road_Conditions"].mode().iloc[0] if "Road_Conditions" in df.columns else "Good"
    )
    defaults["Weather_Conditions"] = (
        df["Weather_Conditions"].mode().iloc[0] if "Weather_Conditions" in df.columns else "Clear"
    )
    defaults["hour"] = int(df["hour"].median()) if "hour" in df.columns else 12
    defaults["weekday_index"] = int(df["weekday_index"].median()) if "weekday_index" in df.columns else 2
    defaults["Fleet_Schedule"] = int(df["Fleet_Schedule"].median()) if "Fleet_Schedule" in df.columns else 0
    defaults["Charging_Preferences"] = (
        int(df["Charging_Preferences"].median()) if "Charging_Preferences" in df.columns else 0
    )
    return defaults


def scenario_to_frame(scenario):
    frame = pd.DataFrame([scenario])
    if "Date_Time" not in frame.columns:
        base_day = int(frame.get("weekday_index", pd.Series([0])).iloc[0]) + 1
        hour = int(frame.get("hour", pd.Series([12])).iloc[0])
        frame["Date_Time"] = pd.Timestamp(2017, 1, min(base_day, 28), hour)
    return prepare_dataset(frame)
