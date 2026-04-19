import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from data_utils import (
    build_scenario_defaults,
    dataset_overview,
    generate_synthetic_dataset,
    load_dataset,
    prepare_dataset,
    scenario_to_frame,
)
from modeling import DEFAULT_TARGETS, predict_target, top_feature_correlations, train_forecasting_suite
from optimizer import optimizer_summary


st.set_page_config(page_title="Smart EV Fleet Optimizer", layout="wide")


@st.cache_data
def load_prepared_dataset(data_mode, uploaded_file, sim_days, sim_vehicles, sim_stations, sim_seed):
    if data_mode == "Digital twin":
        raw_df = generate_synthetic_dataset(
            days=sim_days,
            vehicles=sim_vehicles,
            stations=sim_stations,
            seed=sim_seed,
        )
    else:
        raw_df = load_dataset(uploaded_file)
    if raw_df is None:
        return None, None
    return raw_df, prepare_dataset(raw_df)


@st.cache_resource
def train_target_bundle(prepared_df, target):
    return train_forecasting_suite(prepared_df, target)


def render_overview(prepared_df):
    overview = dataset_overview(prepared_df)
    st.subheader("Fleet pulse")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Records", f"{overview['rows']:,}")
    col2.metric("Avg charging load", f"{overview['avg_load_kw']:.1f} kW")
    col3.metric("Avg queue time", f"{overview['avg_queue_mins']:.1f} min")
    col4.metric("Charge-needed trips", f"{overview['charge_need_rate'] * 100:.1f}%")

    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("**Hourly charging demand**")
        hourly = prepared_df.groupby("hour")["Charging_Load_kW"].mean()
        fig, ax = plt.subplots(figsize=(9, 4))
        hourly.plot(ax=ax, marker="o", color="#0f766e")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Average load (kW)")
        ax.set_title("Charging load pattern through the day")
        ax.grid(alpha=0.2)
        st.pyplot(fig)

    with right:
        st.markdown("**Weather and road stress**")
        stress = (
            prepared_df.groupby(["Weather_Conditions", "Road_Conditions"])["Queue_Time_mins"]
            .mean()
            .unstack(fill_value=0)
        )
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        im = ax2.imshow(stress.values, cmap="YlOrRd", aspect="auto")
        ax2.set_xticks(range(len(stress.columns)))
        ax2.set_xticklabels(stress.columns)
        ax2.set_yticks(range(len(stress.index)))
        ax2.set_yticklabels(stress.index)
        ax2.set_title("Average queue time by weather and road condition")
        fig2.colorbar(im, ax=ax2)
        st.pyplot(fig2)

    st.markdown("**Operational insights**")
    insights = [
        f"Average trip distance is {prepared_df['Distance_to_Destination_km'].mean():.1f} km.",
        f"Average estimated trip energy is {prepared_df['estimated_trip_energy_kwh'].mean():.1f} kWh.",
        f"Peak-hour charging load reaches {prepared_df.groupby('hour')['Charging_Load_kW'].mean().max():.1f} kW.",
        f"Storm conditions raise average queue time to {prepared_df.loc[prepared_df['Weather_Conditions'].eq('Storm'), 'Queue_Time_mins'].mean():.1f} min."
        if prepared_df["Weather_Conditions"].eq("Storm").any()
        else "No storm-condition rows are available for queue benchmarking.",
    ]
    for insight in insights:
        st.write(f"- {insight}")


def render_simulator_status(data_mode, prepared_df):
    st.subheader("Digital fleet status")
    station_load = (
        prepared_df.groupby("Charging_Station_ID")
        .agg(
            avg_load_kw=("Charging_Load_kW", "mean"),
            avg_queue_mins=("Queue_Time_mins", "mean"),
            active_vehicles=("Vehicle_ID", "nunique"),
        )
        .reset_index()
        .sort_values("avg_load_kw", ascending=False)
    )

    fleet_soc = prepared_df.groupby("Vehicle_ID")["State_of_Charge_%"].last().sort_values()
    left, right = st.columns(2)
    with left:
        st.markdown(f"**Mode: {data_mode}**")
        st.dataframe(
            station_load.style.format(
                {"avg_load_kw": "{:.1f}", "avg_queue_mins": "{:.1f}"}
            ),
            use_container_width=True,
        )
    with right:
        fig, ax = plt.subplots(figsize=(7, 4))
        fleet_soc.head(20).plot(kind="barh", ax=ax, color="#f97316")
        ax.set_xlabel("State of charge (%)")
        ax.set_ylabel("Vehicle")
        ax.set_title("Lowest-SOC vehicles right now")
        st.pyplot(fig)


def render_optimizer(prepared_df):
    st.subheader("Scenario optimizer")
    defaults = build_scenario_defaults(prepared_df)

    col1, col2, col3 = st.columns(3)
    with col1:
        battery_capacity = st.slider("Battery capacity (kWh)", 30.0, 120.0, float(defaults["Battery_Capacity_kWh"]), 1.0)
        soc = st.slider("State of charge (%)", 1, 100, int(defaults["State_of_Charge_%"]))
        distance = st.slider("Distance to destination (km)", 1.0, 120.0, float(defaults["Distance_to_Destination_km"]), 1.0)
        consumption = st.slider(
            "Energy consumption rate (kWh/km)",
            0.10,
            0.35,
            float(defaults["Energy_Consumption_Rate_kWh/km"]),
            0.01,
        )
        charging_rate = st.slider("Charging rate (kW)", 10.0, 120.0, float(defaults["Charging_Rate_kW"]), 1.0)

    with col2:
        queue = st.slider("Expected queue time (min)", 0.0, 60.0, float(defaults["Queue_Time_mins"]), 0.5)
        traffic = st.slider("Traffic level", 0, 11, int(defaults["Traffic_Data"]))
        road_options = sorted(prepared_df["Road_Conditions"].dropna().unique())
        weather_options = sorted(prepared_df["Weather_Conditions"].dropna().unique())
        road = st.selectbox(
            "Road condition",
            road_options,
            index=road_options.index(defaults["Road_Conditions"]) if defaults["Road_Conditions"] in road_options else 0,
        )
        weather = st.selectbox(
            "Weather condition",
            weather_options,
            index=weather_options.index(defaults["Weather_Conditions"]) if defaults["Weather_Conditions"] in weather_options else 0,
        )
        hour = st.slider("Planned departure hour", 0, 23, int(defaults["hour"]))

    with col3:
        fleet_size = st.slider("Fleet size", 10, 200, int(defaults["Fleet_Size"]))
        station_capacity = st.slider("Station capacity (EVs)", 1, 8, int(prepared_df["Station_Capacity_EV"].median()))
        time_priority = st.slider("Time priority", 0.5, 2.0, 1.2, 0.1)
        battery_priority = st.slider("Battery safety priority", 0.5, 2.0, 1.4, 0.1)
        reliability_priority = st.slider("Reliability priority", 0.5, 2.0, 1.1, 0.1)

    scenario = {
        "Battery_Capacity_kWh": battery_capacity,
        "State_of_Charge_%": soc,
        "Distance_to_Destination_km": distance,
        "Energy_Consumption_Rate_kWh/km": consumption,
        "Charging_Rate_kW": charging_rate,
        "Queue_Time_mins": queue,
        "Traffic_Data": traffic,
        "Road_Conditions": road,
        "Weather_Conditions": weather,
        "hour": hour,
        "weekday_index": int(defaults["weekday_index"]),
        "Fleet_Size": fleet_size,
        "Station_Capacity_EV": station_capacity,
        "Fleet_Schedule": defaults["Fleet_Schedule"],
        "Charging_Preferences": defaults["Charging_Preferences"],
        "Vehicle_ID": int(prepared_df["Vehicle_ID"].median()) if "Vehicle_ID" in prepared_df.columns else 1,
        "Charging_Station_ID": int(prepared_df["Charging_Station_ID"].median()) if "Charging_Station_ID" in prepared_df.columns else 1,
        "Current_Latitude": float(prepared_df["Current_Latitude"].median()),
        "Current_Longitude": float(prepared_df["Current_Longitude"].median()),
        "Destination_Latitude": float(prepared_df["Destination_Latitude"].median()),
        "Destination_Longitude": float(prepared_df["Destination_Longitude"].median()),
        "Time_Spent_Charging_mins": float(prepared_df["Time_Spent_Charging_mins"].median()),
        "Energy_Drawn_kWh": float(prepared_df["Energy_Drawn_kWh"].median()),
        "Session_Start_Hour": int(prepared_df["Session_Start_Hour"].median()),
        "Temperature_C": float(prepared_df["Temperature_C"].median()),
        "Wind_Speed_m/s": float(prepared_df["Wind_Speed_m/s"].median()),
        "Precipitation_mm": float(prepared_df["Precipitation_mm"].median()),
        "Weekday": int(prepared_df["Weekday"].median()),
        "Charging_Load_kW": float(prepared_df["Charging_Load_kW"].median()),
        "time_priority": time_priority,
        "battery_priority": battery_priority,
        "reliability_priority": reliability_priority,
    }
    scenario_df = scenario_to_frame(scenario)

    load_bundle = train_target_bundle(prepared_df, "Charging_Load_kW")
    queue_bundle = train_target_bundle(prepared_df, "Queue_Time_mins")
    predicted_load = predict_target(load_bundle, scenario_df)
    predicted_queue = predict_target(queue_bundle, scenario_df)
    scenario["Queue_Time_mins"] = predicted_queue

    summary = optimizer_summary(prepared_df, scenario, predicted_load, predicted_queue)
    best_strategy = summary["best_strategy"]

    st.markdown("**Recommended action plan**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Charge needed", f"{best_strategy['charge_needed_kwh']:.2f} kWh")
    c2.metric("Charging time", f"{best_strategy['charging_minutes']:.1f} min")
    c3.metric("Predicted load", f"{summary['predicted_load_kw']:.1f} kW")
    c4.metric("Available range", f"{summary['estimated_range_km']:.1f} km")

    st.success(
        f"Best strategy: {best_strategy['strategy']} with optimization score {best_strategy['optimization_score']:.1f}."
    )
    st.write(
        {
            "Current energy (kWh)": round(summary["current_energy_kwh"], 2),
            "Trip energy needed (kWh)": round(summary["trip_energy_kwh"], 2),
            "Predicted queue time (min)": round(summary["predicted_queue_mins"], 2),
            "Post-trip energy buffer (kWh)": round(best_strategy["buffer_after_trip_kwh"], 2),
        }
    )

    left, right = st.columns(2)
    with left:
        st.markdown("**Charging strategy comparison**")
        st.dataframe(
            summary["strategy_table"].style.format(
                {
                    "charge_needed_kwh": "{:.2f}",
                    "charging_minutes": "{:.1f}",
                    "drive_minutes": "{:.1f}",
                    "queue_minutes": "{:.1f}",
                    "buffer_after_trip_kwh": "{:.2f}",
                    "optimization_score": "{:.1f}",
                }
            ),
            use_container_width=True,
        )

    with right:
        st.markdown("**Best historical operating windows**")
        st.dataframe(
            summary["route_candidates"].style.format(
                {
                    "Distance_to_Destination_km": "{:.2f}",
                    "Queue_Time_mins": "{:.1f}",
                    "candidate_drive_mins": "{:.1f}",
                    "candidate_total_mins": "{:.1f}",
                }
            ),
            use_container_width=True,
        )


def render_forecast_lab(prepared_df):
    st.subheader("Forecast lab")
    valid_targets = [target for target in DEFAULT_TARGETS if target in prepared_df.columns]
    selected_target = st.selectbox("Target to forecast", valid_targets)
    bundle = train_target_bundle(prepared_df, selected_target)

    st.info(
        f"Forecasts use a time-ordered split with {bundle['train_rows']:,} training rows and {bundle['test_rows']:,} test rows."
    )
    st.dataframe(
        bundle["metrics"].style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "R2": "{:.3f}"}),
        use_container_width=True,
    )
    st.success(f"Best model for {selected_target}: {bundle['best_model_name']}")

    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(7, 5))
        sample = bundle["prediction_frame"].head(400)
        ax.scatter(sample["actual"], sample["predicted"], alpha=0.55, color="#2563eb")
        diagonal_min = min(sample["actual"].min(), sample["predicted"].min())
        diagonal_max = max(sample["actual"].max(), sample["predicted"].max())
        ax.plot([diagonal_min, diagonal_max], [diagonal_min, diagonal_max], "r--")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{selected_target}: actual vs predicted")
        st.pyplot(fig)

    with right:
        st.markdown("**Top numeric drivers**")
        driver_df = bundle["top_drivers"]
        if driver_df.empty:
            st.info("No numeric feature correlations are available for this target.")
        else:
            fig2, ax2 = plt.subplots(figsize=(7, 5))
            ax2.barh(driver_df["feature"], driver_df["correlation"], color="#0f766e")
            ax2.set_xlabel("Correlation with target")
            ax2.set_title(f"Strongest numeric drivers for {selected_target}")
            ax2.invert_yaxis()
            st.pyplot(fig2)


def render_data_explorer(raw_df, prepared_df):
    st.subheader("Data explorer")
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        weather_filter = st.multiselect(
            "Weather filter",
            sorted(prepared_df["Weather_Conditions"].dropna().unique()),
            default=sorted(prepared_df["Weather_Conditions"].dropna().unique()),
        )
    with filter_col2:
        road_filter = st.multiselect(
            "Road filter",
            sorted(prepared_df["Road_Conditions"].dropna().unique()),
            default=sorted(prepared_df["Road_Conditions"].dropna().unique()),
        )
    with filter_col3:
        hour_range = st.slider("Hour range", 0, 23, (0, 23))

    filtered = prepared_df[
        prepared_df["Weather_Conditions"].isin(weather_filter)
        & prepared_df["Road_Conditions"].isin(road_filter)
        & prepared_df["hour"].between(hour_range[0], hour_range[1])
    ]

    st.write(f"Showing {len(filtered):,} rows after filters.")
    st.dataframe(filtered.head(50), use_container_width=True)

    with st.expander("Raw dataset preview"):
        st.dataframe(raw_df.head(25), use_container_width=True)

    with st.expander("Prepared data summary"):
        st.write(filtered.describe(include="all"))

    with st.expander("Missing values in raw dataset"):
        missing = raw_df.isnull().sum().sort_values(ascending=False)
        st.write(missing[missing > 0])


def main():
    st.title("Smart EV Fleet Optimizer")
    st.markdown(
        "An end-to-end EV fleet decision-support dashboard for forecasting charging demand, stress-testing trip scenarios, and recommending smarter charging actions."
    )

    st.sidebar.markdown("### Data source")
    data_mode = st.sidebar.radio(
        "Choose operating mode",
        ["Digital twin", "Upload CSV", "Use local CSV"],
        index=0,
    )
    uploaded_file = None
    sim_days = 21
    sim_vehicles = 24
    sim_stations = 6
    sim_seed = 42

    if data_mode == "Digital twin":
        sim_days = st.sidebar.slider("Simulation days", 7, 60, 21)
        sim_vehicles = st.sidebar.slider("Vehicles", 8, 120, 24)
        sim_stations = st.sidebar.slider("Stations", 2, 20, 6)
        sim_seed = st.sidebar.slider("Scenario seed", 1, 999, 42)
    elif data_mode == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader("Upload EV dataset", type=["csv"])

    raw_df, prepared_df = load_prepared_dataset(
        data_mode,
        uploaded_file,
        sim_days,
        sim_vehicles,
        sim_stations,
        sim_seed,
    )

    if raw_df is None or prepared_df is None:
        st.warning(
            "No data source is available. Use Digital twin mode, upload a CSV, or place `ev_charging_dataset.csv` in the project folder."
        )
        return

    st.sidebar.markdown("### Dataset")
    st.sidebar.write(f"Rows: {len(prepared_df):,}")
    st.sidebar.write(f"Columns: {prepared_df.shape[1]}")
    st.sidebar.write(
        f"Date range: {prepared_df['Date_Time'].min().date()} to {prepared_df['Date_Time'].max().date()}"
        if "Date_Time" in prepared_df.columns and prepared_df["Date_Time"].notna().any()
        else "Date range unavailable"
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Executive Overview", "Scenario Optimizer", "Forecast Lab", "Data Explorer"]
    )

    with tab1:
        render_overview(prepared_df)
        render_simulator_status(data_mode, prepared_df)
    with tab2:
        render_optimizer(prepared_df)
    with tab3:
        render_forecast_lab(prepared_df)
    with tab4:
        render_data_explorer(raw_df, prepared_df)


if __name__ == "__main__":
    main()
