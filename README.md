 Live Link 
 https://ai-based-ev-fleet-optimizer.onrender.com
# Smart EV Fleet Optimizer

This project upgrades the original EV charging prototype into a smarter decision-support dashboard for fleet operations.
It can now run as a self-contained digital twin without depending on any external dataset.

## What the app now does

- Forecasts key operational targets with a time-aware validation split
- Generates a synthetic EV fleet, charging stations, weather, traffic, and trip demand for digital-twin simulation
- Recommends charging strategies based on trip energy, SOC, queue time, traffic, and reliability priorities
- Surfaces the best historical operating windows for similar scenarios
- Provides an executive view of charging load, queue stress, and fleet health
- Includes a data explorer for filtered analysis and quality checks

## Project structure

- `ev_fleet_dashboard.py` - Streamlit application
- `data_utils.py` - dataset loading, cleaning, and feature engineering
- `modeling.py` - forecasting pipelines and evaluation helpers
- `optimizer.py` - optimization scoring and scenario recommendations
- `ev_charging_dataset.csv` - source dataset

## Setup

1. Open a terminal in the project folder:
   ```bash
   cd "c:\Users\Admin\OneDrive\Desktop\project_11"
   ```
2. Activate the virtual environment if you want to use the local one:
   ```bash
   .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Run

```bash
streamlit run ev_fleet_dashboard.py
```

The app supports three modes:

- `Digital twin`: fully synthetic simulation, no dataset required
- `Upload CSV`: use your own dataset
- `Use local CSV`: automatically load `ev_charging_dataset.csv` if present

## Recommended next data upgrades

The current dataset is strong enough for forecasting and scenario scoring, but a more realistic fleet optimizer would improve further with:

- multiple vehicles instead of a single `Vehicle_ID`
- multiple charging stations instead of a single `Charging_Station_ID`
- station pricing and availability over time
- real route alternatives and detour distances
- trip priority or SLA constraints
