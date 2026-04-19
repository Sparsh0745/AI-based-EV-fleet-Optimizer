import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_TARGETS = [
    "Charging_Load_kW",
    "Energy_Drawn_kWh",
    "Queue_Time_mins",
]

LEAKY_FEATURE_MAP = {
    "Charging_Load_kW": ["Energy_Drawn_kWh", "Time_Spent_Charging_mins"],
    "Energy_Drawn_kWh": ["Charging_Load_kW", "Time_Spent_Charging_mins"],
    "Queue_Time_mins": ["Charging_Load_kW", "Time_Spent_Charging_mins"],
}


def _time_ordered_split(df, test_fraction=0.2):
    ordered = df.copy()
    if "Date_Time" in ordered.columns and ordered["Date_Time"].notna().any():
        ordered = ordered.sort_values("Date_Time").reset_index(drop=True)
    split_idx = max(int(len(ordered) * (1 - test_fraction)), 1)
    train_df = ordered.iloc[:split_idx].copy()
    test_df = ordered.iloc[split_idx:].copy()
    if test_df.empty:
        test_df = ordered.iloc[-1:].copy()
        train_df = ordered.iloc[:-1].copy()
    return train_df, test_df


def _build_feature_lists(df, target):
    blocked = {
        target,
        "Date_Time",
    }
    blocked.update(LEAKY_FEATURE_MAP.get(target, []))

    feature_cols = [col for col in df.columns if col not in blocked]
    numeric_features = [
        col for col in feature_cols if pd.api.types.is_numeric_dtype(df[col])
    ]
    categorical_features = [
        col for col in feature_cols if not pd.api.types.is_numeric_dtype(df[col])
    ]
    return feature_cols, numeric_features, categorical_features


def _build_preprocessor(numeric_features, categorical_features):
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )


def train_forecasting_suite(df, target):
    modeling_df = df.dropna(subset=[target]).copy()
    if len(modeling_df) > 8000:
        modeling_df = modeling_df.sort_values("Date_Time").tail(8000).reset_index(drop=True)
    train_df, test_df = _time_ordered_split(modeling_df, test_fraction=0.2)
    feature_cols, numeric_features, categorical_features = _build_feature_lists(modeling_df, target)

    X_train = train_df[feature_cols]
    y_train = train_df[target]
    X_test = test_df[feature_cols]
    y_test = test_df[target]

    preprocessor = _build_preprocessor(numeric_features, categorical_features)
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=20,
            max_depth=6,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1,
        ),
    }

    metrics_rows = []
    fitted_models = {}
    best_name = None
    best_r2 = -np.inf
    best_predictions = None

    for name, estimator in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = float(np.sqrt(mse))
        r2 = r2_score(y_test, preds)

        metrics_rows.append(
            {
                "Model": name,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
            }
        )
        fitted_models[name] = pipeline

        if r2 > best_r2:
            best_name = name
            best_r2 = r2
            best_predictions = preds

    metrics_df = pd.DataFrame(metrics_rows).sort_values("R2", ascending=False).reset_index(drop=True)
    prediction_frame = pd.DataFrame(
        {
            "actual": y_test.values,
            "predicted": best_predictions,
        }
    )
    top_drivers = top_feature_correlations(modeling_df, target)

    return {
        "metrics": metrics_df,
        "best_model_name": best_name,
        "best_model": fitted_models[best_name],
        "prediction_frame": prediction_frame,
        "feature_columns": feature_cols,
        "top_drivers": top_drivers,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
    }


def top_feature_correlations(df, target, top_n=8):
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    if target not in numeric_df.columns:
        return pd.DataFrame(columns=["feature", "correlation"])

    corr = numeric_df.corr(numeric_only=True)[target].drop(labels=[target]).dropna()
    corr = corr.reindex(corr.abs().sort_values(ascending=False).index)
    corr = corr.head(top_n)
    return pd.DataFrame(
        {
            "feature": corr.index,
            "correlation": corr.values,
        }
    )


def predict_target(model_bundle, scenario_df):
    model = model_bundle["best_model"]
    feature_cols = model_bundle["feature_columns"]
    X = scenario_df.reindex(columns=feature_cols)
    return float(model.predict(X)[0])
