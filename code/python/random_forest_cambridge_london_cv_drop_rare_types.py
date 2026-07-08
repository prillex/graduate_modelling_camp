from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


warnings.filterwarnings("ignore")


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

PREFERRED_DATA_PATH = DATA_DIR / "Cambridge_data_cleaned_new.csv"
FALLBACK_DATA_PATH = DATA_DIR / "Cambridge data_cleaned_before_dedup.csv"
MSOA_COORD_PATH = DATA_DIR / "msoa21_coordinates.csv"

TARGET = "price_sold"
TEST_SIZE = 0.20
RANDOM_STATE = 42
CV_SPLITS = 5
USE_LOG_TARGET = True
MIN_PROPERTY_TYPE_KEEP_COUNT = 200
TOP_IMPORTANCE_ROWS = 25

# Approximate city-centre coordinates.
CAMBRIDGE_CENTER_LAT = 52.2053
CAMBRIDGE_CENTER_LON = 0.1218
LONDON_CENTER_LAT = 51.5074
LONDON_CENTER_LON = -0.1278

RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_split": 4,
    "min_samples_leaf": 2,
    "max_features": 0.70,
    "bootstrap": True,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}

FULL_NUMERIC_FEATURES = [
    "num_bed_",
    "num_bath",
    "num_reception",
    "start_year",
    "start_month",
    "distance_to_cambridge_km",
    "distance_to_london_km",
    "Asian",
    "Black",
    "Mixed",
    "White",
    "Other",
    "Minors 0-18",
    "Adults 18-60",
    "Elders >60",
    "0-5km",
    "5-30km",
    ">30km",
    "Work from Home (0km)",
    "Living Offshore",
    "Low",
    "Medium",
    "High",
    "Other Qualification",
    "Work from Home",
    "City Public",
    "Rail",
    "Cycle",
    "Driving",
    "Foot",
    "Other Method",
]

CORE_NUMERIC_FEATURES = [
    "num_bed_",
    "num_bath",
    "num_reception",
    "distance_to_cambridge_km",
    "distance_to_london_km",
]

FULL_CATEGORICAL_FEATURES = ["property_type_model"]
CORE_CATEGORICAL_FEATURES = ["property_type_model"]

CODE_COLUMNS = ["msoa21", "msoa21cd", "msoa21_code", "MSOA21CD", "MSOA21_CD"]
LAT_COLUMNS = ["latitude", "lat", "LAT", "Latitude"]
LON_COLUMNS = ["longitude", "lon", "lng", "LONG", "Longitude"]


def choose_data_path():
    if PREFERRED_DATA_PATH.exists():
        return PREFERRED_DATA_PATH
    if FALLBACK_DATA_PATH.exists():
        return FALLBACK_DATA_PATH

    raise FileNotFoundError(
        "Could not find the modelling data file. Expected either "
        f"{PREFERRED_DATA_PATH} or {FALLBACK_DATA_PATH}."
    )


def first_existing_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def clean_input_data(df):
    required_columns = [TARGET, "start_date", "property_type", "msoa21"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df = df.copy()
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET, "start_date", "property_type", "msoa21"])
    df = df[df[TARGET] > 0].copy()

    df["start_year"] = df["start_date"].dt.year
    df["start_month"] = df["start_date"].dt.month

    return df.reset_index(drop=True)


def filter_rare_property_type_rows(df):
    property_counts = df["property_type"].fillna("Missing").astype(str).value_counts()
    kept_property_levels = property_counts[
        property_counts >= MIN_PROPERTY_TYPE_KEEP_COUNT
    ]
    removed_property_levels = property_counts[
        property_counts < MIN_PROPERTY_TYPE_KEEP_COUNT
    ]

    filtered_df = df[
        df["property_type"].fillna("Missing").astype(str).isin(kept_property_levels.index)
    ].copy()

    filter_info = {
        "original_rows": len(df),
        "final_rows": len(filtered_df),
        "removed_rows": len(df) - len(filtered_df),
        "kept_property_levels": kept_property_levels,
        "removed_property_levels": removed_property_levels,
    }

    return filtered_df.reset_index(drop=True), filter_info


def random_train_test_split(df):
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def load_msoa_coordinates():
    if not MSOA_COORD_PATH.exists():
        raise FileNotFoundError(
            f"Missing MSOA coordinate lookup file: {MSOA_COORD_PATH}"
        )

    coords = pd.read_csv(MSOA_COORD_PATH)
    code_col = first_existing_column(coords, CODE_COLUMNS)
    lat_col = first_existing_column(coords, LAT_COLUMNS)
    lon_col = first_existing_column(coords, LON_COLUMNS)

    if code_col is None or lat_col is None or lon_col is None:
        raise ValueError(
            f"{MSOA_COORD_PATH} must contain MSOA code, latitude, and longitude columns."
        )

    coords = coords.copy()
    coords["msoa21"] = coords[code_col].astype(str)
    coords["msoa_lat"] = pd.to_numeric(coords[lat_col], errors="coerce")
    coords["msoa_lon"] = pd.to_numeric(coords[lon_col], errors="coerce")

    return (
        coords[["msoa21", "msoa_lat", "msoa_lon"]]
        .dropna()
        .drop_duplicates(subset=["msoa21"], keep="first")
    )


def haversine_distance_km(lat, lon, centre_lat, centre_lon):
    earth_radius_km = 6371.0088
    lat1 = np.radians(lat)
    lon1 = np.radians(lon)
    lat2 = np.radians(centre_lat)
    lon2 = np.radians(centre_lon)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2

    return 2 * earth_radius_km * np.arcsin(np.sqrt(a))


def add_distance_features(train_df, test_df):
    coords = load_msoa_coordinates()

    train_df = train_df.merge(coords, on="msoa21", how="left")
    test_df = test_df.merge(coords, on="msoa21", how="left")

    for frame in [train_df, test_df]:
        frame["distance_to_cambridge_km"] = haversine_distance_km(
            frame["msoa_lat"],
            frame["msoa_lon"],
            CAMBRIDGE_CENTER_LAT,
            CAMBRIDGE_CENTER_LON,
        )
        frame["distance_to_london_km"] = haversine_distance_km(
            frame["msoa_lat"],
            frame["msoa_lon"],
            LONDON_CENTER_LAT,
            LONDON_CENTER_LON,
        )

    distance_info = {
        "coordinate_rows": len(coords),
        "train_missing_cambridge_distance": int(
            train_df["distance_to_cambridge_km"].isna().sum()
        ),
        "test_missing_cambridge_distance": int(
            test_df["distance_to_cambridge_km"].isna().sum()
        ),
        "train_missing_london_distance": int(
            train_df["distance_to_london_km"].isna().sum()
        ),
        "test_missing_london_distance": int(test_df["distance_to_london_km"].isna().sum()),
    }

    for col in ["distance_to_cambridge_km", "distance_to_london_km"]:
        fill_value = train_df[col].median()
        train_df[col] = train_df[col].fillna(fill_value)
        test_df[col] = test_df[col].fillna(fill_value)
        distance_info[f"median_{col}"] = fill_value

    return train_df, test_df, distance_info


def add_property_type_model(train_df, test_df):
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["property_type_model"] = train_df["property_type"].fillna("Missing").astype(str)
    test_df["property_type_model"] = test_df["property_type"].fillna("Missing").astype(str)

    return train_df, test_df


def available_columns(df, columns):
    return [col for col in columns if col in df.columns]


def build_xy(train_df, test_df, numeric_features, categorical_features):
    feature_columns = numeric_features + categorical_features

    X_train = train_df[feature_columns].copy()
    X_test = test_df[feature_columns].copy()

    for col in numeric_features:
        X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

    y_train = train_df[TARGET].copy()
    y_test = test_df[TARGET].copy()

    train_model_data = pd.concat([y_train.rename(TARGET), X_train], axis=1).dropna()
    test_model_data = pd.concat([y_test.rename(TARGET), X_test], axis=1).dropna()

    y_train = train_model_data[TARGET]
    X_train = train_model_data.drop(columns=[TARGET])
    y_test = test_model_data[TARGET]
    X_test = test_model_data.drop(columns=[TARGET])

    y_train_model = np.log(y_train) if USE_LOG_TARGET else y_train
    y_test_model = np.log(y_test) if USE_LOG_TARGET else y_test

    return X_train, X_test, y_train, y_test, y_train_model, y_test_model


def build_random_forest_pipeline(numeric_features, categorical_features):
    transformers = [("num", "passthrough", numeric_features)]
    if categorical_features:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    model = RandomForestRegressor(**RF_PARAMS)

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


def predict_price(model, X):
    predictions = model.predict(X)
    if USE_LOG_TARGET:
        return np.exp(predictions), predictions

    return predictions, np.log(np.maximum(predictions, 1))


def calculate_metrics(y_true_price, y_true_model, pred_price, pred_model):
    return {
        "MAE": mean_absolute_error(y_true_price, pred_price),
        "RMSE": np.sqrt(mean_squared_error(y_true_price, pred_price)),
        "MAPE_%": np.mean(np.abs((y_true_price - pred_price) / y_true_price)) * 100,
        "R2_log_price": r2_score(y_true_model, pred_model),
        "R2_price": r2_score(y_true_price, pred_price),
    }


def cross_validate_model(model_name, X_train, y_train_price, y_train_model, numeric_features, categorical_features):
    kfold = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    rows = []

    for fold, (fit_idx, val_idx) in enumerate(kfold.split(X_train), start=1):
        X_fit = X_train.iloc[fit_idx]
        X_val = X_train.iloc[val_idx]
        y_fit_model = y_train_model.iloc[fit_idx]
        y_val_price = y_train_price.iloc[val_idx]
        y_val_model = y_train_model.iloc[val_idx]

        model = build_random_forest_pipeline(numeric_features, categorical_features)
        model.fit(X_fit, y_fit_model)

        pred_price, pred_model = predict_price(model, X_val)
        metrics = calculate_metrics(y_val_price, y_val_model, pred_price, pred_model)
        rows.append({"model": model_name, "fold": fold, **metrics})

    return pd.DataFrame(rows)


def summarize_cv(cv_results):
    metric_columns = ["MAE", "RMSE", "MAPE_%", "R2_log_price", "R2_price"]
    summary = (
        cv_results.groupby("model")[metric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in summary.columns
    ]

    return summary.sort_values("RMSE_mean")


def fit_and_evaluate_test(model_name, X_train, X_test, y_train_model, y_test_price, y_test_model, numeric_features, categorical_features):
    model = build_random_forest_pipeline(numeric_features, categorical_features)
    model.fit(X_train, y_train_model)
    pred_price, pred_model = predict_price(model, X_test)
    metrics = calculate_metrics(y_test_price, y_test_model, pred_price, pred_model)

    return model, {"model": model_name, "n_features_before_encoding": X_test.shape[1], **metrics}


def feature_importance_table(model):
    preprocessor = model.named_steps["preprocess"]
    feature_names = preprocessor.get_feature_names_out()
    importances = model.named_steps["model"].feature_importances_

    return (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def print_metric_table(title, table):
    print(f"\n{title}")
    print(
        table.to_string(
            index=False,
            formatters={
                "MAE": "{:,.0f}".format,
                "RMSE": "{:,.0f}".format,
                "MAPE_%": "{:.2f}".format,
                "R2_log_price": "{:.4f}".format,
                "R2_price": "{:.4f}".format,
            },
        )
    )


def print_cv_summary(summary):
    print("\nCross-validation summary on training set:")
    print(
        summary.to_string(
            index=False,
            formatters={
                "MAE_mean": "{:,.0f}".format,
                "MAE_std": "{:,.0f}".format,
                "RMSE_mean": "{:,.0f}".format,
                "RMSE_std": "{:,.0f}".format,
                "MAPE_%_mean": "{:.2f}".format,
                "MAPE_%_std": "{:.2f}".format,
                "R2_log_price_mean": "{:.4f}".format,
                "R2_log_price_std": "{:.4f}".format,
                "R2_price_mean": "{:.4f}".format,
                "R2_price_std": "{:.4f}".format,
            },
        )
    )


def print_cv_folds(cv_results):
    print("\nCross-validation fold results:")
    print(
        cv_results.sort_values(["model", "fold"]).to_string(
            index=False,
            formatters={
                "MAE": "{:,.0f}".format,
                "RMSE": "{:,.0f}".format,
                "MAPE_%": "{:.2f}".format,
                "R2_log_price": "{:.4f}".format,
                "R2_price": "{:.4f}".format,
            },
        )
    )


def print_feature_importances(feature_importances):
    print("\nFeature importances from final holdout-test models:")
    for model_name, table in feature_importances.groupby("model", sort=False):
        print(f"\n{model_name}")
        print(table.to_string(index=False))


def print_feature_sets(experiments):
    print("\nFeature sets:")
    for experiment in experiments:
        print(f"\n{experiment['name']}")
        print("Numeric features:")
        for feature in experiment["numeric_features"]:
            print(f"- {feature}")
        print("Categorical features:")
        if experiment["categorical_features"]:
            for feature in experiment["categorical_features"]:
                print(f"- {feature}")
        else:
            print("- None")


def print_property_type_filtering(filter_info):
    print("\nProperty type filtering used in this script:")
    print(f"Rows before property-type filtering: {filter_info['original_rows']:,}")
    print(f"Rows removed:                       {filter_info['removed_rows']:,}")
    print(f"Rows after property-type filtering:  {filter_info['final_rows']:,}")
    print(
        "\nProperty types kept as separate model categories "
        f"(full-data count >= {MIN_PROPERTY_TYPE_KEEP_COUNT}):"
    )
    for property_type, count in filter_info["kept_property_levels"].items():
        print(f"- {property_type}: {count:,} rows")

    print(
        "\nProperty types removed from this experiment "
        f"(full-data count < {MIN_PROPERTY_TYPE_KEEP_COUNT}):"
    )
    removed = filter_info["removed_property_levels"]
    if removed.empty:
        print("- None")
    else:
        for property_type, count in removed.items():
            print(f"- {property_type}: {count:,} rows")


def main():
    data_path = choose_data_path()
    df = pd.read_csv(data_path)
    df = clean_input_data(df)
    df, filter_info = filter_rare_property_type_rows(df)

    train_df, test_df = random_train_test_split(df)
    train_df, test_df, distance_info = add_distance_features(train_df, test_df)
    train_df, test_df = add_property_type_model(train_df, test_df)

    full_numeric = available_columns(train_df, FULL_NUMERIC_FEATURES)
    core_numeric = available_columns(train_df, CORE_NUMERIC_FEATURES)

    experiments = [
        {
            "name": "RF Full Features + Cambridge/London Distance",
            "numeric_features": full_numeric,
            "categorical_features": FULL_CATEGORICAL_FEATURES,
        },
        {
            "name": "RF Core Distance Features",
            "numeric_features": core_numeric,
            "categorical_features": CORE_CATEGORICAL_FEATURES,
        },
    ]

    print(f"Data file:                       {data_path}")
    print(f"Rows after cleaning/filtering:   {len(df):,}")
    print(f"Split method:                    random")
    print(f"Random state:                    {RANDOM_STATE}")
    print(f"Training rows:                   {len(train_df):,}")
    print(f"Test rows:                       {len(test_df):,}")
    print(f"CV splits:                       {CV_SPLITS}")
    print(f"MSOA coordinate rows:            {distance_info['coordinate_rows']:,}")
    print(
        "Missing Cambridge distance:      "
        f"train {distance_info['train_missing_cambridge_distance']:,}, "
        f"test {distance_info['test_missing_cambridge_distance']:,}"
    )
    print(
        "Missing London distance:         "
        f"train {distance_info['train_missing_london_distance']:,}, "
        f"test {distance_info['test_missing_london_distance']:,}"
    )
    print(
        "Median distance to Cambridge:    "
        f"{distance_info['median_distance_to_cambridge_km']:.3f} km"
    )
    print(
        "Median distance to London:       "
        f"{distance_info['median_distance_to_london_km']:.3f} km"
    )
    print(f"Cambridge centre:                {CAMBRIDGE_CENTER_LAT}, {CAMBRIDGE_CENTER_LON}")
    print(f"London centre:                   {LONDON_CENTER_LAT}, {LONDON_CENTER_LON}")
    print(f"Target used for fitting:         {'log(price_sold)' if USE_LOG_TARGET else TARGET}")

    print_property_type_filtering(filter_info)
    print_feature_sets(experiments)

    cv_tables = []
    test_rows = []
    importance_tables = []

    for experiment in experiments:
        X_train, X_test, y_train, y_test, y_train_model, y_test_model = build_xy(
            train_df,
            test_df,
            experiment["numeric_features"],
            experiment["categorical_features"],
        )

        print(f"\nRunning cross-validation: {experiment['name']}")
        cv_result = cross_validate_model(
            experiment["name"],
            X_train,
            y_train,
            y_train_model,
            experiment["numeric_features"],
            experiment["categorical_features"],
        )
        cv_tables.append(cv_result)

        print(f"Fitting final holdout-test model: {experiment['name']}")
        model, test_result = fit_and_evaluate_test(
            experiment["name"],
            X_train,
            X_test,
            y_train_model,
            y_test,
            y_test_model,
            experiment["numeric_features"],
            experiment["categorical_features"],
        )
        test_rows.append(test_result)

        importances = feature_importance_table(model)
        importances.insert(0, "model", experiment["name"])
        importance_tables.append(importances)

        print("\nTop feature importances:")
        print(importances.head(TOP_IMPORTANCE_ROWS).to_string(index=False))

    cv_results = pd.concat(cv_tables, ignore_index=True)
    cv_summary = summarize_cv(cv_results)
    test_results = pd.DataFrame(test_rows).sort_values("RMSE")
    feature_importances = pd.concat(importance_tables, ignore_index=True)

    print_cv_folds(cv_results)
    print_cv_summary(cv_summary)
    print_metric_table("Holdout test-set performance:", test_results)
    print_feature_importances(feature_importances)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cv_results.to_csv(
        OUTPUT_DIR / "rf_cambridge_london_drop_rare_types_cv_folds.csv",
        index=False,
    )
    cv_summary.to_csv(
        OUTPUT_DIR / "rf_cambridge_london_drop_rare_types_cv_summary.csv",
        index=False,
    )
    test_results.to_csv(
        OUTPUT_DIR / "rf_cambridge_london_drop_rare_types_test_results.csv",
        index=False,
    )
    feature_importances.to_csv(
        OUTPUT_DIR / "rf_cambridge_london_drop_rare_types_feature_importances.csv",
        index=False,
    )

    print(
        "\nSaved CV fold results to "
        f"{OUTPUT_DIR / 'rf_cambridge_london_drop_rare_types_cv_folds.csv'}"
    )
    print(
        "Saved CV summary to "
        f"{OUTPUT_DIR / 'rf_cambridge_london_drop_rare_types_cv_summary.csv'}"
    )
    print(
        "Saved test results to "
        f"{OUTPUT_DIR / 'rf_cambridge_london_drop_rare_types_test_results.csv'}"
    )
    print(
        "Saved feature importances to "
        f"{OUTPUT_DIR / 'rf_cambridge_london_drop_rare_types_feature_importances.csv'}"
    )


if __name__ == "__main__":
    main()
