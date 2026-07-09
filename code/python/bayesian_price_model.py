from pathlib import Path
import argparse
import os
import tempfile
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


warnings.filterwarnings("ignore")


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PREFERRED_DATA_PATH = DATA_DIR / "Cambridge_data_cleaned_new.csv"
FALLBACK_DATA_PATH = DATA_DIR / "Cambridge data_cleaned_before_dedup.csv"
MSOA_COORD_PATH = DATA_DIR / "msoa21_coordinates.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
PLOT_DIR = OUTPUT_DIR / "bayesian_plots"
PYTENSOR_CACHE_DIR = Path(tempfile.gettempdir()) / "pt_cam"
NUMBA_CACHE_DIR = PYTENSOR_CACHE_DIR / "numba"

PYTENSOR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
NUMBA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
existing_pytensor_flags = os.environ.get("PYTENSOR_FLAGS", "")
if "base_compiledir" not in existing_pytensor_flags:
    cache_flag = f"base_compiledir={PYTENSOR_CACHE_DIR.as_posix()}"
    os.environ["PYTENSOR_FLAGS"] = (
        f"{existing_pytensor_flags},{cache_flag}"
        if existing_pytensor_flags
        else cache_flag
    )
os.environ.setdefault("NUMBA_CACHE_DIR", NUMBA_CACHE_DIR.as_posix())

TARGET = "price_sold"
TEST_SIZE = 0.20
RANDOM_STATE = 42

# MCMC on all 47k training rows can be slow. This keeps the first Bayesian run
# practical while preserving the same split and preprocessing as the final RF.
MAX_TRAIN_ROWS_FOR_MCMC = 8000
POSTERIOR_PREDICTIVE_DRAWS = 1000
PRIOR_DRAWS = 500

DRAWS = 1000
TUNE = 1000
CHAINS = 4
CORES = 2
TARGET_ACCEPT = 0.90
STUDENT_T_NU = 4.0

MIN_PROPERTY_TYPE_KEEP_COUNT = 200
CAMBRIDGE_CENTER_LAT = 52.2053
CAMBRIDGE_CENTER_LON = 0.1218
LONDON_CENTER_LAT = 51.5074
LONDON_CENTER_LON = -0.1278

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

UPDATED_DATA_COLUMN_RENAMES = {
    "Minors.0.18": "Minors 0-18",
    "Adults.18.60": "Adults 18-60",
    "Elders..60": "Elders >60",
    "X0.5km": "0-5km",
    "X5.30km": "5-30km",
    "X.30km": ">30km",
    "Work.from.Home..0km.": "Work from Home (0km)",
    "Living.Offshore": "Living Offshore",
    "Other.Qualification": "Other Qualification",
    "City.Public": "City Public",
    "Other.Method": "Other Method",
}

CODE_COLUMNS = ["msoa21", "msoa21cd", "msoa21_code", "MSOA21CD", "MSOA21_CD"]
LAT_COLUMNS = ["latitude", "lat", "LAT", "Latitude"]
LON_COLUMNS = ["longitude", "lon", "lng", "LONG", "Longitude"]


def require_bayesian_packages():
    try:
        import arviz as az
        import pymc as pm
    except ImportError as exc:
        raise SystemExit(
            "This script needs PyMC and ArviZ for MCMC sampling.\n"
            "Install them first, for example:\n"
            "    python -m pip install pymc arviz\n"
        ) from exc

    return pm, az


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


def random_train_test_split(df):
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Bayesian house-price model using the same preprocessing as the "
            "final RF Cambridge/London distance experiment."
        )
    )
    parser.add_argument(
        "--feature-set",
        choices=["full", "core"],
        default="full",
        help="Use the RF full feature set or the RF core distance feature set.",
    )
    parser.add_argument("--max-train-rows", type=int, default=MAX_TRAIN_ROWS_FOR_MCMC)
    parser.add_argument("--draws", type=int, default=DRAWS)
    parser.add_argument("--tune", type=int, default=TUNE)
    parser.add_argument("--chains", type=int, default=CHAINS)
    parser.add_argument("--cores", type=int, default=CORES)
    parser.add_argument(
        "--posterior-predictive-draws",
        type=int,
        default=POSTERIOR_PREDICTIVE_DRAWS,
    )
    parser.add_argument("--prior-draws", type=int, default=PRIOR_DRAWS)
    parser.add_argument("--target-accept", type=float, default=TARGET_ACCEPT)
    parser.add_argument(
        "--skip-prior-predictive",
        action="store_true",
        help="Skip prior predictive sampling and prior summary output.",
    )
    parser.add_argument(
        "--progressbar",
        action="store_true",
        help="Show PyMC's progress bar. Off by default to avoid Windows encoding issues.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip Bayesian diagnostic and prediction plots.",
    )
    parser.add_argument(
        "--plot-sample-size",
        type=int,
        default=400,
        help="Maximum number of test-set rows shown in interval/scatter plots.",
    )

    return parser.parse_args()


def sample_training_rows(train_df, max_train_rows):
    if max_train_rows is None or max_train_rows <= 0 or len(train_df) <= max_train_rows:
        return train_df.reset_index(drop=True)

    return (
        train_df.sample(max_train_rows, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )


def encode_property_types(train_df, test_df, level_source_df):
    property_levels = sorted(level_source_df["property_type_model"].astype(str).unique())
    property_lookup = {level: i for i, level in enumerate(property_levels)}

    train_property_idx = train_df["property_type_model"].astype(str).map(property_lookup)
    test_property_idx = test_df["property_type_model"].astype(str).map(property_lookup)

    if train_property_idx.isna().any():
        raise ValueError("MCMC training rows contain property types not seen in full train.")
    if test_property_idx.isna().any():
        unknown = sorted(
            test_df.loc[test_property_idx.isna(), "property_type_model"]
            .astype(str)
            .unique()
        )
        raise ValueError(f"Test set contains property types not seen in training data: {unknown}")

    return {
        "property_levels": property_levels,
        "train_property_idx": train_property_idx.astype(int).to_numpy(),
        "test_property_idx": test_property_idx.astype(int).to_numpy(),
    }


def standardize_train_test(train_df, test_df, numeric_features, scale_source_df):
    X_train = train_df[numeric_features].apply(pd.to_numeric, errors="coerce")
    X_test = test_df[numeric_features].apply(pd.to_numeric, errors="coerce")
    X_scale = scale_source_df[numeric_features].apply(pd.to_numeric, errors="coerce")

    x_means = X_scale.mean()
    x_stds = X_scale.std().replace(0, 1)

    X_train = X_train.fillna(x_means)
    X_test = X_test.fillna(x_means)

    X_train_std = ((X_train - x_means) / x_stds).to_numpy(dtype=float)
    X_test_std = ((X_test - x_means) / x_stds).to_numpy(dtype=float)

    y_scale_log = np.log(scale_source_df[TARGET].to_numpy(dtype=float))
    y_train_log = np.log(train_df[TARGET].to_numpy(dtype=float))
    y_test_log = np.log(test_df[TARGET].to_numpy(dtype=float))
    y_mean = y_scale_log.mean()
    y_std = y_scale_log.std()

    y_train_std = (y_train_log - y_mean) / y_std
    x_summary = pd.DataFrame(
        {
            "feature": numeric_features,
            "mean_used_for_standardization": x_means.reindex(numeric_features).to_numpy(),
            "std_used_for_standardization": x_stds.reindex(numeric_features).to_numpy(),
        }
    )

    return {
        "numeric_features": numeric_features,
        "X_train_std": X_train_std,
        "X_test_std": X_test_std,
        "y_train_std": y_train_std,
        "y_train_log": y_train_log,
        "y_test_log": y_test_log,
        "y_mean": y_mean,
        "y_std": y_std,
        "y_test_price": test_df[TARGET].to_numpy(dtype=float),
        "x_summary": x_summary,
    }


def build_model(pm, model_data, category_data):
    coords = {
        "feature": model_data["numeric_features"],
        "property_type": category_data["property_levels"],
    }

    X_train = model_data["X_train_std"]
    y_train = model_data["y_train_std"]
    property_idx = category_data["train_property_idx"]

    with pm.Model(coords=coords) as model:
        intercept = pm.Normal("intercept", mu=0, sigma=1)
        beta = pm.Normal("beta", mu=0, sigma=0.5, dims="feature")

        sigma_property = pm.HalfNormal("sigma_property", sigma=0.5)
        property_raw = pm.Normal("property_raw", mu=0, sigma=1, dims="property_type")
        property_effect = pm.Deterministic(
            "property_effect",
            property_raw * sigma_property,
            dims="property_type",
        )

        sigma = pm.HalfNormal("sigma", sigma=1)

        mu = (
            intercept
            + pm.math.dot(X_train, beta)
            + property_effect[property_idx]
        )

        pm.StudentT(
            "log_price_observed",
            nu=STUDENT_T_NU,
            mu=mu,
            sigma=sigma,
            observed=y_train,
        )

    return model


def get_posterior_dataset(idata):
    posterior_group = idata.posterior
    if hasattr(posterior_group, "ds"):
        posterior_group = posterior_group.ds

    return posterior_group


def flatten_posterior(idata, rng, posterior_predictive_draws):
    posterior_group = get_posterior_dataset(idata)
    posterior = posterior_group.stack(sample=("chain", "draw"))
    n_samples = posterior.sizes["sample"]
    n_keep = min(posterior_predictive_draws, n_samples)
    keep_idx = rng.choice(n_samples, size=n_keep, replace=False)
    kept = posterior.isel(sample=keep_idx)

    return {
        "intercept": kept["intercept"].values,
        "beta": kept["beta"].transpose("sample", "feature").values,
        "property_effect": kept["property_effect"]
        .transpose("sample", "property_type")
        .values,
        "sigma": kept["sigma"].values,
    }


def posterior_predictive_log_prices(posterior, model_data, category_data, rng):
    X_test = model_data["X_test_std"]
    property_idx = category_data["test_property_idx"]

    mu_std = (
        posterior["intercept"][:, None]
        + posterior["beta"] @ X_test.T
        + posterior["property_effect"][:, property_idx]
    )

    noise = rng.standard_t(STUDENT_T_NU, size=mu_std.shape) * posterior["sigma"][:, None]
    y_pred_std = mu_std + noise

    return model_data["y_mean"] + model_data["y_std"] * y_pred_std


def evaluate_predictions(y_test_price, y_test_log, log_draws):
    price_draws = np.exp(log_draws)
    pred_median = np.median(price_draws, axis=0)
    pred_median_log = np.median(log_draws, axis=0)
    lower = np.percentile(price_draws, 2.5, axis=0)
    upper = np.percentile(price_draws, 97.5, axis=0)

    return {
        "MAE": mean_absolute_error(y_test_price, pred_median),
        "RMSE": np.sqrt(mean_squared_error(y_test_price, pred_median)),
        "MAPE_%": np.mean(np.abs((y_test_price - pred_median) / y_test_price)) * 100,
        "R2_log_price": r2_score(y_test_log, pred_median_log),
        "R2_price": r2_score(y_test_price, pred_median),
        "interval_95_coverage_%": np.mean(
            (y_test_price >= lower) & (y_test_price <= upper)
        )
        * 100,
        "median_interval_width": np.median(upper - lower),
    }


def print_metric_table(metrics):
    print("\nBayesian posterior predictive performance:")
    print(f"MAE:                   {metrics['MAE']:,.0f}")
    print(f"RMSE:                  {metrics['RMSE']:,.0f}")
    print(f"MAPE:                  {metrics['MAPE_%']:.2f}%")
    print(f"R2 log price:          {metrics['R2_log_price']:.4f}")
    print(f"R2 price:              {metrics['R2_price']:.4f}")
    print(f"95% interval coverage: {metrics['interval_95_coverage_%']:.2f}%")
    print(f"Median interval width: {metrics['median_interval_width']:,.0f}")


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


def print_prior_specification(model_data, category_data):
    print("\nBayesian prior specification:")
    print("- Target is log(price_sold), standardized using the training set.")
    print("- Numeric features are standardized using the full training set.")
    print("- intercept ~ Normal(0, 1)")
    print("- beta[j] ~ Normal(0, 0.5) for each standardized numeric feature")
    print("- sigma_property ~ HalfNormal(0.5)")
    print("- property_raw[k] ~ Normal(0, 1)")
    print("- property_effect[k] = property_raw[k] * sigma_property")
    print("- sigma ~ HalfNormal(1)")
    print(f"- likelihood: StudentT(nu={STUDENT_T_NU}, mu, sigma)")
    print(
        f"- Numeric beta count: {len(model_data['numeric_features'])}; "
        f"property-type effect count: {len(category_data['property_levels'])}"
    )


def print_standardization_info(model_data):
    print("\nStandardization used for numeric features:")
    print(
        model_data["x_summary"].to_string(
            index=False,
            formatters={
                "mean_used_for_standardization": "{:,.4f}".format,
                "std_used_for_standardization": "{:,.4f}".format,
            },
        )
    )
    print("\nTarget standardization:")
    print(f"- log(price_sold) mean: {model_data['y_mean']:.6f}")
    print(f"- log(price_sold) std:  {model_data['y_std']:.6f}")


def summarize_prior_predictive(prior_idata, model_data):
    prior = prior_idata.prior_predictive["log_price_observed"].values
    prior_std_draws = prior.reshape(-1)
    prior_log_draws = model_data["y_mean"] + model_data["y_std"] * prior_std_draws
    prior_price_draws = np.exp(prior_log_draws)

    return pd.DataFrame(
        [
            {
                "quantity": "prior_predictive_price",
                "mean": np.mean(prior_price_draws),
                "sd": np.std(prior_price_draws),
                "p2_5": np.percentile(prior_price_draws, 2.5),
                "p50": np.percentile(prior_price_draws, 50),
                "p97_5": np.percentile(prior_price_draws, 97.5),
            },
            {
                "quantity": "prior_predictive_log_price",
                "mean": np.mean(prior_log_draws),
                "sd": np.std(prior_log_draws),
                "p2_5": np.percentile(prior_log_draws, 2.5),
                "p50": np.percentile(prior_log_draws, 50),
                "p97_5": np.percentile(prior_log_draws, 97.5),
            },
        ]
    )


def print_prior_summary(prior_summary):
    print("\nPrior predictive summary:")
    print(
        prior_summary.to_string(
            index=False,
            formatters={
                "mean": "{:,.3f}".format,
                "sd": "{:,.3f}".format,
                "p2_5": "{:,.3f}".format,
                "p50": "{:,.3f}".format,
                "p97_5": "{:,.3f}".format,
            },
        )
    )


def posterior_summary_tables(idata, az, model_data, category_data):
    var_names = [
        "intercept",
        "beta",
        "property_effect",
        "sigma",
        "sigma_property",
    ]
    summary = az.summary(idata, var_names=var_names, round_to=4).reset_index()
    summary = summary.rename(columns={"index": "parameter"})

    beta_rows = summary[summary["parameter"].str.startswith("beta[")].copy()
    beta_rows.insert(1, "feature", model_data["numeric_features"])

    property_rows = summary[
        summary["parameter"].str.startswith("property_effect[")
    ].copy()
    property_rows.insert(1, "property_type", category_data["property_levels"])

    return summary, beta_rows, property_rows


def save_model_outputs(
    model_data,
    category_data,
    filter_info,
    distance_info,
    prior_summary,
    posterior_summary,
    beta_summary,
    property_summary,
    metrics,
):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_data["x_summary"].to_csv(
        OUTPUT_DIR / "bayesian_numeric_standardization.csv",
        index=False,
    )
    pd.DataFrame({"property_type": category_data["property_levels"]}).to_csv(
        OUTPUT_DIR / "bayesian_property_type_levels.csv",
        index=False,
    )
    posterior_summary.to_csv(
        OUTPUT_DIR / "bayesian_posterior_summary.csv",
        index=False,
    )
    beta_summary.to_csv(
        OUTPUT_DIR / "bayesian_beta_posterior_summary.csv",
        index=False,
    )
    property_summary.to_csv(
        OUTPUT_DIR / "bayesian_property_type_posterior_summary.csv",
        index=False,
    )
    pd.DataFrame([metrics]).to_csv(
        OUTPUT_DIR / "bayesian_test_metrics.csv",
        index=False,
    )

    if prior_summary is not None:
        prior_summary.to_csv(
            OUTPUT_DIR / "bayesian_prior_predictive_summary.csv",
            index=False,
        )

    preprocessing_rows = [
        {"item": "rows_before_property_type_filtering", "value": filter_info["original_rows"]},
        {"item": "rows_removed_by_property_type_filtering", "value": filter_info["removed_rows"]},
        {"item": "rows_after_property_type_filtering", "value": filter_info["final_rows"]},
        {"item": "msoa_coordinate_rows", "value": distance_info["coordinate_rows"]},
        {
            "item": "train_missing_cambridge_distance_before_fill",
            "value": distance_info["train_missing_cambridge_distance"],
        },
        {
            "item": "test_missing_cambridge_distance_before_fill",
            "value": distance_info["test_missing_cambridge_distance"],
        },
        {
            "item": "train_missing_london_distance_before_fill",
            "value": distance_info["train_missing_london_distance"],
        },
        {
            "item": "test_missing_london_distance_before_fill",
            "value": distance_info["test_missing_london_distance"],
        },
        {
            "item": "median_distance_to_cambridge_km",
            "value": distance_info["median_distance_to_cambridge_km"],
        },
        {
            "item": "median_distance_to_london_km",
            "value": distance_info["median_distance_to_london_km"],
        },
    ]
    pd.DataFrame(preprocessing_rows).to_csv(
        OUTPUT_DIR / "bayesian_preprocessing_summary.csv",
        index=False,
    )

    prior_text = [
        "Bayesian prior specification",
        "Target: standardized log(price_sold)",
        "intercept ~ Normal(0, 1)",
        "beta[j] ~ Normal(0, 0.5)",
        "sigma_property ~ HalfNormal(0.5)",
        "property_raw[k] ~ Normal(0, 1)",
        "property_effect[k] = property_raw[k] * sigma_property",
        "sigma ~ HalfNormal(1)",
        f"likelihood: StudentT(nu={STUDENT_T_NU}, mu, sigma)",
    ]
    (OUTPUT_DIR / "bayesian_prior_specification.txt").write_text(
        "\n".join(prior_text),
        encoding="utf-8",
    )

    print(f"\nSaved Bayesian numeric standardization to {OUTPUT_DIR / 'bayesian_numeric_standardization.csv'}")
    print(f"Saved Bayesian property levels to {OUTPUT_DIR / 'bayesian_property_type_levels.csv'}")
    print(f"Saved Bayesian posterior summary to {OUTPUT_DIR / 'bayesian_posterior_summary.csv'}")
    print(f"Saved Bayesian beta posterior summary to {OUTPUT_DIR / 'bayesian_beta_posterior_summary.csv'}")
    print(
        "Saved Bayesian property-type posterior summary to "
        f"{OUTPUT_DIR / 'bayesian_property_type_posterior_summary.csv'}"
    )
    print(f"Saved Bayesian test metrics to {OUTPUT_DIR / 'bayesian_test_metrics.csv'}")
    print(f"Saved Bayesian prior specification to {OUTPUT_DIR / 'bayesian_prior_specification.txt'}")
    if prior_summary is not None:
        print(
            "Saved Bayesian prior predictive summary to "
            f"{OUTPUT_DIR / 'bayesian_prior_predictive_summary.csv'}"
        )


def save_prediction_intervals(test_df, log_draws):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    price_draws = np.exp(log_draws)
    pred = pd.DataFrame(
        {
            "property_type": test_df["property_type"].astype(str).to_numpy(),
            "property_type_model": test_df["property_type_model"].astype(str).to_numpy(),
            "msoa21": test_df["msoa21"].astype(str).to_numpy(),
            "start_date": test_df["start_date"].to_numpy(),
            "distance_to_cambridge_km": test_df["distance_to_cambridge_km"].to_numpy(),
            "distance_to_london_km": test_df["distance_to_london_km"].to_numpy(),
            "actual_price": test_df[TARGET].to_numpy(dtype=float),
            "pred_median": np.median(price_draws, axis=0),
            "pred_lower_95": np.percentile(price_draws, 2.5, axis=0),
            "pred_upper_95": np.percentile(price_draws, 97.5, axis=0),
            "actual_log_price": np.log(test_df[TARGET].to_numpy(dtype=float)),
            "pred_median_log_price": np.median(log_draws, axis=0),
        }
    )
    pred["residual"] = pred["actual_price"] - pred["pred_median"]
    pred["abs_error"] = np.abs(pred["residual"])
    pred["ape_%"] = pred["abs_error"] / pred["actual_price"] * 100
    output_path = OUTPUT_DIR / "bayesian_prediction_intervals.csv"
    pred.to_csv(output_path, index=False)
    print(f"\nSaved prediction intervals to {output_path}")

    return pred, output_path


def format_price_axis(ax):
    import matplotlib.ticker as mticker

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x / 1000:,.0f}k"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y / 1000:,.0f}k"))


def save_prediction_plots(pred, plot_sample_size):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    pred = pred.copy()
    pred["residual"] = pred["actual_price"] - pred["pred_median"]
    pred["interval_width"] = pred["pred_upper_95"] - pred["pred_lower_95"]
    pred["covered_by_95_interval"] = (
        (pred["actual_price"] >= pred["pred_lower_95"])
        & (pred["actual_price"] <= pred["pred_upper_95"])
    )

    sample_n = min(plot_sample_size, len(pred))
    plot_df = pred.sample(sample_n, random_state=RANDOM_STATE) if sample_n < len(pred) else pred

    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.scatter(
        plot_df["actual_price"],
        plot_df["pred_median"],
        s=14,
        alpha=0.45,
        edgecolors="none",
    )
    lower_limit = min(plot_df["actual_price"].min(), plot_df["pred_median"].min())
    upper_limit = max(plot_df["actual_price"].max(), plot_df["pred_median"].max())
    ax.plot([lower_limit, upper_limit], [lower_limit, upper_limit], color="black", lw=1)
    ax.set_title("Actual Price vs Bayesian Median Prediction")
    ax.set_xlabel("Actual price")
    ax.set_ylabel("Predicted median price")
    format_price_axis(ax)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "actual_vs_predicted.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.hist(pred["residual"], bins=60, color="#4C78A8", alpha=0.82)
    ax.axvline(0, color="black", lw=1)
    ax.set_title("Prediction Residual Distribution")
    ax.set_xlabel("Actual price - predicted median")
    ax.set_ylabel("Count")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x / 1000:,.0f}k"))
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "residual_histogram.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(
        plot_df["pred_median"],
        plot_df["residual"],
        s=14,
        alpha=0.45,
        edgecolors="none",
    )
    ax.axhline(0, color="black", lw=1)
    ax.set_title("Residuals vs Predicted Median Price")
    ax.set_xlabel("Predicted median price")
    ax.set_ylabel("Residual")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x / 1000:,.0f}k"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y / 1000:,.0f}k"))
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "residuals_vs_predicted.png", dpi=180)
    plt.close(fig)

    interval_df = plot_df.sort_values("pred_median").reset_index(drop=True)
    x = np.arange(len(interval_df))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.vlines(
        x,
        interval_df["pred_lower_95"],
        interval_df["pred_upper_95"],
        color="#9ECAE9",
        alpha=0.55,
        lw=1,
    )
    colors = np.where(interval_df["covered_by_95_interval"], "#2CA02C", "#D62728")
    ax.scatter(x, interval_df["actual_price"], s=13, c=colors, alpha=0.75, label="Actual")
    ax.plot(x, interval_df["pred_median"], color="#1F77B4", lw=1.5, label="Median prediction")
    ax.set_title("Bayesian 95% Prediction Intervals")
    ax.set_xlabel("Sampled test observations sorted by predicted median")
    ax.set_ylabel("Price")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y / 1000:,.0f}k"))
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "prediction_intervals_sample.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(
        plot_df["pred_median"],
        plot_df["interval_width"],
        s=14,
        alpha=0.45,
        edgecolors="none",
    )
    ax.set_title("Prediction Uncertainty vs Predicted Price")
    ax.set_xlabel("Predicted median price")
    ax.set_ylabel("95% interval width")
    format_price_axis(ax)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "interval_width_vs_predicted.png", dpi=180)
    plt.close(fig)


def clean_feature_name(name):
    replacements = {
        "num_bed_": "Bedrooms",
        "num_bath": "Bathrooms",
        "num_reception": "Reception rooms",
        "distance_to_cambridge_km": "Distance to Cambridge",
        "distance_to_london_km": "Distance to London",
        "start_year": "Start year",
        "start_month": "Start month",
    }
    return replacements.get(name, name)


def interval_columns(summary):
    known_pairs = [
        ("hdi_3%", "hdi_97%"),
        ("eti89_lb", "eti89_ub"),
        ("hdi_5.5%", "hdi_94.5%"),
    ]
    for lower_col, upper_col in known_pairs:
        if lower_col in summary.columns and upper_col in summary.columns:
            return lower_col, upper_col

    lower_candidates = [col for col in summary.columns if col.endswith("_lb")]
    upper_candidates = [col for col in summary.columns if col.endswith("_ub")]
    if lower_candidates and upper_candidates:
        return lower_candidates[0], upper_candidates[0]

    raise ValueError(
        "Could not find posterior interval columns in ArviZ summary. "
        f"Available columns: {list(summary.columns)}"
    )


def save_coefficient_plot(beta_summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    summary = beta_summary.copy()
    summary["label"] = summary["feature"].map(clean_feature_name)
    summary = summary.sort_values("mean")
    lower_col, upper_col = interval_columns(summary)

    y = np.arange(len(summary))
    fig_height = max(5.5, 0.28 * len(summary))
    fig, ax = plt.subplots(figsize=(8.5, fig_height))
    ax.hlines(y, summary[lower_col], summary[upper_col], color="#4C78A8", lw=3)
    ax.scatter(summary["mean"], y, color="#F58518", zorder=3)
    ax.axvline(0, color="#222222", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(summary["label"])
    ax.set_xlabel("Posterior coefficient on standardized log-price scale")
    ax.set_title("Bayesian Numeric Feature Posterior Intervals")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "coefficient_posterior_intervals.png", dpi=180)
    plt.close(fig)


def save_property_effect_plot(property_summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    summary = property_summary.copy().sort_values("mean")
    lower_col, upper_col = interval_columns(summary)
    y = np.arange(len(summary))
    fig_height = max(5.5, 0.35 * len(summary))
    fig, ax = plt.subplots(figsize=(8.5, fig_height))
    ax.hlines(y, summary[lower_col], summary[upper_col], color="#8AB17D", lw=3)
    ax.scatter(summary["mean"], y, color="#264653", zorder=3)
    ax.axvline(0, color="#222222", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(summary["property_type"])
    ax.set_xlabel("Posterior property-type effect on standardized log-price scale")
    ax.set_title("Bayesian Property-Type Posterior Intervals")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "property_type_posterior_intervals.png", dpi=180)
    plt.close(fig)


def save_trace_plot(idata, az):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        az.plot_trace(
            idata,
            var_names=["intercept", "sigma", "sigma_property"],
        )
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "mcmc_trace_core_parameters.png", dpi=180)
        plt.close("all")
    except Exception as exc:
        print(f"Could not save trace plot: {exc}")


def save_all_plots(idata, az, pred, beta_summary, property_summary, plot_sample_size):
    save_prediction_plots(pred, plot_sample_size)
    save_coefficient_plot(beta_summary)
    save_property_effect_plot(property_summary)
    save_trace_plot(idata, az)
    print(f"Saved Bayesian plots to {PLOT_DIR}")


def prepare_data(feature_set):
    data_path = choose_data_path()
    df = pd.read_csv(data_path)
    df = df.rename(columns=UPDATED_DATA_COLUMN_RENAMES)
    df = clean_input_data(df)
    df, filter_info = filter_rare_property_type_rows(df)

    full_train_df, test_df = random_train_test_split(df)
    full_train_df, test_df, distance_info = add_distance_features(full_train_df, test_df)
    full_train_df, test_df = add_property_type_model(full_train_df, test_df)

    if feature_set == "core":
        numeric_features = available_columns(full_train_df, CORE_NUMERIC_FEATURES)
    else:
        numeric_features = available_columns(full_train_df, FULL_NUMERIC_FEATURES)

    return data_path, df, full_train_df, test_df, filter_info, distance_info, numeric_features


def main():
    args = parse_args()
    pm, az = require_bayesian_packages()

    rng = np.random.default_rng(RANDOM_STATE)
    (
        data_path,
        df,
        full_train_df,
        test_df,
        filter_info,
        distance_info,
        numeric_features,
    ) = prepare_data(args.feature_set)
    train_df = sample_training_rows(full_train_df, args.max_train_rows)

    model_data = standardize_train_test(
        train_df=train_df,
        test_df=test_df,
        numeric_features=numeric_features,
        scale_source_df=full_train_df,
    )
    category_data = encode_property_types(
        train_df=train_df,
        test_df=test_df,
        level_source_df=full_train_df,
    )

    print(f"Data file:                       {data_path}")
    print(f"Rows after cleaning/filtering:   {len(df):,}")
    print(f"Split method:                    random")
    print(f"Random state:                    {RANDOM_STATE}")
    print(f"Full training rows:              {len(full_train_df):,}")
    print(f"MCMC training rows used:         {len(train_df):,}")
    print(f"Test rows:                       {len(test_df):,}")
    print(f"Feature set:                     {args.feature_set}")
    print(f"Numeric feature count:           {len(model_data['numeric_features'])}")
    print(f"Property type levels:            {len(category_data['property_levels'])}")
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
    print(f"MCMC draws/chains:               {args.draws} draws x {args.chains} chains")
    print(f"MCMC tuning draws:               {args.tune}")
    print(f"MCMC cores:                      {args.cores}")
    print(f"Prior predictive draws:          {args.prior_draws}")
    print(
        "Posterior predictive draws:      "
        f"{args.posterior_predictive_draws}"
    )
    print(f"Target:                          standardized log(price_sold)")

    print_property_type_filtering(filter_info)
    print_prior_specification(model_data, category_data)
    print_standardization_info(model_data)

    model = build_model(pm, model_data, category_data)
    prior_summary = None

    if not args.skip_prior_predictive:
        with model:
            prior_idata = pm.sample_prior_predictive(
                draws=args.prior_draws,
                random_seed=RANDOM_STATE,
                return_inferencedata=True,
            )
        prior_summary = summarize_prior_predictive(prior_idata, model_data)
        print_prior_summary(prior_summary)

    with model:
        idata = pm.sample(
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            cores=args.cores,
            target_accept=args.target_accept,
            random_seed=RANDOM_STATE,
            return_inferencedata=True,
            progressbar=args.progressbar,
        )

    posterior_summary, beta_summary, property_summary = posterior_summary_tables(
        idata=idata,
        az=az,
        model_data=model_data,
        category_data=category_data,
    )

    print("\nPosterior summary:")
    print(posterior_summary.to_string(index=False))

    print("\nNumeric beta posterior summary:")
    print(beta_summary.to_string(index=False))

    print("\nProperty-type posterior summary:")
    print(property_summary.to_string(index=False))

    posterior = flatten_posterior(idata, rng, args.posterior_predictive_draws)
    log_draws = posterior_predictive_log_prices(posterior, model_data, category_data, rng)
    metrics = evaluate_predictions(
        model_data["y_test_price"],
        model_data["y_test_log"],
        log_draws,
    )
    print_metric_table(metrics)
    pred, _ = save_prediction_intervals(test_df, log_draws)

    save_model_outputs(
        model_data=model_data,
        category_data=category_data,
        filter_info=filter_info,
        distance_info=distance_info,
        prior_summary=prior_summary,
        posterior_summary=posterior_summary,
        beta_summary=beta_summary,
        property_summary=property_summary,
        metrics=metrics,
    )

    if not args.no_plots:
        save_all_plots(
            idata=idata,
            az=az,
            pred=pred,
            beta_summary=beta_summary,
            property_summary=property_summary,
            plot_sample_size=args.plot_sample_size,
        )


if __name__ == "__main__":
    main()
