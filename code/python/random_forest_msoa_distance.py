from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "cleaned" / "Cambridge data_cleaned.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
PLOT_DIR = OUTPUT_DIR / "rf_distance_without_msoa_plots"

TARGET = "price_sold"
TEST_SIZE = 0.20
RANDOM_STATE = 42

USE_LOG_TARGET = False
MIN_PROPERTY_TYPE_COUNT = 500
MIN_MSOA_COUNT = 100
TOP_IMPORTANCE_ROWS = 25
PLOT_SAMPLE_SIZE = 1000

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


# The cleaned data stores the social columns with R-style dotted names; map them
# back to the readable names used in NUMERIC_FEATURES below.
COLUMN_RENAME = {
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


NUMERIC_FEATURES = [
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

BASE_CATEGORICAL_FEATURES = ["property_type_model"]
MSOA_CATEGORICAL_FEATURE = "msoa21_model"


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

    return df


def random_train_test_split(df):
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def common_levels(series, min_count):
    counts = series.fillna("Missing").astype(str).value_counts()
    return set(counts[counts >= min_count].index)


def collapse_rare_levels(series, levels, other_label):
    values = series.fillna(other_label).astype(str)
    return values.where(values.isin(levels), other_label)


def add_msoa_distance_feature(train_df, test_df):
    # Distances to Cambridge (market square) and London (Trafalgar Square) are
    # already provided in the cleaned data as dist_cambridge / dist_london,
    # both in km. Use them directly rather than recomputing from coordinates.
    train_df = train_df.copy()
    test_df = test_df.copy()

    distance_sources = {
        "distance_to_cambridge_km": "dist_cambridge",
        "distance_to_london_km": "dist_london",
    }

    missing_columns = [
        source_col
        for source_col in distance_sources.values()
        if source_col not in train_df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Missing distance columns in cleaned data: {missing_columns}"
        )

    distance_info = {}
    for model_col, source_col in distance_sources.items():
        train_df[model_col] = pd.to_numeric(train_df[source_col], errors="coerce")
        test_df[model_col] = pd.to_numeric(test_df[source_col], errors="coerce")

        fill_value = train_df[model_col].median()
        distance_info[model_col] = {
            "train_missing": int(train_df[model_col].isna().sum()),
            "test_missing": int(test_df[model_col].isna().sum()),
            "median_km": fill_value,
        }

        train_df[model_col] = train_df[model_col].fillna(fill_value)
        test_df[model_col] = test_df[model_col].fillna(fill_value)

    return train_df, test_df, distance_info


def add_categorical_model_features(train_df, test_df):
    property_counts = train_df["property_type"].fillna("Missing").astype(str).value_counts()
    property_levels = set(property_counts[property_counts >= MIN_PROPERTY_TYPE_COUNT].index)
    collapsed_property_levels = property_counts[
        property_counts < MIN_PROPERTY_TYPE_COUNT
    ]

    msoa_levels = common_levels(train_df["msoa21"], MIN_MSOA_COUNT)

    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df["property_type_model"] = collapse_rare_levels(
        train_df["property_type"],
        property_levels,
        "Other property type",
    )
    test_df["property_type_model"] = collapse_rare_levels(
        test_df["property_type"],
        property_levels,
        "Other property type",
    )

    train_df["msoa21_model"] = collapse_rare_levels(
        train_df["msoa21"],
        msoa_levels,
        "Other MSOA",
    )
    test_df["msoa21_model"] = collapse_rare_levels(
        test_df["msoa21"],
        msoa_levels,
        "Other MSOA",
    )

    feature_info = {
        "n_property_levels": len(property_levels),
        "n_msoa_levels": len(msoa_levels),
        "property_levels": sorted(property_levels),
        "collapsed_property_levels": collapsed_property_levels,
    }

    return train_df, test_df, feature_info


def available_columns(df, columns):
    return [col for col in columns if col in df.columns]


def build_feature_lists(df, include_msoa_dummy):
    numeric_features = available_columns(df, NUMERIC_FEATURES)
    categorical_features = list(BASE_CATEGORICAL_FEATURES)

    if include_msoa_dummy:
        categorical_features.append(MSOA_CATEGORICAL_FEATURE)

    return numeric_features, categorical_features


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

    if USE_LOG_TARGET:
        y_train_model = np.log(y_train)
        y_test_model = np.log(y_test)
    else:
        y_train_model = y_train
        y_test_model = y_test

    return X_train, X_test, y_train, y_test, y_train_model, y_test_model


def build_random_forest_pipeline(numeric_features, categorical_features):
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    model = RandomForestRegressor(**RF_PARAMS)

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


def predict_price(model, X_test):
    predictions = model.predict(X_test)
    if USE_LOG_TARGET:
        return np.exp(predictions), predictions

    return predictions, np.log(np.maximum(predictions, 1))


def evaluate_model(name, model, X_test, y_test):
    predicted_price, predicted_log_price = predict_price(model, X_test)

    # Compare like with like: R2_price in price space, R2_log_price in log
    # space. predicted_log_price is log of the predicted price in both target
    # modes, so pair it with log of the actual price (never the raw price).
    log_actual = np.log(np.maximum(np.asarray(y_test, dtype=float), 1.0))

    return {
        "model": name,
        "n_features_before_encoding": X_test.shape[1],
        "MAE": mean_absolute_error(y_test, predicted_price),
        "RMSE": np.sqrt(mean_squared_error(y_test, predicted_price)),
        "MAPE_%": np.mean(np.abs((y_test - predicted_price) / y_test)) * 100,
        "R2_log_price": r2_score(log_actual, predicted_log_price),
        "R2_price": r2_score(y_test, predicted_price),
    }


def feature_importance_table(model):
    preprocessor = model.named_steps["preprocess"]
    feature_names = preprocessor.get_feature_names_out()
    importances = model.named_steps["model"].feature_importances_

    return (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importances,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def format_price_axis(ax):
    import matplotlib.ticker as mticker

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x / 1000:,.0f}k"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y / 1000:,.0f}k"))


def safe_r2_score(y_true, y_pred):
    if len(y_true) < 2 or np.isclose(np.var(y_true), 0):
        return np.nan

    return r2_score(y_true, y_pred)


def property_type_metric_table(plot_df):
    rows = []

    for property_type, group in plot_df.groupby("property_type", dropna=False):
        actual = group["actual_price"].to_numpy(dtype=float)
        predicted = group["predicted_price"].to_numpy(dtype=float)
        residual = actual - predicted
        absolute_error = np.abs(residual)

        rows.append(
            {
                "property_type": property_type,
                "n_test": len(group),
                "actual_mean": actual.mean(),
                "predicted_mean": predicted.mean(),
                "MAE": absolute_error.mean(),
                "RMSE": np.sqrt(np.mean(residual**2)),
                "MAPE_%": np.mean(absolute_error / actual) * 100,
                "median_abs_error": np.median(absolute_error),
                "mean_residual": residual.mean(),
                "R2_price": safe_r2_score(actual, predicted),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["MAE", "n_test"], ascending=[False, False])
        .reset_index(drop=True)
    )


def print_property_type_metrics(metrics):
    print("\nRandom Forest distance model: performance by property_type")
    print(
        metrics.to_string(
            index=False,
            formatters={
                "actual_mean": "{:,.0f}".format,
                "predicted_mean": "{:,.0f}".format,
                "MAE": "{:,.0f}".format,
                "RMSE": "{:,.0f}".format,
                "MAPE_%": "{:.2f}".format,
                "median_abs_error": "{:,.0f}".format,
                "mean_residual": "{:,.0f}".format,
                "R2_price": lambda x: "" if pd.isna(x) else f"{x:.4f}",
            },
        )
    )


def print_r2_plot_exclusions(metrics):
    excluded = metrics.loc[
        metrics["R2_price"].isna() | (metrics["R2_price"] < 0),
        ["property_type", "n_test", "R2_price"],
    ].copy()

    if excluded.empty:
        print("\nAll property types with test observations are shown in the R2 plot.")
        return

    excluded["reason"] = np.where(
        excluded["R2_price"].isna(),
        "R2 cannot be calculated",
        "R2 is negative",
    )
    excluded.to_csv(
        OUTPUT_DIR / "rf_distance_without_msoa_property_type_r2_plot_excluded.csv",
        index=False,
    )

    print("\nProperty types not shown in the R2 plot:")
    print(
        excluded.to_string(
            index=False,
            formatters={
                "R2_price": lambda x: "" if pd.isna(x) else f"{x:.4f}",
            },
        )
    )


def save_property_type_plots(plot_df, metrics):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    metrics_for_plot = metrics.sort_values("MAE", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(metrics_for_plot["property_type"], metrics_for_plot["MAE"], color="#4C78A8")
    ax.set_title("Random Forest Distance Model: MAE by Property Type")
    ax.set_xlabel("MAE")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x / 1000:,.0f}k"))
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "property_type_mae.png", dpi=180)
    plt.close(fig)

    metrics_for_plot = metrics.sort_values("MAPE_%", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(metrics_for_plot["property_type"], metrics_for_plot["MAPE_%"], color="#F58518")
    ax.set_title("Random Forest Distance Model: MAPE by Property Type")
    ax.set_xlabel("MAPE (%)")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "property_type_mape.png", dpi=180)
    plt.close(fig)

    print_r2_plot_exclusions(metrics)
    r2_metrics = (
        metrics.dropna(subset=["R2_price"])
        .loc[lambda df: df["R2_price"] >= 0]
        .sort_values("R2_price", ascending=True)
    )
    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.barh(r2_metrics["property_type"], r2_metrics["R2_price"], color="#4C78A8")
    ax.axvline(0, color="black", lw=1)
    ax.set_title("Random Forest Distance Model: Test R2 by Property Type")
    ax.set_xlabel("Test R2")
    ax.set_ylabel("")
    ax.set_xlim(0, max(1.0, r2_metrics["R2_price"].max() * 1.12))
    for bar in bars:
        value = bar.get_width()
        ax.text(
            value + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            ha="left",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "property_type_r2.png", dpi=180)
    plt.close(fig)

    means = metrics.sort_values("actual_mean", ascending=True)
    y = np.arange(len(means))
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(means["actual_mean"], y, label="Mean actual price", color="#2CA02C")
    ax.scatter(means["predicted_mean"], y, label="Mean predicted price", color="#D62728")
    for row_idx, row in enumerate(means.itertuples(index=False)):
        ax.plot(
            [row.actual_mean, row.predicted_mean],
            [row_idx, row_idx],
            color="#BDBDBD",
            lw=1,
            zorder=0,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(means["property_type"])
    ax.set_title("Random Forest Distance Model: Mean Actual vs Predicted by Property Type")
    ax.set_xlabel("Price")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x / 1000:,.0f}k"))
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "property_type_mean_actual_vs_predicted.png", dpi=180)
    plt.close(fig)

    box_df = plot_df.copy()
    box_df["absolute_error"] = box_df["residual"].abs()
    ordered_types = (
        box_df.groupby("property_type")["absolute_error"]
        .median()
        .sort_values(ascending=True)
        .index
        .tolist()
    )
    data = [
        box_df.loc[box_df["property_type"] == property_type, "absolute_error"]
        for property_type in ordered_types
    ]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.boxplot(data, orientation="horizontal", tick_labels=ordered_types, showfliers=False)
    ax.set_title("Random Forest Distance Model: Absolute Error by Property Type")
    ax.set_xlabel("Absolute error")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x / 1000:,.0f}k"))
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "property_type_absolute_error_boxplot.png", dpi=180)
    plt.close(fig)


def save_without_msoa_plots(X_test, y_test, predicted_price, importances, property_types):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    plot_df = pd.DataFrame(
        {
            "actual_price": y_test.to_numpy(dtype=float),
            "predicted_price": predicted_price,
            "residual": y_test.to_numpy(dtype=float) - predicted_price,
            "property_type": property_types.astype(str).to_numpy(),
            "distance_to_cambridge_km": X_test["distance_to_cambridge_km"].to_numpy(
                dtype=float
            ),
        }
    )
    plot_df.to_csv(OUTPUT_DIR / "rf_distance_without_msoa_predictions.csv", index=False)
    property_metrics = property_type_metric_table(plot_df)
    property_metrics.to_csv(
        OUTPUT_DIR / "rf_distance_without_msoa_property_type_metrics.csv",
        index=False,
    )
    print_property_type_metrics(property_metrics)

    sample_n = min(PLOT_SAMPLE_SIZE, len(plot_df))
    sample_df = (
        plot_df.sample(sample_n, random_state=RANDOM_STATE)
        if sample_n < len(plot_df)
        else plot_df
    )

    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.scatter(
        sample_df["actual_price"],
        sample_df["predicted_price"],
        s=14,
        alpha=0.45,
        edgecolors="none",
    )
    lower_limit = min(sample_df["actual_price"].min(), sample_df["predicted_price"].min())
    upper_limit = max(sample_df["actual_price"].max(), sample_df["predicted_price"].max())
    ax.plot([lower_limit, upper_limit], [lower_limit, upper_limit], color="black", lw=1)
    ax.set_title("Random Forest Distance Model: Actual vs Predicted")
    ax.set_xlabel("Actual price")
    ax.set_ylabel("Predicted price")
    format_price_axis(ax)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "actual_vs_predicted.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.hist(plot_df["residual"], bins=60, color="#4C78A8", alpha=0.82)
    ax.axvline(0, color="black", lw=1)
    ax.set_title("Random Forest Distance Model: Residual Distribution")
    ax.set_xlabel("Actual price - predicted price")
    ax.set_ylabel("Count")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x / 1000:,.0f}k"))
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "residual_histogram.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(
        sample_df["predicted_price"],
        sample_df["residual"],
        s=14,
        alpha=0.45,
        edgecolors="none",
    )
    ax.axhline(0, color="black", lw=1)
    ax.set_title("Random Forest Distance Model: Residuals vs Predicted")
    ax.set_xlabel("Predicted price")
    ax.set_ylabel("Residual")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x / 1000:,.0f}k"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y / 1000:,.0f}k"))
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "residuals_vs_predicted.png", dpi=180)
    plt.close(fig)

    top_importances = importances.head(20).iloc[::-1].copy()
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top_importances["feature"], top_importances["importance"], color="#4C78A8")
    ax.set_title("Random Forest Distance Model: Top Feature Importances")
    ax.set_xlabel("Random forest importance")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "top_feature_importances.png", dpi=180)
    plt.close(fig)

    distance_df = plot_df.copy()
    distance_df["distance_bin"] = pd.qcut(
        distance_df["distance_to_cambridge_km"],
        q=min(10, distance_df["distance_to_cambridge_km"].nunique()),
        duplicates="drop",
    )
    distance_summary = (
        distance_df.groupby("distance_bin", observed=True)
        .agg(
            distance_mid=("distance_to_cambridge_km", "mean"),
            actual_price=("actual_price", "mean"),
            predicted_price=("predicted_price", "mean"),
        )
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(
        sample_df["distance_to_cambridge_km"],
        sample_df["actual_price"],
        s=12,
        alpha=0.25,
        edgecolors="none",
        label="Actual test prices",
    )
    ax.plot(
        distance_summary["distance_mid"],
        distance_summary["actual_price"],
        color="#2CA02C",
        lw=2,
        marker="o",
        label="Mean actual price by distance bin",
    )
    ax.plot(
        distance_summary["distance_mid"],
        distance_summary["predicted_price"],
        color="#D62728",
        lw=2,
        marker="o",
        label="Mean predicted price by distance bin",
    )
    ax.set_title("Random Forest Distance Model: Price vs Distance")
    ax.set_xlabel("Distance to Cambridge centre (km)")
    ax.set_ylabel("Price")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y / 1000:,.0f}k"))
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "price_vs_distance.png", dpi=180)
    plt.close(fig)

    save_property_type_plots(plot_df, property_metrics)

    print(f"\nSaved RF without-MSOA-dummy plots to {PLOT_DIR}")
    print(
        "Saved RF without-MSOA-dummy predictions to "
        f"{OUTPUT_DIR / 'rf_distance_without_msoa_predictions.csv'}"
    )
    print(
        "Saved RF without-MSOA-dummy property-type metrics to "
        f"{OUTPUT_DIR / 'rf_distance_without_msoa_property_type_metrics.csv'}"
    )


def run_experiment(experiment_name, include_msoa_dummy, train_df, test_df):
    numeric_features, categorical_features = build_feature_lists(
        train_df,
        include_msoa_dummy,
    )
    X_train, X_test, y_train, y_test, y_train_model, _ = build_xy(
        train_df,
        test_df,
        numeric_features,
        categorical_features,
    )

    print(f"\n=== {experiment_name} ===")
    print(f"MSOA dummy included:         {include_msoa_dummy}")
    print(f"Distance feature included:   {'distance_to_cambridge_km' in numeric_features}")
    print(f"Training rows used:          {len(X_train):,}")
    print(f"Test rows used:              {len(X_test):,}")
    print(f"Numeric features:            {len(numeric_features)}")
    print(f"Categorical features:        {len(categorical_features)}")
    print(f"Features before one-hot:     {X_train.shape[1]}")

    model = build_random_forest_pipeline(numeric_features, categorical_features)
    model.fit(X_train, y_train_model)

    result = evaluate_model(
        experiment_name,
        model,
        X_test,
        y_test,
    )

    importances = feature_importance_table(model)

    print("\nTop feature importances:")
    print(importances.head(TOP_IMPORTANCE_ROWS).to_string(index=False))

    if not include_msoa_dummy:
        predicted_price, _ = predict_price(model, X_test)
        property_types = test_df.loc[X_test.index, "property_type"]
        save_without_msoa_plots(
            X_test,
            y_test,
            predicted_price,
            importances,
            property_types,
        )

    return result


def print_results(results):
    results = pd.DataFrame(results).sort_values("RMSE")
    print("\nRandom Forest with MSOA-distance test-set performance:")
    print(
        results.to_string(
            index=False,
            formatters={
                "MAE": "{:,.0f}".format,
                "RMSE": "{:,.0f}".format,
                "MAPE_%": "{:,.2f}".format,
                "R2_log_price": "{:.4f}".format,
                "R2_price": "{:.4f}".format,
            },
        )
    )


def print_property_type_preprocessing(feature_info):
    print("\nProperty type preprocessing for model input:")
    print(
        "Property types kept as separate model categories "
        f"(training count >= {MIN_PROPERTY_TYPE_COUNT}):"
    )
    for property_type in feature_info["property_levels"]:
        print(f"- {property_type}")

    collapsed = feature_info["collapsed_property_levels"]
    print(
        "\nProperty types collapsed into 'Other property type' "
        f"(training count < {MIN_PROPERTY_TYPE_COUNT}):"
    )
    if collapsed.empty:
        print("- None")
    else:
        for property_type, count in collapsed.items():
            print(f"- {property_type}: {count:,} training rows")


def main():
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns=COLUMN_RENAME)
    df = clean_input_data(df)
    train_df, test_df = random_train_test_split(df)
    train_df, test_df, distance_info = add_msoa_distance_feature(train_df, test_df)
    train_df, test_df, feature_info = add_categorical_model_features(
        train_df,
        test_df,
    )

    print(f"Total rows after cleaning:       {len(df):,}")
    print(f"Split method:                    random")
    print(f"Random state:                    {RANDOM_STATE}")
    print(f"Training rows:                   {len(train_df):,}")
    print(f"Test rows:                       {len(test_df):,}")
    print(f"Property type levels:            {feature_info['n_property_levels']}")
    print(f"MSOA levels:                     {feature_info['n_msoa_levels']}")
    for model_col, info in distance_info.items():
        print(f"{model_col}:")
        print(f"    Train rows missing:          {info['train_missing']:,}")
        print(f"    Test rows missing:           {info['test_missing']:,}")
        print(f"    Median used for fill:        {info['median_km']:.3f} km")
    print(f"Target used for fitting:         {'log(price_sold)' if USE_LOG_TARGET else TARGET}")
    print_property_type_preprocessing(feature_info)

    results = [
        run_experiment(
            "RF Distance Without MSOA Dummy",
            False,
            train_df,
            test_df,
        ),
        run_experiment(
            "RF Distance With MSOA Dummy",
            True,
            train_df,
            test_df,
        ),
    ]
    print_results(results)


if __name__ == "__main__":
    main()
