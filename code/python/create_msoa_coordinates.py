from pathlib import Path
import csv
import json
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "cleaned" / "Cambridge data_cleaned.csv"
OUTPUT_PATH = BASE_DIR / "data" / "spatial" / "msoa21_coordinates.csv"
ONS_FEATURE_URL = (
    "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/"
    "MSOA_December_2021_EW_PWC_V2/FeatureServer/0/query"
)


def fetch_json(url):
    with urlopen(url, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def get_msoa_codes():
    df = pd.read_csv(DATA_PATH, usecols=["msoa21"])
    return sorted(df["msoa21"].dropna().astype(str).unique())


def query_ons_population_weighted_centroids(msoa_codes):
    quoted_codes = ", ".join(f"'{code}'" for code in msoa_codes)
    params = {
        "where": f"MSOA21CD IN ({quoted_codes})",
        "outFields": "MSOA21CD",
        "returnGeometry": "true",
        "outSR": "4326",
        "f": "json",
    }
    data = fetch_json(f"{ONS_FEATURE_URL}?{urlencode(params)}")

    if "error" in data:
        raise RuntimeError(f"ONS query failed: {data['error']}")

    rows = []
    for feature in data.get("features", []):
        msoa_code = feature["attributes"]["MSOA21CD"]
        geometry = feature["geometry"]
        rows.append(
            {
                "msoa21": msoa_code,
                "latitude": geometry["y"],
                "longitude": geometry["x"],
                "source": (
                    "ONS MSOA December 2021 EW Population Weighted Centroids V2"
                ),
            }
        )

    return rows


def main():
    msoa_codes = get_msoa_codes()
    rows = query_ons_population_weighted_centroids(msoa_codes)

    found_codes = {row["msoa21"] for row in rows}
    missing_codes = sorted(set(msoa_codes) - found_codes)
    if missing_codes:
        raise ValueError(f"Missing coordinates for MSOA codes: {missing_codes}")

    rows = sorted(rows, key=lambda row: row["msoa21"])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "msoa21",
                "latitude",
                "longitude",
                "source",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
