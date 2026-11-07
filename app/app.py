from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------- Paths and settings ----------
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "price_model.joblib"
LOCS_PATH = ROOT / "models" / "locations.csv"
FEATS = ["city", "loc_key", "log_area", "bedrooms", "baths", "latitude", "longitude"]
TYPICAL_ERROR = 0.103  # median error on the test set

st.set_page_config(page_title="Pakistan House Price Predictor", page_icon="🏠")


# ---------- Load model and locations (once) ----------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_locations():
    return pd.read_csv(LOCS_PATH)


model = load_model()
locs = load_locations()


def fmt_pkr(p):
    return f"PKR {p/1e7:.2f} crore" if p >= 1e7 else f"PKR {p/1e5:.1f} lakh"


# ---------- Page ----------
st.title("🏠 Pakistan House Price Predictor")
st.caption(
    "Estimates the asking price of a house for sale in 5 Pakistani cities. "
    "Trained on 66,000 Zameen.com listings (2018–2019 data)."
)

city = st.selectbox("City", sorted(locs["city"].unique()))

city_locs = locs[(locs["city"] == city) & (locs["listings"] >= 10)].sort_values(
    "location"
)
location = st.selectbox("Location", city_locs["location"])

col1, col2 = st.columns(2)
unit = col1.radio("Size unit", ["Marla", "Kanal"], horizontal=True)
if unit == "Marla":
    size = col2.number_input(
        "Size (Marla)", min_value=1.0, max_value=100.0, value=10.0, step=0.5
    )
    area_marla = size
else:
    size = col2.number_input(
        "Size (Kanal)", min_value=0.5, max_value=5.0, value=1.0, step=0.5
    )
    area_marla = size * 20

col3, col4 = st.columns(2)
bedrooms = col3.slider("Bedrooms", 1, 10, 4)
baths = col4.slider("Bathrooms", 1, 10, 4)

# ---------- Predict ----------
if st.button("Estimate price", type="primary"):
    row = city_locs[city_locs["location"] == location].iloc[0]
    X = pd.DataFrame(
        [
            {
                "city": city,
                "loc_key": f"{city} | {location}",
                "log_area": np.log(area_marla),
                "bedrooms": bedrooms,
                "baths": baths,
                "latitude": row["latitude"],
                "longitude": row["longitude"],
            }
        ]
    )[FEATS]

    price = float(np.exp(model.predict(X)[0]))
    low, high = price * (1 - TYPICAL_ERROR), price * (1 + TYPICAL_ERROR)

    st.subheader(f"Estimated price: {fmt_pkr(price)}")
    st.write(f"Likely range: **{fmt_pkr(low)} – {fmt_pkr(high)}**")
    st.write(f"That's about **PKR {price/area_marla/1e5:.1f} lakh per Marla**.")
    st.caption(
        "Based on 2018–2019 asking prices. Current market prices are likely higher."
    )
