# HDB Resale Price Prediction Streamlit App
# Takes in flat details and predicts resale price using the trained model

import joblib
import streamlit as st
import pandas as pd

## page setup, no icon since i didn't want an emoji showing in the tab
st.set_page_config(
    page_title="HDB Resale Price Estimator",
    layout="wide"
)

## custom css for the dark theme, just hardcoding the hex codes instead of using css variables since it's easier to follow
## bg #14171C, panels #1C2029, accent orange #F17B33, text #F4F1E9
st.markdown("""
<style>

.stApp {
    background-color: #14171C;
}

html, body, [class*="css"] {
    font-family: 'Segoe UI', Arial, sans-serif;
    color: #F4F1E9;
}

h1, h2, h3 {
    font-family: 'Trebuchet MS', 'Segoe UI', sans-serif !important;
    color: #F4F1E9 !important;
}

p, span, label, .stMarkdown {
    color: #F4F1E9 !important;
}

section[data-testid="stSidebar"] {
    background-color: #1C2029;
    border-right: 2px solid #F17B33;
}

section[data-testid="stSidebar"] * {
    color: #F4F1E9 !important;
}

## title box at the top
.hero-panel {
    background-color: #1C2029;
    border-radius: 6px;
    padding: 2.5rem 2.5rem;
    margin-bottom: 1.5rem;
}
.hero-title {
    font-family: 'Trebuchet MS', 'Segoe UI', sans-serif;
    font-size: 2.8rem;
    font-weight: 700;
    color: #F4F1E9;
    margin-bottom: 0.6rem;
}
.hero-subtitle {
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 1.05rem;
    color: #C7CBD4;
}

## box showing the predicted price, same bg color as the title box above
.block-plate {
    background-color: #1C2029;
    border-radius: 4px;
    padding: 1.5rem 2rem;
    text-align: center;
    margin-bottom: 1rem;
}
.block-plate .plate-label {
    font-family: 'Trebuchet MS', 'Segoe UI', sans-serif;
    color: #F17B33;
    letter-spacing: 0.3em;
    font-size: 0.8rem;
    text-transform: uppercase;
    opacity: 0.9;
}
.block-plate .plate-value {
    font-family: 'Courier New', monospace;
    color: #FFFFFF;
    font-size: 2.6rem;
    font-weight: 700;
    line-height: 1.3;
}

.summary-card {
    background-color: #1C2029;
    border: 1px solid #3A4250;
    border-left: 3px solid #F17B33;
    border-radius: 4px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.6rem;
}
.summary-card .label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #9AA3B4;
}
.summary-card .value {
    font-family: 'Trebuchet MS', 'Segoe UI', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #F4F1E9;
}

div.stButton > button {
    background-color: #F17B33;
    color: #14171C;
    font-family: 'Trebuchet MS', 'Segoe UI', sans-serif;
    font-weight: 700;
    border: none;
    border-radius: 4px;
    padding: 0.6rem 1.2rem;
    width: 100%;
}
div.stButton > button:hover {
    background-color: #C85F22;
    color: #FFFFFF;
}

[data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
    color: #F4F1E9 !important;
}

div[data-testid="stAlert"] {
    background-color: #1C2029 !important;
    border: none !important;
    border-left: 3px solid #F17B33 !important;
    border-radius: 0 !important;
}
div[data-testid="stAlert"] * {
    background-color: transparent !important;
    color: #F4F1E9 !important;
}
div[data-testid="stAlert"] svg {
    fill: #F17B33 !important;
    color: #F17B33 !important;
}

## hiding streamlit's default menu/footer/header links
#MainMenu {
    visibility: hidden;
}
footer {
    visibility: hidden;
}
header[data-testid="stHeader"] {
    visibility: hidden;
}
[data-testid="stHeaderActionElements"] {
    display: none;
}
a {
    pointer-events: none;
    text-decoration: none;
    color: inherit !important;
}
</style>
""", unsafe_allow_html=True)
## unsafe_allow_html allows raw HTML/CSS injection so Streamlit can apply the custom styling defined above (hiding the default menu/footer/header links and disabling link interaction).

## loading the model, cached so it doesn't reload on every rerun
@st.cache_resource
def load_model():
    return joblib.load("hdb_final_model.pkl")

try:
    model = load_model()
except FileNotFoundError:
    st.error("The trained prediction model could not be found. Please ensure 'hdb_final_model.pkl' is in the same folder as this application.")
    st.stop()
except Exception as e:
    st.error(f"The prediction model could not be loaded due to an unexpected error: {e}")
    st.stop()

## dropdown/slider options
regions = ["Central", "East", "North", "North-East", "West"]

flat_types = ["1 ROOM", "2 ROOM", "3 ROOM", "4 ROOM", "5 ROOM", "EXECUTIVE", "MULTI-GENERATION"]

storey_ranges = [
    "01 TO 03", "04 TO 06", "07 TO 09", "10 TO 12", "13 TO 15", "16 TO 18",
    "19 TO 21", "22 TO 24", "25 TO 27", "28 TO 30", "31 TO 33", "34 TO 36",
    "37 TO 39", "40 TO 42", "43 TO 45", "46 TO 48", "49 TO 51"
]

flat_models = [
    "2-room", "3Gen", "Adjoined flat", "Apartment", "DBSS", "Improved",
    "Improved-Maisonette", "Maisonette", "Model A", "Model A-Maisonette",
    "Model A2", "Multi Generation", "New Generation", "Premium Apartment",
    "Premium Apartment Loft", "Premium Maisonette", "Simplified",
    "Standard", "Terrace", "Type S1", "Type S2"
]

## town to region mapping, needed since the model was trained on region not town
region_dict = {
    "ANG MO KIO": "North-East", "BEDOK": "East", "BISHAN": "Central",
    "BUKIT BATOK": "West", "BUKIT MERAH": "Central", "BUKIT PANJANG": "West",
    "BUKIT TIMAH": "Central", "CENTRAL AREA": "Central", "CHOA CHU KANG": "West",
    "CLEMENTI": "West", "GEYLANG": "Central", "HOUGANG": "North-East",
    "JURONG EAST": "West", "JURONG WEST": "West", "KALLANG/WHAMPOA": "Central",
    "MARINE PARADE": "Central", "PASIR RIS": "East", "PUNGGOL": "North-East",
    "QUEENSTOWN": "Central", "SEMBAWANG": "North", "SENGKANG": "North-East",
    "SERANGOON": "North-East", "TAMPINES": "East", "TOA PAYOH": "Central",
    "WOODLANDS": "North", "YISHUN": "North"
}

towns = sorted(region_dict.keys())

## rough floor area ranges per flat type, used to flag weird combos these are approximate, not hard limits from the model
typical_floor_area = {
    "1 ROOM": (28, 45),
    "2 ROOM": (35, 50),
    "3 ROOM": (50, 75),
    "4 ROOM": (75, 105),
    "5 ROOM": (100, 135),
    "EXECUTIVE": (130, 165),
    "MULTI-GENERATION": (140, 175)
}

## title section
st.markdown("""
<div class="hero-panel">
    <div class="hero-title">HDB Resale Price Estimator</div>
    <div class="hero-subtitle">Enter the property details in the sidebar and generate an estimated resale price using the trained machine learning model.</div>
</div>
""", unsafe_allow_html=True)

## placeholder box for the price, gets filled in after prediction
plate_placeholder = st.empty()
plate_placeholder.markdown("""
<div class="block-plate">
    <div class="plate-label">Estimated Resale Price</div>
    <div class="plate-value">Enter details &amp; predict</div>
</div>
""", unsafe_allow_html=True)

st.divider()

## inputs go in the sidebar so the main area is free for the result
with st.sidebar:
    st.header("Property Details")
    town_selected = st.selectbox("Town", towns)
    flat_type_selected = st.selectbox("Flat Type", flat_types)
    floor_area_selected = st.slider("Floor Area (sqm)", 30, 200, 70)
    storey_range_selected = st.selectbox("Storey Range", storey_ranges)
    flat_model_selected = st.selectbox("Flat Model", flat_models)
    lease_commence_selected = st.slider("Lease Commence Year", 1966, 2022, 2000)
    predict_clicked = st.button("Predict HDB Price")

if predict_clicked:

    ## warn if the floor area looks off for the selected flat type doesn't give no output, just gives a heads up to the user that the predicted value might be inaccurate
    low, high = typical_floor_area[flat_type_selected]
    if not (low <= floor_area_selected <= high):
        st.warning(
            f"A {flat_type_selected} flat is typically between {low} and {high} sqm. "
            f"The floor area you entered ({floor_area_selected} sqm) is outside this range, "
            "so the prediction below may be less reliable."
        )

    with st.spinner("Generating prediction..."):

        try:
            ## numerical features
            input_row = {
                "floor_area_sqm": floor_area_selected,
                "lease_commence_date": lease_commence_selected
            }

            ## one-hot encode all of the non-numeric values, start at 0 then flip the selected one to 1
            for r in regions:
                input_row["region_" + r] = 0
            for ft in flat_types:
                input_row["flat_type_" + ft] = 0
            for fm in flat_models:
                input_row["flat_model_" + fm] = 0
            for sr in storey_ranges:
                input_row["storey_range_" + sr] = 0

            region_selected = region_dict[town_selected]

            input_row["region_" + region_selected] = 1
            input_row["flat_type_" + flat_type_selected] = 1
            input_row["flat_model_" + flat_model_selected] = 1
            input_row["storey_range_" + storey_range_selected] = 1

            df_input = pd.DataFrame([input_row])

            prediction = model.predict(df_input)[0]

        except Exception as e:
            st.error(
                "Something went wrong while generating the prediction. "
                f"Details: {e}"
            )
            st.stop()

    ## update the price box with the actual prediction
    plate_placeholder.markdown(f"""
    <div class="block-plate">
        <div class="plate-label">Estimated Resale Price</div>
        <div class="plate-value">${prediction:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

    st.success("Prediction generated successfully.")

    price_per_sqm = prediction / floor_area_selected
    st.metric("Estimated Price per sqm", f"${price_per_sqm:,.0f}")

    st.divider()
    st.subheader("Prediction Summary")

    ## showing the inputs as cards, split across 2 columns
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f'<div class="summary-card"><div class="label">Town</div><div class="value">{town_selected}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="summary-card"><div class="label">Region</div><div class="value">{region_selected}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="summary-card"><div class="label">Flat Type</div><div class="value">{flat_type_selected}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="summary-card"><div class="label">Storey Range</div><div class="value">{storey_range_selected}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="summary-card"><div class="label">Flat Model</div><div class="value">{flat_model_selected}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="summary-card"><div class="label">Floor Area</div><div class="value">{floor_area_selected} sqm</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="summary-card"><div class="label">Lease Commence Year</div><div class="value">{lease_commence_selected}</div></div>', unsafe_allow_html=True)

    st.divider()
    st.caption("Disclaimer: The predicted resale price is an estimate generated using a machine learning model and should not be treated as an official property valuation.")
