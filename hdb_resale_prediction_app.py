## HDB Resale Price Prediction Streamlit App
## This app collects housing details from the user and predicts the resale price using a trained machine learning model.
## Import the required libraries for loading the model and building the app interface.
import joblib
import streamlit as st
import pandas as pd

## Configure the page. This must be the first Streamlit command in the script.
st.set_page_config(
    page_title="HDB Resale Price Estimator",
    layout="wide"
)

## Inject custom CSS for a distinctive look based on real HDB visual language: a dark charcoal background and a "block number signplate" style for the headline prediction figure. No external font is loaded here, since that would require the browser to reach out to an external link - instead this uses fonts that are already built into every browser/operating system.
st.markdown("""
<style>

/* All colors below are written directly as hex codes rather than CSS variables, so every rule is self-contained and easy to read on its own. 
Background: #14171C (charcoal). Panels/cards: #1C2029. Price box: #2B2438 (deep plum). Accent: #F17B33 (orange), used for borders and the button. Text: #F4F1E9 (warm off-white). */

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

/* The hero section that introduces the app. Uses a background panel and larger type to feel prominent, but deliberately has no border this time, since the boxed/bordered look was flagged as too busy earlier. */
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

/* The box that displays the estimated price. Uses the same background color as the hero panel above, so the two feel like one consistent dark theme rather than two different shades of dark. */
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
/* Hover uses a darker shade of the same orange instead of white, so the button does not flash bright on a dark background. */
div.stButton > button:hover {
    background-color: #C85F22;
    color: #FFFFFF;
}

[data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
    color: #F4F1E9 !important;
}

/* Streamlit's default success/warning/error boxes are green/yellow/red, which clash with the dark theme and look too much like a default AI-app template. This overrides all of them to the same dark panel style with a thin orange left edge, so they stay consistent no matter which type of message is shown. Several selector variants are included, since the internal element names for this component have changed across Streamlit versions - only one of these needs to match for the override to work. */
div[data-testid="stAlert"],
div[data-testid="stNotification"],
div[data-baseweb="notification"],
.stAlert,
[class*="stAlert"] {
    background-color: #1C2029 !important;
    background: #1C2029 !important;
    border: none !important;
    border-left: 3px solid #F17B33 !important;
    border-radius: 0 !important;
}
div[data-testid="stAlert"] *,
div[data-testid="stNotification"] *,
div[data-baseweb="notification"] *,
.stAlert *,
[class*="stAlert"] * {
    background-color: transparent !important;
    color: #F4F1E9 !important;
}
/* The success/warning/error icons are SVGs that were still showing green/yellow/red on their own, separate from the box background above. This forces every alert icon to the same orange accent. */
div[data-testid="stAlert"] svg,
div[data-testid="stNotification"] svg,
div[data-baseweb="notification"] svg,
.stAlert svg {
    fill: #F17B33 !important;
    color: #F17B33 !important;
}

/* Streamlit adds its own links automatically: a small chain-link icon next to every heading (for jumping to that section), plus a hamburger menu and footer with links to Streamlit's own site/docs/GitHub. All of that is hidden here so no clickable links appear anywhere in the app. */
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

## Load the trained machine learning model.
## Cached so the model file is only read from disk once per session, not on every rerun.
## Stop the app immediately if the model file is missing.
@st.cache_resource
def load_model():
    return joblib.load("hdb_final_model.pkl")

try:
    model = load_model()
except FileNotFoundError:
    st.error("The trained prediction model could not be found. Please ensure 'hdb_final_model.pkl' is in the same folder as this application.")
    st.stop()
except Exception as e:
    ## Catch anything else that goes wrong while loading the model
    ## (e.g. a corrupted file or a version mismatch) so the app fails gracefully.
    st.error(f"The prediction model could not be loaded due to an unexpected error: {e}")
    st.stop()

## Define the categorical options shown in the dropdowns and sliders.
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

## Map each town to the broader region expected by the trained model.
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

## Create a sorted list of town names for the dropdown menu.
towns = sorted(region_dict.keys())

## Typical floor area range (sqm) for each flat type, used only for input validation. These are approximate real-world ranges, not hard model limits, so an out-of-range combination is flagged as a warning rather than blocked.
typical_floor_area = {
    "1 ROOM": (28, 45),
    "2 ROOM": (35, 50),
    "3 ROOM": (50, 75),
    "4 ROOM": (75, 105),
    "5 ROOM": (100, 135),
    "EXECUTIVE": (130, 165),
    "MULTI-GENERATION": (140, 175)
}

## Build the main page hero section with a background panel
st.markdown("""
<div class="hero-panel">
    <div class="hero-title">HDB Resale Price Estimator</div>
    <div class="hero-subtitle">Enter the property details in the sidebar and generate an estimated resale price using the trained machine learning model.</div>
</div>
""", unsafe_allow_html=True)


## Placeholder container for the price display "plate" shown in the main area
plate_placeholder = st.empty()
plate_placeholder.markdown("""
<div class="block-plate">
    <div class="plate-label">Estimated Resale Price</div>
    <div class="plate-value">Enter details &amp; predict</div>
</div>
""", unsafe_allow_html=True)

st.divider()

## Gather the user's property details using input widgets in the sidebar.
## Keeping inputs in the sidebar leaves the main area free for the result.
with st.sidebar:
    st.header("Property Details")
    town_selected = st.selectbox("Town", towns)
    flat_type_selected = st.selectbox("Flat Type", flat_types)
    floor_area_selected = st.slider("Floor Area (sqm)", 30, 200, 70)
    storey_range_selected = st.selectbox("Storey Range", storey_ranges)
    flat_model_selected = st.selectbox("Flat Model", flat_models)
    lease_commence_selected = st.slider("Lease Commence Year", 1966, 2022, 2000)
    predict_clicked = st.button("Predict HDB Price")

## Running the prediction workflow when the user clicks the button.
if predict_clicked:

    ## Input validation: flag unusual flat type / floor area combinations.
    ## This does not block the prediction, since the model can still produce an output, but it warns the user that the combination is atypical and the estimate may be less reliable as a result.
    low, high = typical_floor_area[flat_type_selected]
    if not (low <= floor_area_selected <= high):
        st.warning(
            f"A {flat_type_selected} flat is typically between {low} and {high} sqm. "
            f"The floor area you entered ({floor_area_selected} sqm) is outside this range, "
            "so the prediction below may be less reliable."
        )

    with st.spinner("Generating prediction..."):

        try:
            ## Creating a dictionary with the numerical input features.
            input_row = {
                "floor_area_sqm": floor_area_selected,
                "lease_commence_date": lease_commence_selected
            }

            ## Creating one-hot encoded features for every categorical option.
            ## All values start at 0 and the selected choice is changed to 1.
            for r in regions:
                input_row["region_" + r] = 0
            for ft in flat_types:
                input_row["flat_type_" + ft] = 0
            for fm in flat_models:
                input_row["flat_model_" + fm] = 0
            for sr in storey_ranges:
                input_row["storey_range_" + sr] = 0

            ## Converting the selected town into its mapped region for prediction.
            region_selected = region_dict[town_selected]

            ## Setting the chosen categories to 1 so the model receives the expected input format.
            input_row["region_" + region_selected] = 1
            input_row["flat_type_" + flat_type_selected] = 1
            input_row["flat_model_" + flat_model_selected] = 1
            input_row["storey_range_" + storey_range_selected] = 1

            ## Converting the completed input dictionary into a DataFrame for the model.
            df_input = pd.DataFrame([input_row])

            ## Generating the resale price prediction.
            prediction = model.predict(df_input)[0]

        except Exception as e:
            ## Catch any failure during feature preparation or prediction (e.g. a feature mismatch) and show a clear, user-facing message instead of letting the app crash.
            st.error(
                "Something went wrong while generating the prediction. "
                f"Details: {e}"
            )
            st.stop()

    ## Fill in the block signplate with the actual predicted price.
    plate_placeholder.markdown(f"""
    <div class="block-plate">
        <div class="plate-label">Estimated Resale Price</div>
        <div class="plate-value">${prediction:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

    st.success("Prediction generated successfully.")

    ## Show the price per square metre as an extra, easy-to-scan data point.
    price_per_sqm = prediction / floor_area_selected
    st.metric("Estimated Price per sqm", f"${price_per_sqm:,.0f}")

    st.divider()
    st.subheader("Prediction Summary")

    ## Displaying the selected inputs as cards, in two columns for easier reading.
    ## Each st.markdown call below renders one "summary-card" div styled in
    ## the CSS block earlier, with a small label on top and the value below it.
    c1, c2 = st.columns(2)
    with c1:
        ## Town card
        st.markdown(f'<div class="summary-card"><div class="label">Town</div><div class="value">{town_selected}</div></div>', unsafe_allow_html=True)
        ## Region card, derived from the town via region_dict
        st.markdown(f'<div class="summary-card"><div class="label">Region</div><div class="value">{region_selected}</div></div>', unsafe_allow_html=True)
        ## Flat type card
        st.markdown(f'<div class="summary-card"><div class="label">Flat Type</div><div class="value">{flat_type_selected}</div></div>', unsafe_allow_html=True)
        ## Storey range card
        st.markdown(f'<div class="summary-card"><div class="label">Storey Range</div><div class="value">{storey_range_selected}</div></div>', unsafe_allow_html=True)
    with c2:
        ## Flat model card
        st.markdown(f'<div class="summary-card"><div class="label">Flat Model</div><div class="value">{flat_model_selected}</div></div>', unsafe_allow_html=True)
        ## Floor area card
        st.markdown(f'<div class="summary-card"><div class="label">Floor Area</div><div class="value">{floor_area_selected} sqm</div></div>', unsafe_allow_html=True)
        ## Lease commence year card
        st.markdown(f'<div class="summary-card"><div class="label">Lease Commence Year</div><div class="value">{lease_commence_selected}</div></div>', unsafe_allow_html=True)

    st.divider()
    ## Add a disclaimer so users understand the prediction is only an estimate.
    st.caption("Disclaimer: The predicted resale price is an estimate generated using a machine learning model and should not be treated as an official property valuation.")
