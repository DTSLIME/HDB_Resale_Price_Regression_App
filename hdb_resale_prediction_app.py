# HDB Resale Price Prediction Streamlit App
# This app collects housing details from the user and predicts the resale price using a trained machine learning model.

import joblib
import streamlit as st
import pandas as pd

## Import the required libraries for loading the model and building the app interface.

## Load the trained machine learning model.
## Stop the app immediately if the model file is missing.
try:
    model = joblib.load("hdb_final_model.pkl")
except FileNotFoundError:
    st.error("The trained prediction model could not be found. Please ensure 'hdb_final_model.pkl' is in the same folder as this application.")
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

## Build the main page title and intro text for the app.
st.title("HDB Resale Price Prediction")
st.write("Enter the property details below to estimate the resale price using the trained machine learning model.")
st.divider()

## Gather the user's property details using input widgets such as dropdowns and sliders.
left, right = st.columns(2)
with left:
    town_selected = st.selectbox("Select Town", towns)
    flat_type_selected = st.selectbox("Select Flat Type", flat_types)
    floor_area_selected = st.slider("Floor Area (sqm)", 30, 200, 70)

with right:
    storey_range_selected = st.selectbox("Select Storey Range", storey_ranges)
    flat_model_selected = st.selectbox("Select Flat Model", flat_models)
    lease_commence_selected = st.slider("Lease Commence Year", 1966, 2022, 2000)
## This adds a horizontal line to divide the prediction button from the column
st.divider()

## Running the prediction workflow when the user clicks the button.
if st.button("Predict HDB Price"):

    with st.spinner("Generating prediction..."):

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

    ## Displaying the prediction result to the user.
    st.success("Prediction generated successfully.")
    st.metric("Estimated HDB Resale Price", f"${prediction:,.2f}")

    st.divider()
    st.subheader("Prediction Summary")

    ## Displaying the selected inputs in two columns for easier reading.
    c1, c2 = st.columns(2)
    with c1:
        st.write(f"**Town:** {town_selected}")
        st.write(f"**Region:** {region_selected}")
        st.write(f"**Flat Type:** {flat_type_selected}")
        st.write(f"**Storey Range:** {storey_range_selected}")
    with c2:
        st.write(f"**Flat Model:** {flat_model_selected}")
        st.write(f"**Floor Area:** {floor_area_selected} sqm")
        st.write(f"**Lease Commence Year:** {lease_commence_selected}")

    st.divider()
    ## Add a disclaimer so users understand the prediction is only an estimate.
    st.caption("Disclaimer: The predicted resale price is an estimate generated using a machine learning model and should not be treated as an official property valuation.")