## Importing the libraries
import joblib
import streamlit as st
import numpy as np
import pandas as pd

## Loading the trained model
model = joblib.load("hdb_final_model.pkl")

## Using the same data used to build the training data
regions = ["Central", "East", "North", "North-East", "West"]
flat_types = ["1 ROOM", "2 ROOM", "3 ROOM", "4 ROOM", "5 ROOM", "EXECUTIVE", "MULTI-GENERATION"]
storey_ranges = ["01 TO 03", "04 TO 06", "07 TO 09", "10 TO 12", "13 TO 15", "16 TO 18",
                  "19 TO 21", "22 TO 24", "25 TO 27", "28 TO 30", "31 TO 33", "34 TO 36",
                  "37 TO 39", "40 TO 42", "43 TO 45", "46 TO 48", "49 TO 51"]
flat_models = ["2-room", "3Gen", "Adjoined flat", "Apartment", "DBSS", "Improved",
               "Improved-Maisonette", "Maisonette", "Model A", "Model A-Maisonette",
               "Model A2", "Multi Generation", "New Generation", "Premium Apartment",
               "Premium Apartment Loft", "Premium Maisonette", "Simplified", "Standard",
               "Terrace", "Type S1", "Type S2"]

## Same region mapping used in the notebook, so the town the user picks can be converted into the region the model was actually trained on
region_dict = {
    "ANG MO KIO": "North-East",
    "BEDOK": "East",
    "BISHAN": "Central",
    "BUKIT BATOK": "West",
    "BUKIT MERAH": "Central",
    "BUKIT PANJANG": "West",
    "BUKIT TIMAH": "Central",
    "CENTRAL AREA": "Central",
    "CHOA CHU KANG": "West",
    "CLEMENTI": "West",
    "GEYLANG": "Central",
    "HOUGANG": "North-East",
    "JURONG EAST": "West",
    "JURONG WEST": "West",
    "KALLANG/WHAMPOA": "Central",
    "MARINE PARADE": "Central",
    "PASIR RIS": "East",
    "PUNGGOL": "North-East",
    "QUEENSTOWN": "Central",
    "SEMBAWANG": "North",
    "SENGKANG": "North-East",
    "SERANGOON": "North-East",
    "TAMPINES": "East",
    "TOA PAYOH": "Central",
    "WOODLANDS": "North",
    "YISHUN": "North"
}
towns = sorted(region_dict.keys())

## Streamlit app title
st.title("HDB Resale Price Prediction")

## User inputs (buttons)
town_selected = st.selectbox("Select Town", towns)
flat_type_selected = st.selectbox("Select Flat Type", flat_types)
storey_range_selected = st.selectbox("Select Storey Range", storey_ranges)
flat_model_selected = st.selectbox("Select Flat Model", flat_models)

## User inputs (sliders)
floor_area_selected = st.slider("Floor Area (sqm)",
min_value=30,
max_value=200,
value=70)

lease_commence_selected = st.slider("Lease Commence Date",
min_value=1966,
max_value=2022,
value=2000)

## The prediction button
if st.button("Predict HDB price"):

    ## Build one dictionary entry for every column the model was trained on,
    ## starting all the one-hot columns at 0
    input_row = {
        "floor_area_sqm": floor_area_selected,
        "lease_commence_date": lease_commence_selected,
    }

    for r in regions:
        input_row["region_" + r] = 0
    for ft in flat_types:
        input_row["flat_type_" + ft] = 0
    for fm in flat_models:
        input_row["flat_model_" + fm] = 0
    for sr in storey_ranges:
        input_row["storey_range_" + sr] = 0

    ## Set only the selected category's column to 1, the model was one-hot encoded without drop_first, so every category including the one selected here - always has its own real column, with no missing "reference category" case to handle
    region_selected = region_dict[town_selected]
    input_row["region_" + region_selected] = 1
    input_row["flat_type_" + flat_type_selected] = 1
    input_row["flat_model_" + flat_model_selected] = 1
    input_row["storey_range_" + storey_range_selected] = 1

    ## Convert input data to a DataFrame
    df_input = pd.DataFrame([input_row])

    ## Predict
    y_unseen_pred = model.predict(df_input)[0]
    st.success(f"Predicted Price: ${y_unseen_pred:,.2f}")
