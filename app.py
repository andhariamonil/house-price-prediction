# Build a Streamlit dashboard for predicting house prices

import streamlit as st
import pandas as pd
import joblib


st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)


@st.cache_resource
def load_model():
    # Load the trained machine learning pipeline
    return joblib.load("models/house_price_pipeline.pkl")


model = load_model()


st.title("🏠 House Price Prediction")
st.write("Enter the house details below to predict its estimated price.")


st.sidebar.header("House Details")


area = st.sidebar.number_input(
    "Area",
    min_value=1.0,
    value=1500.0
)


bedrooms = st.sidebar.number_input(
    "Bedrooms",
    min_value=0,
    value=3,
    step=1
)


bathrooms = st.sidebar.number_input(
    "Bathrooms",
    min_value=0.0,
    value=2.0,
    step=1.0
)


floors = st.sidebar.number_input(
    "Floors",
    min_value=1.0,
    value=1.0,
    step=1.0
)


year_built = st.sidebar.number_input(
    "Year Built",
    min_value=1800,
    max_value=2026,
    value=2015,
    step=1
)


garage = st.sidebar.number_input(
    "Garage",
    min_value=0,
    value=1,
    step=1
)


location = st.sidebar.text_input(
    "Location",
    value="Ahmedabad"
)


condition = st.sidebar.selectbox(
    "Condition",
    [
        "New",
        "Good",
        "Average",
        "Old"
    ]
)


if st.sidebar.button("Predict Price"):

    house_age = 2026 - year_built

    total_rooms = bedrooms + bathrooms

    area_per_bedroom = (
        area / bedrooms
        if bedrooms > 0
        else 0
    )

    input_data = pd.DataFrame({
        "Area": [area],
        "Bedrooms": [bedrooms],
        "Bathrooms": [bathrooms],
        "Floors": [floors],
        "YearBuilt": [year_built],
        "Location": [location],
        "Condition": [condition],
        "Garage": [garage],
        "HouseAge": [house_age],
        "TotalRooms": [total_rooms],
        "AreaPerBedroom": [area_per_bedroom]
    })

    prediction = model.predict(input_data)[0]

    st.success("Prediction generated successfully!")

    st.metric(
        "Estimated House Price",
        f"{prediction:,.2f}"
    )


st.divider()


st.subheader("Entered House Information")


display_data = pd.DataFrame({
    "Feature": [
        "Area",
        "Bedrooms",
        "Bathrooms",
        "Floors",
        "Year Built",
        "Location",
        "Condition",
        "Garage"
    ],
    "Value": [
        area,
        bedrooms,
        bathrooms,
        floors,
        year_built,
        location,
        condition,
        garage
    ]
})


st.dataframe(
    display_data,
    use_container_width=True,
    hide_index=True
)


st.info(
    "The prediction is generated using the trained machine learning pipeline "
    "saved in models/house_price_pipeline.pkl."
)