import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import r2_score, mean_absolute_error

# Page configuration
st.set_page_config(
    page_title="Vehicle CO2 Emission Predictor",
    layout="centered",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_model():
    try:
        model = joblib.load("vehicle_co2_emission_model.pkl")
        return model
    except FileNotFoundError:
        st.error("Model file 'vehicle_co2_emission_model.pkl' not found. Please ensure the file is in the project root.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()


@st.cache_data
def load_dataset_head():
    try:
        df = pd.read_csv("data/vehicle_emissions.csv")
        return df
    except FileNotFoundError:
        return None


def main():
    st.title("Vehicle CO2 Emission Predictor")
    st.markdown("Enter vehicle details below and click Predict to estimate CO2 emissions (g/km).")

    df = load_dataset_head()

    # (Metrics will be computed and shown after the user clicks Predict)

    # Prepare default options
    makes = sorted(df['Make'].unique()) if df is not None else []
    models = sorted(df['Model'].unique()) if df is not None else []
    classes = sorted(df['Vehicle_Class'].unique()) if df is not None else []
    transmissions = sorted(df['Transmission'].unique()) if df is not None else []

    col1, col2 = st.columns(2)

    with col1:
        model_year = st.number_input("Model Year", min_value=1980, max_value=2030, value=2021, step=1)
        make = st.selectbox("Make", options=makes, index=0 if makes else 0)
        model_name = st.selectbox("Model", options=models, index=0 if models else 0)
        vehicle_class = st.selectbox("Vehicle Class", options=classes, index=0 if classes else 0)
        engine_size = st.number_input("Engine Size (L)", min_value=0.0, max_value=10.0, value=2.0, step=0.1, format="%.1f")
        cylinders = st.number_input("Cylinders", min_value=1, max_value=16, value=4, step=1)

    with col2:
        transmission = st.selectbox("Transmission", options=transmissions, index=0 if transmissions else 0)
        fccity = st.number_input("Fuel Consumption in City (L/100 km)", min_value=0.0, max_value=50.0, value=9.0, step=0.1, format="%.1f")
        fchwy = st.number_input("Fuel Consumption Hwy (L/100 km)", min_value=0.0, max_value=50.0, value=7.0, step=0.1, format="%.1f")
        fccomb = st.number_input("Fuel Consumption Combined (L/100km)", min_value=0.0, max_value=50.0, value=8.0, step=0.1, format="%.1f")
        smog_level = st.number_input("Smog Level", min_value=0, max_value=10, value=3, step=1)

    st.markdown("---")

    # Custom CSS for the button
    st.markdown("""
    <style>
    .stButton>button {
        padding: 5px !important;
        background-color: #4CAF50 !important;
        color: white !important;
        border-radius: 5px !important;
        transition: background-color 0.3s !important;
        margin: 0 auto !important;
        display: block !important;
    }
    .stButton>button:hover {
        background-color: #45a049 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    if st.button("Predict CO2 Emissions", key="predict_button"):
        model = load_model()

        # Compute and display dataset metrics if possible
        metric_cols = [
            "Model_Year",
            "Make",
            "Model",
            "Vehicle_Class",
            "Engine_Size",
            "Cylinders",
            "Transmission",
            "Fuel_Consumption_in_City(L/100 km)",
            "Fuel_Consumption_in_City_Hwy(L/100 km)",
            "Fuel_Consumption_comb(L/100km)",
            "Smog_Level",
        ]

        if df is not None and set(metric_cols).issubset(df.columns) and "CO2_Emissions" in df.columns:
            try:
                X = df[metric_cols]
                y = df["CO2_Emissions"]
                preds = model.predict(X)
                r2 = r2_score(y, preds)
                mae = mean_absolute_error(y, preds)
                # Display metrics in a single styled card
                st.markdown(f"""
                    <div style="padding: 20px; border-radius: 10px; margin: 10px 0;">
                        <h4 style="margin-bottom: 15px; color: #fffff;">Model Performance Metrics</h4>
                        <div style="display: flex; justify-content: space-around;">
                            <div style="text-align: center;">
                                <div style="font-size: 2rem; font-weight: bold; color: #4CAF50;">{r2*100:.2f}%</div>
                                <div style="font-size: 0.9rem; color: #666;">Model R² (Accuracy)</div>
                            </div>
                            <div style="text-align: center;">
                                <div style="font-size: 2rem; font-weight: bold; color: #FF9800;">{mae:.2f}</div>
                                <div style="font-size: 0.9rem; color: #666;">Mean Absolute Error (g/km)</div>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            except Exception:
                st.info("Could not compute model metrics on the dataset.")

        # Build input row with exact column names used in training
        input_row = {
            "Model_Year": int(model_year),
            "Make": make,
            "Model": model_name,
            "Vehicle_Class": vehicle_class,
            "Engine_Size": float(engine_size),
            "Cylinders": int(cylinders),
            "Transmission": transmission,
            "Fuel_Consumption_in_City(L/100 km)": float(fccity),
            "Fuel_Consumption_in_City_Hwy(L/100 km)": float(fchwy),
            "Fuel_Consumption_comb(L/100km)": float(fccomb),
            "Smog_Level": int(smog_level),
        }

        input_df = pd.DataFrame([input_row])

        try:
            pred = model.predict(input_df)
            # If prediction returns array-like
            co2 = float(pred[0]) if hasattr(pred, '__len__') else float(pred)
            st.success(f"Predicted CO2 Emissions: {co2:.0f} g/km")
            with st.expander("Input used for prediction"):
                st.dataframe(input_df, use_container_width=True)
        except Exception as e:
            st.error(f"Prediction failed: {e}")


if __name__ == '__main__':
    main()

