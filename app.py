import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(
    page_title="IoT Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
)

# -------------------------------
# Load trained model
# -------------------------------
MODEL_PATH = (
    Path(__file__).parent
    / "outputs"
    / "predictive_maintenance_model.pkl"
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

# -------------------------------
# Dashboard header
# -------------------------------
st.title("⚙️ IoT Predictive Maintenance Dashboard")
st.write(
    "Monitor machine sensor readings and predict potential failures "
    "using a trained Random Forest classifier."
)

st.divider()

# -------------------------------
# Sensor inputs
# -------------------------------
st.subheader("📡 Machine Sensor Readings")

col1, col2 = st.columns(2)

with col1:
    temperature = st.slider(
        "Temperature (°C)",
        min_value=20.0,
        max_value=110.0,
        value=65.0,
        step=1.0,
    )

    vibration = st.slider(
        "Vibration",
        min_value=0.0,
        max_value=15.0,
        value=5.0,
        step=0.1,
    )

with col2:
    pressure = st.slider(
        "Pressure",
        min_value=40.0,
        max_value=140.0,
        value=100.0,
        step=1.0,
    )

    operating_hours = st.number_input(
        "Operating Hours",
        min_value=100,
        max_value=10000,
        value=2000,
        step=100,
    )

st.divider()

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔍 Predict Machine Condition", type="primary"):
    try:
        model = load_model()

        # Preserve the feature order used during training
        input_data = pd.DataFrame(
            [[temperature, vibration, pressure, operating_hours]],
            columns=[
                "temperature",
                "vibration",
                "pressure",
                "operating_hours",
            ],
        )

        prediction = int(model.predict(input_data)[0])

        st.subheader("📊 Prediction Result")

        metric1, metric2 = st.columns(2)

        with metric1:
            st.metric("Temperature", f"{temperature:.1f} °C")
            st.metric("Vibration", f"{vibration:.1f}")

        with metric2:
            st.metric("Pressure", f"{pressure:.1f}")
            st.metric("Operating Hours", f"{operating_hours:,}")

        if prediction == 1:
            st.error(
                "⚠️ Potential Machine Failure Predicted"
            )
            st.warning(
                "The model predicts the failure class. "
                "Inspect the machine and verify the readings."
            )
        else:
            st.success(
                "✅ No Machine Failure Predicted"
            )
            st.info(
                "The model predicts the non-failure class "
                "for these sensor readings."
            )

        # Display probability only if the model supports it
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            classes = list(model.classes_)

            if 1 in classes:
                failure_probability = probabilities[classes.index(1)]
                st.subheader("Estimated Failure Probability")
                st.progress(float(failure_probability))
                st.write(
                    f"Model-estimated probability: "
                    f"{failure_probability:.1%}"
                )
                st.caption(
                    "This probability is model-dependent and is not "
                    "a validated estimate of real-world failure risk."
                )

    except FileNotFoundError:
        st.error(
            "Model file not found. Check that "
            "'outputs/predictive_maintenance_model.pkl' exists."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")

st.divider()

st.caption(
    "Academic demonstration using simulated sensor data. "
    "Predictions are not a substitute for industrial safety systems."
)
