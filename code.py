import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Cantilever Beam ML App", page_icon="📈", layout="wide")

st.title("Cantilever Beam Slot Optimization (ML Surrogate Model)")
st.caption("Predict Resonance Frequency, Deflection, and Voltage using Random Forest / XGBoost")

# -----------------------------
# ABOUT SECTION (FROM PDF)
# -----------------------------
with st.expander(" About this Project (Read first)", expanded=True):

    # Create 2 columns
    left_col, right_col = st.columns([3, 1])  # text larger, image smaller

    with left_col:
        st.markdown(
        """
### What is this project about?


**Comparative Analysis of ML Models for Optimization of Tapered Slot Dimensions in Cantilever Beam for Vibration Energy Harvesting.** 
A **Piezoelectric Energy Harvester (PEH)** converts ambient vibrations into electrical energy.
The PEH is typically a **cantilever beam** with a piezoelectric layer.

In your report:
- Performed **Finite Element Simulation (ANSYS Workbench)** to generate data
- Varied the **trapezoidal slot dimensions on the length of a cantilever beam**
- Then trained ML models (**Random Forest & XGBoost**) to act as a **fast surrogate model** instead of running ANSYS every time 

---

###  Why we need Machine Learning here?

ANSYS simulation is accurate but **time-consuming** when exploring many combinations of slot dimensions.
To solve this:
- You generated **310 datasets** by parametric study
- Trained ML models to predict:
   Resonance Frequency  
   Deflection  
   Voltage  

The report shows that ML achieves high prediction accuracy:
- Frequency & Voltage: **R² ≈ 0.97–0.99**
- XGBoost performs better for deflection

---

###  How to input data in this app?

You will input the **slot dimensions** of the trapezoidal hollow structure:

| Input Parameter | Meaning |
|---|---|
| Length | Beam length (mm) |
| breadth_s | smaller width of slot (mm) |
| breadth_l | larger width of slot (mm) |

These are the same parameters used in your dataset generation methodology.

---

###  What outputs will you get?

The model will predict:

1) **Resonance Frequency (Hz)**  
   → Lower frequency means the harvester can work at more realistic environmental vibrations. 

2) **Deflection (mm / µm)**  
   → Higher deflection increases strain on piezo layer.

3) **Voltage (V)**  
   → Voltage output increases with strain/deflection (piezoelectric effect). 

---

###  ML workflow used in this app (Chained prediction)

- First predict Frequency + Deflection using geometry inputs
- Then use predicted Deflection to predict Voltage 
**Pipeline:**

---

###  Recommended input range

For best predictions:
- Use values within the dataset range (training limits)
- Avoid extreme values outside the dataset
        """
    )
    
    with right_col:
        st.markdown("### Beam Geometry")
        st.image("beam.png", caption="Trapezoidal Slot in Cantilever Beam", use_container_width=True)

# -----------------------------
# SIDEBAR SETTINGS
# -----------------------------
st.sidebar.header("⚙️ Model Settings")

model_choice = st.sidebar.selectbox("Select ML Model", ["Random Forest", "XGBoost"])
test_size = st.sidebar.slider("Test Size", 0.05, 0.30, 0.05, 0.01)
random_state = st.sidebar.number_input("Random State", value=42, step=1)

st.sidebar.markdown("---")
st.sidebar.header(" Input Slot Dimensions")

L = st.sidebar.number_input("Length (mm)", value=13.5)
bs = st.sidebar.number_input("breadth_s (mm)", value=2.5)
bl = st.sidebar.number_input("breadth_l (mm)", value=3.3)

# -----------------------------
# UTILS
# -----------------------------
def train_evaluate_predict(model_type, features, target, test_size=0.05, random_state=42):

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=random_state
    )

    if model_type == "Random Forest":
        model = RandomForestRegressor(n_estimators=300, max_depth=50, random_state=random_state)

    elif model_type == "XGBoost":
        model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=50,
            learning_rate=0.9,
            objective='reg:squarederror',
            random_state=random_state
        )
    else:
        raise ValueError("Invalid model type")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)

    return model, score, y_test, y_pred


def create_scatter_plot(y_test, y_pred, title, xlabel, ylabel, r2_value):
    fig = plt.figure(figsize=(7, 5))
    plt.scatter(y_test, y_pred, alpha=0.7)

    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)

    plt.text(
        0.05, 0.9, f"R² Score: {r2_value:.4f}",
        transform=plt.gca().transAxes,
        fontsize=12,
        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.5)
    )

    plt.tight_layout()
    return fig


# -----------------------------
# DATA LOADING
# -----------------------------
st.subheader("📂 Dataset Loading")
uploaded_file = st.file_uploader("Upload dataset CSV file (same format used in ANSYS parametric study)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    try:
        df = pd.read_csv("Datasets.csv")
        st.info("Using local file: Datasets.csv")
    except Exception:
        st.warning("Upload a CSV file or place 'Datasets.csv' in same folder as app.py")
        st.stop()

with st.expander("🔍 View Dataset"):
    st.dataframe(df)

required_cols = ["Length", "breadth_s", "breadth_l", "Resonance Frequency", "Deflection", "Voltage"]
for col in required_cols:
    if col not in df.columns:
        st.error(f"Missing required column: {col}")
        st.stop()

# -----------------------------
# TRAIN MODELS
# -----------------------------
st.subheader(" Train ML Models")

features_geom = df[["Length", "breadth_s", "breadth_l"]]
target_freq = df["Resonance Frequency"]
target_def = df["Deflection"]

features_deflection = df[["Deflection"]]
target_volt = df["Voltage"]

train_button = st.button("✅ Train Models")

if train_button:
    with st.spinner("Training models..."):
        model_freq, r2_freq, y_test_freq, y_pred_freq = train_evaluate_predict(
            model_choice, features_geom, target_freq, test_size=test_size, random_state=random_state
        )

        model_def, r2_def, y_test_def, y_pred_def = train_evaluate_predict(
            model_choice, features_geom, target_def, test_size=test_size, random_state=random_state
        )

        model_volt, r2_volt, y_test_volt, y_pred_volt = train_evaluate_predict(
            model_choice, features_deflection, target_volt, test_size=test_size, random_state=random_state
        )

    st.success("✅ Training Completed!")

    col1, col2, col3 = st.columns(3)
    col1.metric("R² Resonance Frequency", f"{r2_freq:.4f}")
    col2.metric("R² Deflection", f"{r2_def:.4f}")
    col3.metric("R² Voltage", f"{r2_volt:.4f}")

    # PLOTS
    st.subheader("📊 Model Performance Plots")

    p1, p2, p3 = st.columns(3)

    fig1 = create_scatter_plot(y_test_freq, y_pred_freq,
                               "Resonance Frequency",
                               "Actual", "Predicted", r2_freq)

    fig2 = create_scatter_plot(y_test_def, y_pred_def,
                               "Deflection",
                               "Actual", "Predicted", r2_def)

    fig3 = create_scatter_plot(y_test_volt, y_pred_volt,
                               "Voltage",
                               "Actual", "Predicted", r2_volt)

    with p1:
        st.pyplot(fig1)
    with p2:
        st.pyplot(fig2)
    with p3:
        st.pyplot(fig3)

    # -----------------------------
    # PREDICTION SECTION
    # -----------------------------
    st.subheader("🎯 Predictions for User Input")

    new_data = pd.DataFrame({"Length": [L], "breadth_s": [bs], "breadth_l": [bl]})

    pred_freq = model_freq.predict(new_data)[0]
    pred_def = model_def.predict(new_data)[0]

    # Adjustments used in your XGBoost script
    if model_choice == "XGBoost":
        pred_def = pred_def - 0.2

    pred_volt = model_volt.predict(np.array(pred_def).reshape(-1, 1))[0]

    if model_choice == "XGBoost":
        pred_volt = pred_volt + 9

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Resonance Frequency (Hz)", f"{pred_freq:.4f}")
    c2.metric("Predicted Deflection", f"{pred_def:.4f}")
    c3.metric("Predicted Voltage (V)", f"{pred_volt:.4f}")

    st.info(
        "Tip: Use values inside dataset range for best accuracy. "
        "If you input extreme values beyond dataset, ML may give unreliable results."
    )

