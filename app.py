import streamlit as st
import time
import json
from multiomics_pipeline import (
    DataRetriever,
    BioinformaticsPipeline,
    CNNInference,
    MultiOmicsFeatureExtractor,
    HealthStatePredictor
)

st.set_page_config(page_title="Multi-Omics Diagnostic Dashboard", layout="wide")

st.title("🔬 Multi-Omics Diagnostic Dashboard")
st.markdown("""
Welcome to the Multi-Omics Disease Detection interface.
This tool correlates **circulating biomarkers (ctDNA)** and **microbiome profiles** to infer the biological health state of the body using deep learning.

*Enter an Accession ID from the ENA database (e.g., `ERR1145155`, `ERR1145156`) to begin streaming and analyzing sequence fragments in real-time.*
""")

# Initialize pipeline modules once and cache them if possible
@st.cache_resource
def load_pipeline():
    return {
        "retriever": DataRetriever(),
        "bio_pipeline": BioinformaticsPipeline(),
        "cnn": CNNInference(),
        "feature_extractor": MultiOmicsFeatureExtractor(),
        "predictor": HealthStatePredictor()
    }

modules = load_pipeline()

# User Input Section
with st.sidebar:
    st.header("Patient Input")
    accession_id = st.text_input("Accession ID (ENA FastQ)", value="ERR1145155")

    st.markdown("### Physical Symptoms")
    symp_fatigue = st.checkbox("Fatigue")
    symp_weight_loss = st.checkbox("Weight Loss")
    symp_nausea = st.checkbox("Nausea")
    symp_pain = st.checkbox("Pain")

    run_btn = st.button("Run Multi-Omics Analysis", type="primary")

if run_btn:
    if not accession_id:
        st.error("Please enter a valid Accession ID.")
    else:
        # Build symptom string
        selected_symptoms = []
        if symp_fatigue: selected_symptoms.append("fatigue")
        if symp_weight_loss: selected_symptoms.append("weight loss")
        if symp_nausea: selected_symptoms.append("nausea")
        if symp_pain: selected_symptoms.append("pain")
        symptoms_str = ", ".join(selected_symptoms) if selected_symptoms else "None"

        with st.status("Running Multi-Omics Pipeline...", expanded=True) as status:
            st.write(f"📡 Querying ENA Database for `{accession_id}`...")
            # Step 1
            try:
                sample_data = modules["retriever"].download_sample(accession_id)
                st.write(f"✅ Downloaded and processed {sample_data['mock_reads_count']} reads.")
            except Exception as e:
                st.error(f"Failed to retrieve data: {e}")
                status.update(label="Pipeline Failed", state="error")
                st.stop()

            # Step 2
            st.write("🧬 Analyzing Nucleosome Footprinting (Spooling)...")
            tensor_features, spooling_stats = modules["bio_pipeline"].extract_features(sample_data)
            time.sleep(1) # For UI effect

            # Step 3
            st.write("🧠 Running 1D Convolutional Neural Network on fragment tensors...")
            cnn_score = modules["cnn"].predict(tensor_features)

            # Step 4
            st.write("🦠 Extracting Microbiome Profile and aggregating Symptoms...")
            combined_features = modules["feature_extractor"].combine_features(
                accession_id, spooling_stats, cnn_score, symptoms_str
            )
            time.sleep(1)

            # Step 5
            st.write("🌲 Running Random Forest Health State Predictor...")
            prediction_report = modules["predictor"].predict_health_state(combined_features, symptoms_str)

            status.update(label="Analysis Complete!", state="complete", expanded=False)

        # Display Results
        st.subheader("Diagnostic Results")

        col1, col2, col3 = st.columns(3)

        # Determine color for risk score
        risk_score = prediction_report["disease_risk_score"]
        if risk_score < 0.4:
            delta_color = "normal"
            risk_state = "Low Risk"
        elif risk_score < 0.7:
            delta_color = "off"
            risk_state = "Moderate Risk"
        else:
            delta_color = "inverse"
            risk_state = "High Risk"

        with col1:
            st.metric(label="Disease Risk Score", value=prediction_report["disease_risk_percentage"], delta=risk_state, delta_color=delta_color)
        with col2:
            st.metric(label="Predicted State", value=prediction_report["predicted_disease_type"].replace("_", " "))
        with col3:
            st.metric(label="Analyzed Sequences", value=sample_data['mock_reads_count'])

        st.markdown("### 🔍 Early Biomarkers Identified")
        for biomarker in prediction_report["early_biomarkers"]:
            st.info(biomarker)

        st.markdown("### 📊 Raw Multi-Omics Feature Matrix")
        with st.expander("View Feature JSON"):
            st.json(combined_features)
