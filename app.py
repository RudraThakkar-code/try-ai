import streamlit as st
import pandas as pd
from multiomics_pipeline import run_pipeline_for_id

st.set_page_config(page_title="Multi-Omics Search Portal", layout="wide")

st.title("🧬 Multi-Omics Disease Detection Portal")
st.markdown("Enter an ENA Accession ID to fetch data and run the diagnostic pipeline.")

# 1. User Input
accession_id = st.text_input("Enter Accession Number (e.g., ERR1145155)", "ERR1145155")

if st.button("Search & Analyze"):
    if accession_id:
        with st.spinner(f"Searching databases for {accession_id}..."):
            # 2. Call the logic
            prediction_report, combined_features = run_pipeline_for_id(accession_id)

            if not prediction_report:
                st.error("Failed to run pipeline. Check the backend logs or ensure the accession ID is correct.")
            else:
                # 3. Display Results in Tabs
                tab1, tab2 = st.tabs(["Diagnostic Report", "Raw Data Summary"])

                with tab1:
                    st.subheader("Final Diagnostic Prediction")

                    risk_score = prediction_report["disease_risk_score"]
                    risk_status = "High Risk" if risk_score > 0.7 else "Moderate Risk" if risk_score > 0.4 else "Low Risk"

                    st.json({
                        "Patient_ID": prediction_report["accession_id"],
                        "Disease_Risk": prediction_report["disease_risk_percentage"],
                        "Status": risk_status,
                        "Predicted_Disease_Type": prediction_report["predicted_disease_type"].replace("_", " "),
                        "Early_Biomarkers": prediction_report["early_biomarkers"]
                    })

                with tab2:
                    st.subheader("Microbiome & Physical Symptoms")
                    st.info("Data retrieved from ENA and Metadata CSV.")
                    st.write("**Physical Symptoms:**", prediction_report.get("outer_body_accountability", "None"))
                    st.write("**Raw Multi-Omics Feature Matrix:**")
                    st.json(combined_features)
    else:
        st.warning("Please enter a valid ID.")
