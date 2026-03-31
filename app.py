import streamlit as st
import pandas as pd
import numpy as np
import requests
from Bio import Entrez
from fpdf import FPDF
from multiomics_pipeline import run_pipeline_for_id

# --- NEW: Database Search Functions ---
@st.cache_data
def search_ncbi(accession_id):
    """Fetch gene/nucleotide summary from NCBI Entrez."""
    Entrez.email = "your_email@example.com"  # Required by NCBI
    try:
        handle = Entrez.esearch(db="nucleotide", term=accession_id)
        record = Entrez.read(handle)
        if record["IdList"]:
            summary_handle = Entrez.esummary(db="nucleotide", id=record["IdList"][0])
            summary = Entrez.read(summary_handle)
            return summary[0]
        return None
    except Exception:
        return None

@st.cache_data
def search_uniprot(accession_id):
    """Fetch protein data from UniProt REST API."""
    url = f"https://rest.uniprot.org/uniprotkb/search?query={accession_id}&format=json"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return data['results'][0] if data['results'] else None
        return None
    except Exception:
        return None

def generate_pdf_report(report_data):
    pdf = FPDF()
    pdf.add_page()

    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Multi-Omics Diagnostic Report", ln=True, align='C')
    pdf.ln(10)

    # Body Content
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Patient ID: {report_data['accession_id']}", ln=True)
    pdf.cell(200, 10, txt=f"Risk Percentage: {report_data['disease_risk_percentage']}", ln=True)
    pdf.cell(200, 10, txt=f"Predicted State: {report_data['predicted_disease_type']}", ln=True)

    pdf.ln(5)
    pdf.cell(200, 10, txt="Biomarkers Detected:", ln=True)
    for bio in report_data['early_biomarkers']:
        pdf.cell(200, 10, txt=f"- {bio}", ln=True)

    return pdf.output(dest='S').encode('latin-1')

st.set_page_config(page_title="Multi-Omics Search Portal", layout="wide")

st.title("🧬 Multi-Omics Disease Detection Portal")
st.markdown("Enter an ENA Accession ID to fetch data and run the diagnostic pipeline.")

import matplotlib.pyplot as plt
import shap

# 1. User Input
accession_id = st.text_input("Enter Accession Number (e.g., ERR1145155)", "ERR1145155")

if st.button("Search & Analyze"):
    if accession_id:
        with st.spinner(f"Searching databases for {accession_id}..."):
            st.info("Scanning distributed genomic chunks...")
            # 2. Call the logic
            ret = run_pipeline_for_id(accession_id)
            if ret[0] is None:
                prediction_report = None
            else:
                prediction_report, combined_features, shap_values, model_inputs = ret

            if not prediction_report:
                st.error("Failed to run pipeline. Check the backend logs or ensure the accession ID is correct.")
            else:
                st.sidebar.success(f"Running in {prediction_report.get('mode', 'Unknown')} Mode")
                st.sidebar.info(f"Data Analyzed: {prediction_report.get('total_size_mb', 0):.1f} MB")

                tab1, tab2 = st.tabs(["Diagnostic Report", "Raw Data Summary"])

                with tab1:
                    st.subheader("Final Diagnostic Prediction")

                    risk_score = prediction_report["disease_risk_score"]
                    risk_status = "High Risk" if risk_score > 0.7 else "Moderate Risk" if risk_score > 0.4 else "Low Risk"

                    colA, colB = st.columns(2)
                    with colA:
                         st.metric("Hybrid Risk Score", prediction_report["disease_risk_percentage"])
                         st.metric("Predicted State", prediction_report["predicted_disease_type"].replace("_", " "))
                    with colB:
                         st.metric("TensorFlow Cancer Risk", f"{round(prediction_report['experimental_cancer_risk'] * 100, 2)}%")
                         st.metric("TF Microbiome Imbalance", f"{round(prediction_report['experimental_microbiome_risk'] * 100, 2)}%")
                         st.metric("TF Metabolic Risk", f"{round(prediction_report['experimental_metabolic_risk'] * 100, 2)}%")

                    st.markdown("### Interpretability (SHAP)")
                    st.write("Feature impacts on Cancer Risk (TensorFlow Model):")

                    # SHAP GradientExplainer returns a list of arrays: one for each output class.
                    # Index 0 corresponds to the first output (Cancer Risk).
                    # Each element in the output list is a list of arrays (one for each input branch).
                    try:
                        shap_cancer = shap_values[0]

                        # Concatenate SHAP values across the 3 input branches to match the 11 feature names
                        shap_concat = np.concatenate([shap_cancer[0][0], shap_cancer[1][0], shap_cancer[2][0]])
                        inputs_concat = np.concatenate([model_inputs[0][0], model_inputs[1][0], model_inputs[2][0]])

                        feature_names = [
                            "Mean Length", "Prop 145bp", "CNN Score", "CHIP Noise",
                            "F. nucleatum", "B. fragilis", "E. coli",
                            "Fatigue", "Weight Loss", "Pain", "Nausea"
                        ]

                        fig, ax = plt.subplots(figsize=(6,4))
                        # shap.summary_plot with plot_type='bar' expects an array of shape (N, M),
                        # so we wrap the single instance in a list to pretend it's a batch of 1
                        shap.summary_plot(np.array([shap_concat]), features=np.array([inputs_concat]), feature_names=feature_names, plot_type='bar', show=False)
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"SHAP plotting failed: {e}")

                    st.markdown("### Early Biomarkers")
                    for bio in prediction_report["early_biomarkers"]:
                         st.info(bio)

                    pdf_bytes = generate_pdf_report(prediction_report)
                    st.download_button(
                        label="📄 Download PDF Report",
                        data=pdf_bytes,
                        file_name=f"report_{accession_id}.pdf",
                        mime="application/pdf"
                    )

                    # Create a link to the ENA browser for the specific sample
                    ena_link = f"https://www.ebi.ac.uk/ena/browser/view/{accession_id}"

                    # Display the source in the UI
                    st.subheader("🔗 Source Evidence")
                    st.markdown(f"""
Your sample was cross-referenced with the **European Nucleotide Archive (ENA)**.
* **Accession ID:** `{accession_id}`
* **Study Reference:** [Project PRJEB6070 - Colorectal Cancer Cohort](https://www.ebi.ac.uk/ena/browser/view/PRJEB6070)
* **Raw Data Link:** [View full sequencing metadata on ENA]({ena_link})
                    """)

                with tab2:
                    st.subheader("Microbiome & Physical Symptoms")
                    st.info("Data retrieved from ENA and Metadata CSV.")
                    st.write("**Physical Symptoms:**", prediction_report.get("outer_body_accountability", "None"))
                    st.write("**Raw Multi-Omics Feature Matrix:**")
                    st.json(combined_features)

            st.subheader("🌐 Global Database Cross-Reference")
            ncbi_col, uni_col = st.columns(2)

            with ncbi_col:
                st.markdown("### NCBI Nucleotide")
                ncbi_data = search_ncbi(accession_id)
                if ncbi_data:
                    st.success(f"Found: {ncbi_data.get('Title', 'No Title')}")
                    st.write(f"**TaxID:** {ncbi_data.get('TaxId')}")
                else:
                    st.warning("No matching record in NCBI.")

            with uni_col:
                st.markdown("### UniProt (Proteins)")
                uni_data = search_uniprot(accession_id)
                if uni_data:
                    try:
                        protein_name = uni_data['proteinDescription']['recommendedName']['fullName']['value']
                        st.success(f"Found: {protein_name}")
                        st.write(f"**Organism:** {uni_data['organism']['scientificName']}")
                    except KeyError:
                         st.success("Found record, but protein description format differs.")
                else:
                    st.warning("No matching record in UniProt.")
    else:
        st.warning("Please enter a valid ID.")
