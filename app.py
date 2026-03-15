import streamlit as st
import pandas as pd
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

# 1. User Input
accession_id = st.text_input("Enter Accession Number (e.g., ERR1145155)", "ERR1145155")

if st.button("Search & Analyze"):
    if accession_id:
        with st.spinner(f"Searching databases for {accession_id}..."):
            st.info("Scanning distributed genomic chunks...")
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

                    pdf_bytes = generate_pdf_report(prediction_report)
                    st.download_button(
                        label="📄 Download PDF Report",
                        data=pdf_bytes,
                        file_name=f"report_{accession_id}.pdf",
                        mime="application/pdf"
                    )

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
