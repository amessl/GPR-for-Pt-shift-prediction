import streamlit as st
import numpy as np
from tempfile import NamedTemporaryFile
from src.inference.infer_single import single_inference_app

st.title("Pt Chemical Shift Prediction \n Web App")

st.write("This application allows you to predict the chemical shift of Pt complexes based on a Gaussian Process Regressor and provides prediction uncertainties to "
         "narrow down the chemical shift range in an experimental measurement. Select the molecular representation you would like to use and also the confidence interval.")

st.image('TOC_1.png')

representation = st.selectbox("Select Molecular Representation",
                              options=["ChEAP", "GAPE", "SOAP"])

confidence_interval = st.selectbox("Select Confidence Interval",
                                   options=[f"68 %", "95 %", "99.7 %"])

file = st.file_uploader("Drag and drop XYZ file", type=["xyz"])

if st.button("Run Prediction"):
    if file is None:
        st.warning("Crazy idea: upload your input xyz-file before running inference.")
    else:
        with NamedTemporaryFile(delete=False, suffix='.xyz') as temp_file:
            temp_file.write(file.getvalue())
            filename = temp_file.name
        with st.spinner("Running inference...."):
            prediction, uncertainty = single_inference_app(representation, filename, confidence_interval)

        st.header("Results")
        st.divider()
        st.write(f"Prediction:    {np.round(prediction[0])} ppm")
        st.divider()
        st.write(f"Uncertainty:    +/- {np.round(uncertainty[0])} ppm")
        st.divider()