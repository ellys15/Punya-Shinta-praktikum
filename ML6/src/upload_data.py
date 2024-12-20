import streamlit as st
import pandas as pd

# Upload Data Page
def app():
    st.markdown("<div class='content-card'>", unsafe_allow_html=True)
    st.header("Upload Data")
    uploaded_file = st.file_uploader("Unggah file CSV Anda di sini:", type=["csv"])
    if uploaded_file is not None:
        st.session_state.data = pd.read_csv(uploaded_file)
        st.write("Pratinjau Data:")
        st.dataframe(st.session_state.data)
    st.markdown("</div>", unsafe_allow_html=True)
