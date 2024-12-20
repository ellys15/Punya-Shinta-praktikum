import streamlit as st
from upload_data import app as upload_data_app
from analisis_data import app as analisis_data_app
from klasifikasi_data import app as klasifikasi_data_app
from transferlearning import app as transfer_learning_app

# Custom CSS for styling
st.markdown("""
    <style>
    .main-header {
        font-size: 30px;
        font-weight: bold;
        text-align: center;
        margin-top: -20px;
        color: #4CAF50;
    }
    .sub-header {
        font-size: 18px;
        text-align: center;
        color: #555;
        margin-bottom: 20px;
    }
    .content-card {
        background-color: #f9f9f9;
        border-radius: 8px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Main Header
st.markdown("<div class='main-header'>Aplikasi Klasifikasi Tabular</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Eksplorasi dan Analisis Data Tabular dengan Mudah</div>", unsafe_allow_html=True)

# Sidebar for navigation
with st.sidebar:
    st.title("Navigasi")
    selected_page = st.radio(
        "Pilih halaman:",
        ["Beranda", "Upload Data", "Analisis Data", "Klasifikasi", "Transfer Learning"]
    )

# Render the selected page
if selected_page == "Beranda":
    st.markdown("<div class='content-card'>", unsafe_allow_html=True)
    st.header("Beranda")
    st.write("""
    Selamat datang di aplikasi *Klasifikasi Tabular*.
    Gunakan aplikasi ini untuk melakukan analisis dan klasifikasi data tabular secara interaktif.
    Pilih halaman dari menu navigasi untuk memulai.
    """)
    st.markdown("</div>", unsafe_allow_html=True)
elif selected_page == "Upload Data":
    upload_data_app()
elif selected_page == "Analisis Data":
    analisis_data_app()
elif selected_page == "Klasifikasi":
    klasifikasi_data_app()
elif selected_page == "Transfer Learning":
    transfer_learning_app()
