import streamlit as st

# Analisis Data Page
def app():
    st.markdown("<div class='content-card'>", unsafe_allow_html=True)
    st.header("Analisis Data")
    if st.session_state.data is not None:
        st.write("Deskripsi Statistik:")
        st.write(st.session_state.data.describe())
        st.write("Jumlah Nilai Kosong (NaN) per Kolom:")
        st.write(st.session_state.data.isna().sum())
    else:
        st.warning("Silakan unggah data terlebih dahulu di menu Upload Data.")
    st.markdown("</div>", unsafe_allow_html=True)
