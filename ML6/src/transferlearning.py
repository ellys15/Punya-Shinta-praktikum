import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pytorch_tabnet.tab_model import TabNetClassifier
import numpy as np
import torch

def app():
    if 'data' not in st.session_state:
        st.warning("Unggah data terlebih dahulu melalui halaman Upload Data.")
        return

    data = st.session_state['data']

    # Pilih fitur dan target
    all_columns = data.columns
    fitur = st.multiselect("Pilih kolom fitur:", all_columns)
    target = st.selectbox("Pilih kolom target:", all_columns)

    if fitur and target:
        # Memisahkan fitur dan target
        X = pd.get_dummies(data[fitur]).values  # Konversi fitur ke numerik
        y = pd.Categorical(data[target]).codes  # Konversi target ke numerik

        # Split data menjadi train dan test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Pastikan data berupa numpy array
        X_train = np.array(X_train, dtype=np.float32)
        X_test = np.array(X_test, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.int64)
        y_test = np.array(y_test, dtype=np.int64)

        # Cek bentuk data
        st.write("Shape X_train:", X_train.shape)
        st.write("Shape y_train:", y_train.shape)

        if st.button("Jalankan TabNet"):
            st.text("Melatih model TabNet...")

            # Inisialisasi model
            model = TabNetClassifier(
                n_d=8, n_a=8, n_steps=3,
                gamma=1.3, lambda_sparse=1e-4,
                optimizer_fn=torch.optim.Adam,
                optimizer_params=dict(lr=2e-2),
                verbose=0
            )

            # Melatih model
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                eval_metric=['accuracy'],
                max_epochs=100,
                patience=10,
                batch_size=32,
                virtual_batch_size=16
            )

            # Evaluasi model
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            # Tampilkan hasil
            st.success("Pelatihan selesai!")
            st.write(f"Akurasi Model: *{accuracy:.2f}*")
            st.dataframe(pd.DataFrame({"Actual": y_test, "Predicted": y_pred}))

            # Simpan model
            model.save_model("tabnet_model.zip")
            st.write("Model disimpan sebagai 'tabnet_model.zip'.")