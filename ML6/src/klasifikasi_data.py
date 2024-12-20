import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
import numpy as np

# Augment data function
def augment_data(X, y):
    noise = np.random.normal(0, 0.01, X.shape)
    X_augmented = X + noise
    y_augmented = y
    return np.vstack((X, X_augmented)), np.hstack((y, y_augmented))

# Klasifikasi Data Page
def app():
    st.markdown("<div class='content-card'>", unsafe_allow_html=True)
    st.header("Klasifikasi")
    if st.session_state.data is not None:
        # Pilih kolom fitur
        feature_columns = st.multiselect(
            "Pilih kolom fitur:", 
            options=st.session_state.data.columns,
            default=[col for col in st.session_state.data.columns if col != st.session_state.data.columns[-1]]
        )
        target_column = st.selectbox("Pilih kolom target:", st.session_state.data.columns)

        if not feature_columns:
            st.warning("Harap pilih setidaknya satu kolom fitur.")
            return

        algorithm = st.selectbox("Pilih algoritma:", ["Logistic Regression", "Decision Tree", "Random Forest", "SVM", "Neural Network"])

        if st.button("Mulai Klasifikasi"):
            try:
                X = st.session_state.data[feature_columns]
                y = st.session_state.data[target_column]

                # Konversi kolom kategori ke numerik
                X = pd.get_dummies(X)
                if y.dtype == 'object':
                    y = y.astype('category').cat.codes

                # Handling missing values
                imputer = SimpleImputer(strategy="mean")
                X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
                y = y.dropna()
                X = X.loc[y.index]

                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Augment data
                X_train, y_train = augment_data(X_train.to_numpy(), y_train.to_numpy())

                # Model selection
                if algorithm == "Logistic Regression":
                    model = LogisticRegression(max_iter=1000)
                elif algorithm == "Decision Tree":
                    model = DecisionTreeClassifier()
                elif algorithm == "Random Forest":
                    model = RandomForestClassifier()
                elif algorithm == "SVM":
                    model = SVC()
                elif algorithm == "Neural Network":
                    model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)

                # Train and evaluate model
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                accuracy = accuracy_score(y_test, predictions)

                st.success(f"Akurasi Model ({algorithm}): {accuracy:.2f}")
                st.write("Hasil Prediksi:")
                st.write(predictions)
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
    else:
        st.warning("Silakan unggah data terlebih dahulu di menu Upload Data.")
    st.markdown("</div>", unsafe_allow_html=True)
