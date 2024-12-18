import streamlit as st
import pandas as pd
import pickle

# Load the trained model
pickle_in = open('classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)

# Function to preprocess input data
def preprocess_input(jk, umur, berat, tinggi, lila):
    """
    Preprocess user input to match the dataset structure with one-hot encoded gender (JK_L and JK_P).
    """
    data = pd.DataFrame({
        'Berat': [berat],
        'Tinggi': [tinggi],
        'LiLA': [lila],
        'Umur': [umur],  # Age in months
        'JK_L': [1 if jk == "L" else 0],  # JK_L is 1 if gender is Laki-Laki
        'JK_P': [1 if jk == "P" else 0],  # JK_P is 1 if gender is Perempuan
    })
    return data

# Streamlit app
st.set_page_config(
    page_title="Prediksi Status Gizi Balita",
    page_icon="👶",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Custom CSS for UI Styling
st.markdown("""
    <style>
    body {
        background-color: #f9f9f9;
        font-family: "Arial", sans-serif;
    }
    .title {
        font-size: 36px;
        font-weight: bold;
        color: #4caf50;
        text-align: center;
        margin-top: -20px;
        margin-bottom: 20px;
    }
    .sidebar .sidebar-content {
        background: #fffbe6;
    }
    .main-button {
        background-color: #4caf50;
        color: white;
        font-size: 16px;
        padding: 10px 15px;
        border-radius: 8px;
        border: none;
    }
    .main-button:hover {
        background-color: #45a049;
    }
    </style>
""", unsafe_allow_html=True)

# Page Title
st.markdown('<div class="title">Prediksi Status Gizi Balita</div>', unsafe_allow_html=True)

# App Subtitle
st.subheader("Streamlit Gizi Balita Classifier ML SVM")

# Input fields with better formatting
st.markdown("### Masukkan Data Balita:")
berat = st.number_input("Berat Badan (kg):", min_value=0.0, value=0.0, step=0.1)
tinggi = st.number_input("Tinggi Badan (cm):", min_value=0.0, value=0.0, step=0.1)
lila = st.number_input("Lingkar Lengan Atas (cm):", min_value=0.0, value=0.0, step=0.1)
umur = st.number_input("Umur (0-60 bulan):", min_value=0.0, value=0.0, step=0.1)  # Numeric input for age in months
jk = st.selectbox("Jenis Kelamin:", ["L", "P"], help="Pilih jenis kelamin balita (L = Laki-Laki, P = Perempuan)")

# Prediction button with custom style
if st.button("Prediksi", help="Klik tombol ini untuk melihat hasil prediksi status gizi."):
    try:
        # Preprocess input
        input_data = preprocess_input(jk, umur, berat, tinggi, lila)

        # Ensure all required features are present and numeric
        if input_data.isnull().values.any() or (input_data < 0).any(axis=None):
            raise ValueError("Pastikan semua input diisi dengan nilai valid (positif).")

        # Predict the class (status gizi)
        prediction = classifier.predict(input_data)[0]

        # Convert prediction to human-readable label
        label_mapping = {
            0: '🥄 Gizi Kurang',
            1: '🥦 Normal',
            2: '🍔 Beresiko Gizi Lebih',
            3: '🍩 Gizi Lebih',
            4: '🎂 Obesitas'
        }

        # Display result
        st.success(f"Prediksi Status Gizi: {label_mapping[prediction]}")
        st.balloons()

    except ValueError as e:
        st.error(f"Error: {e}. Periksa input Anda.")
    except Exception as e:
        st.error(f"Terjadi kesalahan tak terduga: {e}")
