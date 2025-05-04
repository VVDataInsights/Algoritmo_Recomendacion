import streamlit as st
import joblib
import json
import pandas as pd
import numpy as np
import os

# --- Título de la app ---
st.title("Recomendador de Productos B2C y B2B - Corona")

# --- Cargar modelos y estructuras ---
try:
    model_xgb = joblib.load("modelos/b2c_model_xgb.pkl")
    model_lfm = joblib.load("modelos/b2c_model_lfm.pkl")
except Exception as e:
    st.error(f"Error cargando los modelos: {e}")

# --- Cargar archivos JSON ---
def cargar_json(path):
    with open(path, "r") as f:
        return json.load(f)

try:
    encoders = cargar_json("modelos/b2c_encoders.json")
    with open("modelos/b2c_user_features.json") as f:
        user_features = json.load(f)
    with open("modelos/b2c_item_features.json") as f:
        item_features = json.load(f)
    with open("modelos/b2c_dataset.json") as f:
        dataset = json.load(f)
except Exception as e:
    st.error(f"Error cargando estructuras JSON: {e}")

# --- Input del usuario ---
cliente_id = st.text_input("Ingrese el ID del cliente o empresa:", "")
top_n = st.slider("Número de recomendaciones", 1, 20, 5)

# --- Simulación de recomendación ---
def recomendar(cliente_id, top_n):
    # Aquí debes insertar la lógica real
    return pd.DataFrame({
        "Producto": [f"Producto_{i}" for i in range(1, top_n + 1)],
        "Score XGBoost": np.round(np.random.rand(top_n), 4),
        "Score LightFM": np.round(np.random.rand(top_n), 4),
        "Score Híbrido": np.round(np.random.rand(top_n), 4)
    })

# --- Al hacer clic ---
if st.button("Generar Recomendaciones"):
    if not cliente_id:
        st.warning("Por favor ingrese un ID.")
    else:
        st.success(f"Recomendaciones para el cliente: {cliente_id}")
        recomendaciones = recomendar(cliente_id, top_n)
        st.dataframe(recomendaciones)
