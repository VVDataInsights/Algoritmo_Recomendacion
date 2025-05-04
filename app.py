import streamlit as st
import json
import pandas as pd
import os
import joblib

st.set_page_config(page_title="Recomendador Corona", layout="centered")

st.title("Recomendador de Productos B2C y B2B - Corona")

# --- Cargar estructuras JSON ---
def cargar_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


try:
    b2c_dataset = pd.DataFrame(joblib.load("modelos/b2c_dataset.pkl"))
    b2c_user_features = pd.DataFrame(joblib.load("modelos/b2c_user_features.pkl"))
    b2c_item_features = pd.DataFrame(joblib.load("modelos/b2c_item_features.pkl"))
    b2c_model_lfm = pd.DataFrame(joblib.load("modelos/b2c_model_lfm.pkl"))
    b2c_model_xgb = pd.DataFrame(joblib.load("modelos/b2c_model_xgb.pkl"))

    b2b_dataset = pd.DataFrame(joblib.load("modelos/b2b_dataset.pkl"))
    b2b_user_features = pd.DataFrame(joblib.load("modelos/b2b_user_features.pkl"))
    b2b_item_features = pd.DataFrame(joblib.load("modelos/b2b_item_features.pkl"))
    b2b_model_lfm = pd.DataFrame(joblib.load("modelos/b2b_model_lfm.pkl"))
    b2b_model_xgb = pd.DataFrame(joblib.load("modelos/b2b_model_xgb.pkl"))
    b2b_encoder = pd.DataFrame(joblib.load("modelos/b2b_encoder.pkl"))

    st.success("Modelos y estructuras pkl cargados correctamente.")
except Exception as e:
    st.error(f"Error cargando estructuras pkl: {e}")

# --- Interfaz ---
cliente_id = st.text_input("Ingrese el ID del cliente o empresa:")
n_top = st.slider("Número de recomendaciones", 1, 20, 5)

if st.button("Generar Recomendaciones"):
    if not cliente_id:
        st.warning("Por favor ingrese un ID válido.")
    else:
        try:
            # Verificar si es B2C o B2B
            if cliente_id in b2c_dataset['id_cliente'].values:
                df = b2c_model_lfm.copy()
                st.subheader("Recomendaciones B2C")
            elif cliente_id in b2b_dataset['id_b2b'].values:
                df = b2b_model_lfm.copy()
                st.subheader("Recomendaciones B2B")
            else:
                st.warning("ID no encontrado en B2C ni en B2B.")
                st.stop()

            # Filtrar recomendaciones para el cliente
            recomendaciones = df[df['id_cliente'] == cliente_id].sort_values("score", ascending=False).head(n_top)

            st.dataframe(recomendaciones)
        except Exception as e:
            st.error(f"Error generando recomendaciones: {e}")
