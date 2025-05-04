from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# --- Cargar modelos B2C ---
b2c_model_xgb = joblib.load('modelos/b2c_model_xgb.pkl')
b2c_model_lfm = joblib.load('modelos/b2c_model_lfm.pkl')
b2c_dataset = joblib.load('modelos/b2c_dataset.pkl')
b2c_user_features = joblib.load('modelos/b2c_user_features.pkl')
b2c_item_features = joblib.load('modelos/b2c_item_features.pkl')
b2c_encoders = joblib.load('modelos/b2c_encoders.pkl')

# --- Cargar modelos B2B ---
b2b_model_xgb = joblib.load('modelos/b2b_model_xgb.pkl')
b2b_model_lfm = joblib.load('modelos/b2b_model_lfm.pkl')
b2b_dataset = joblib.load('modelos/b2b_dataset.pkl')
b2b_user_features = joblib.load('modelos/b2b_user_features.pkl')
b2b_item_features = joblib.load('modelos/b2b_item_features.pkl')
b2b_encoders = joblib.load('modelos/b2b_encoders.pkl')

# Dummy data para simular recomendaciones (esto se reemplaza con las funciones reales)
def recomendar_dummy(cliente_id, tipo="b2c"):
    productos = [f"Producto_{i}" for i in range(5)]
    puntajes = np.round(np.random.rand(5), 2)
    df = pd.DataFrame({'producto': productos, 'puntaje': puntajes})
    return df.sort_values('puntaje', ascending=False)

@app.route('/', methods=['GET', 'POST'])
def home():
    recomendaciones = None
    if request.method == 'POST':
        cliente_id = request.form['cliente_id']
        tipo = request.form['tipo']
        recomendaciones = recomendar_dummy(cliente_id, tipo)
    return render_template('index.html', recomendaciones=recomendaciones)

if __name__ == '__main__':
    app.run(debug=True)
