import pandas as pd
import numpy as np
from joblib import load
import xgboost as xgb

# Cargar modelo XGBoost desde JSON (no joblib)
booster = xgb.Booster()
booster.load_model("./modelos/modelo_xgb.json")

# Cargar modelos y datos auxiliares
model_lfm = load("./modelos/lightfm_model_b2b.pkl")
encoders = load("./modelos/encoders_b2b.pkl")
dataset = load("./modelos/lightfm_dataset_b2b.pkl")
df_precios_productos = pd.read_csv("./precios_productos_promedio.csv")
b2b = pd.read_csv("./b2b_nuevo.csv")

def recomendar_hibrido_b2b(cliente_id, top_n=10, alpha=0.5):
    if cliente_id not in dataset.mapping()[0]:
        return f"Cliente {cliente_id} no está en el dataset de LightFM."

    productos_comprados = set(b2b[b2b['id_b2b'] == cliente_id]['producto'].unique())
    productos_totales = b2b['producto'].unique()
    productos_candidatos = list(set(productos_totales) - productos_comprados)

    if not productos_candidatos:
        return f"Cliente {cliente_id} ya ha comprado todos los productos."

    usuario_interno = dataset.mapping()[0][cliente_id]
    productos_idx = [dataset.mapping()[2][p] for p in productos_candidatos if p in dataset.mapping()[2]]
    productos_validos = [p for p in productos_candidatos if p in dataset.mapping()[2]]

    scores_lfm = model_lfm.predict(
        user_ids=np.repeat(usuario_interno, len(productos_idx)),
        item_ids=productos_idx
    )

    df_lfm = pd.DataFrame({
        'producto': productos_validos,
        'score_lfm': scores_lfm
    })

    cliente_info = b2b[b2b['id_b2b'] == cliente_id].iloc[0]
    nuevas_filas = []

    for producto in df_lfm['producto']:
        fila = cliente_info.copy()
        fila['producto'] = producto
        nuevas_filas.append(fila)

    pred_df = pd.DataFrame(nuevas_filas)

    for col in encoders:
        pred_df[col] = encoders[col].transform(pred_df[col])

    # Asegúrate de usar solo las features que espera el booster
    booster_features = booster.feature_names
    X_pred = xgb.DMatrix(pred_df[booster_features])
    scores_xgb = booster.predict(X_pred)

    df_lfm['score_xgb'] = scores_xgb
    df_lfm['score_hibrido'] = alpha * df_lfm['score_lfm'] + (1 - alpha) * df_lfm['score_xgb']

    df_lfm = df_lfm.merge(df_precios_productos, on='producto', how='left')
    df_lfm['valor_esperado'] = df_lfm['score_hibrido'] * df_lfm['precio_promedio']

    alineacion = b2b[['producto', 'alineación con portafolio estratégico b2b']].drop_duplicates()
    df_lfm = df_lfm.merge(alineacion, on='producto', how='left')

    return df_lfm.sort_values('score_hibrido', ascending=False).head(top_n)
