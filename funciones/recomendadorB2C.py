import pandas as pd
import numpy as np
from joblib import load
from lightfm.data import Dataset

# --- Cargar modelos y datos ---
model_lfm = load("./modelos/lightfm_model_b2c.pkl")
modelo_xgb = load("./modelos/xgb_model_b2c.pkl")
dataset = load("./modelos/lightfm_dataset_b2c.pkl")
user_features = load("./modelos/b2c_user_features.pkl")  # Para LightFM
item_features = load("./modelos/b2c_item_features.pkl")

xgb_user_feats = load("./modelos/b2c_user_features_df.pkl")  # Para XGBoost
xgb_item_feats = load("./modelos/b2c_item_features_df.pkl")

b2c = pd.read_csv("./b2c_nuevo.csv")
precios = pd.read_csv("./df_precios.csv")

# --- Función principal ---
def recomendar_hibrido(cliente_id, top_n=10, alpha=0.5):
    if cliente_id not in dataset.mapping()[0]:
        return f"Cliente {cliente_id} no está en el dataset de LightFM."

    # Filtrar productos que no haya comprado
    productos_comprados = set(b2c[b2c['id'] == cliente_id]['producto'].unique())
    productos_totales = b2c['producto'].unique()
    productos_candidatos = list(set(productos_totales) - productos_comprados)

    if not productos_candidatos:
        return f"Cliente {cliente_id} ya ha comprado todos los productos."

    # --- LightFM ---
    usuario_interno = dataset.mapping()[0][cliente_id]
    productos_idx = [dataset.mapping()[2][p] for p in productos_candidatos if p in dataset.mapping()[2]]
    scores_lfm = model_lfm.predict(
        user_ids=np.repeat(usuario_interno, len(productos_idx)),
        item_ids=productos_idx,
        user_features=user_features,
        item_features=item_features
    )

    lfm_scores = pd.DataFrame({
        'producto': [p for p in productos_candidatos if p in dataset.mapping()[2]],
        'score_lfm': scores_lfm
    })

    # --- XGBoost ---
    df_pred = pd.DataFrame({'id': cliente_id, 'producto': lfm_scores['producto']})
    df_pred = df_pred.merge(xgb_user_feats, on='id', how='left')
    df_pred = df_pred.merge(xgb_item_feats, on='producto', how='left')
    df_pred.fillna("desconocido", inplace=True)

    # Reducción de cardinalidad en 'color'
    if 'color' in df_pred.columns and df_pred['color'].nunique() > 20:
        top_colores = xgb_item_feats['color'].value_counts().nlargest(20).index
        df_pred['color'] = df_pred['color'].where(df_pred['color'].isin(top_colores), 'otros')

    for col in ['cluster', 'categoria_macro', 'subcategoria', 'color']:
        if col in df_pred.columns:
            df_pred[col] = df_pred[col].astype('category')

    X_pred = df_pred.drop(columns=['id', 'producto'])
    scores_xgb = modelo_xgb.predict_proba(X_pred)[:, 1]

    # --- Combinar scores ---
    if lfm_scores['score_lfm'].nunique() > 1:
        lfm_scores['score_lfm_norm'] = (lfm_scores['score_lfm'] - lfm_scores['score_lfm'].min()) / (lfm_scores['score_lfm'].max() - lfm_scores['score_lfm'].min())
    else:
        lfm_scores['score_lfm_norm'] = 0.5

    lfm_scores['score_xgb'] = scores_xgb
    lfm_scores['score_hibrido'] = alpha * lfm_scores['score_lfm_norm'] + (1 - alpha) * lfm_scores['score_xgb']

    # --- Alineación estratégica y precio ---
    alineacion_dict = b2c.drop_duplicates('producto').set_index('producto')['alineación con portafolio estratégico'].to_dict()
    lfm_scores['alineación estratégica'] = lfm_scores['producto'].map(alineacion_dict)

    precios_dict = precios.set_index('producto')['precio'].to_dict()
    lfm_scores['precio'] = lfm_scores['producto'].map(precios_dict)

    # Filtrar productos sin precio
    lfm_scores = lfm_scores[lfm_scores['precio'].notna()]

    # --- Valor esperado ---
    lfm_scores['valor_esperado'] = lfm_scores['score_hibrido'] * lfm_scores['precio']

    return lfm_scores.sort_values('score_hibrido', ascending=False).head(top_n)[[
        'producto', 'score_lfm_norm', 'score_xgb', 'score_hibrido',
        'alineación estratégica', 'precio', 'valor_esperado'
    ]]
