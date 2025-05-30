import pandas as pd
import numpy as np
from joblib import load
import xgboost as xgb
from lightfm.data import Dataset


# Cargar modelos y datos auxiliares
model_lfm = load("./modelos/lightfm_model_b2b.pkl")
model_xgb = load("./modelos/xgb_model_b2b.pkl")

encoders = load("./modelos/encoders_b2b.pkl")
dataset = load("./modelos/lightfm_dataset_b2b.pkl")
df_precios_productos = pd.read_csv("./precios_productos_promedio.csv")
b2b = pd.read_csv("./b2b_nuevo.csv")

# Inicialización
dataset = Dataset()
dataset.fit(
    users=b2b['id_b2b'].unique(),
    items=b2b['producto'].unique()
)

dataset.fit_partial(
    users=b2b['id_b2b'],
    items=b2b['producto'],
    user_features=np.unique([
        *("municipio:" + b2b['municipio']),
        *("zona:" + b2b['zona']),
        *("unidades:" + b2b['Total de unidades'].astype(str)),
        *("edificaciones:" + b2b['Total de edificaciones en obra'].astype(str))
    ]),
    item_features=np.unique([
        *("cat_macro:" + b2b['categoria_b2b_macro']),
        *("cat:" + b2b['categoria_b2b']),
        *("subcat:" + b2b['subcategoria_b2b'])
    ])
)

# Interacciones
(interactions, _) = dataset.build_interactions(
    ((row['id_b2b'], row['producto'], row['valor_total']) for _, row in b2b.iterrows())
)

# User features
user_features = dataset.build_user_features(
    ((row['id_b2b'], [
        f"municipio:{row['municipio']}",
        f"zona:{row['zona']}",
        f"unidades:{row['Total de unidades']}",
        f"edificaciones:{row['Total de edificaciones en obra']}"
    ]) for _, row in b2b.iterrows())
)

# Item features
item_features = dataset.build_item_features(
    ((row['producto'], [
        f"cat_macro:{row['categoria_b2b_macro']}",
        f"cat:{row['categoria_b2b']}",
        f"subcat:{row['subcategoria_b2b']}"
    ]) for _, row in b2b.iterrows())
)

def recomendar_hibrido_b2b(cliente_id, top_n=10, alpha=0.5):
    if cliente_id not in dataset.mapping()[0]:
        return f"Cliente {cliente_id} no está en el dataset de LightFM."

    productos_comprados = set(b2b[b2b['id_b2b'] == cliente_id]['producto'].unique())
    productos_totales = b2b['producto'].unique()
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

    df_lfm = pd.DataFrame({
        'producto': [p for p in productos_candidatos if p in dataset.mapping()[2]],
        'score_lfm': scores_lfm
    })

    # --- XGBoost ---
    cliente_info = b2b[b2b['id_b2b'] == cliente_id].iloc[0]
    nuevas_filas = []

    for producto in df_lfm['producto']:
        fila = cliente_info.copy()
        fila['producto'] = producto
        nuevas_filas.append(fila)

    pred_df = pd.DataFrame(nuevas_filas)

    for col in encoders:
        pred_df[col] = encoders[col].transform(pred_df[col])

    X_pred = pred_df[model_xgb.get_booster().feature_names]
    scores_xgb = model_xgb.predict_proba(X_pred)[:, 1]

    # Combinar scores
    # Normalizar LFM si hay más de un score y no son todos iguales
    if df_lfm['score_lfm'].nunique() > 1:
        min_lfm = df_lfm['score_lfm'].min()
        max_lfm = df_lfm['score_lfm'].max()
        df_lfm['score_lfm_norm'] = (df_lfm['score_lfm'] - min_lfm) / (max_lfm - min_lfm)
    else:
        df_lfm['score_lfm_norm'] = 0.5

    df_lfm['score_xgb'] = scores_xgb
    df_lfm['score_hibrido'] = alpha * df_lfm['score_lfm_norm'] + (1 - alpha) * df_lfm['score_xgb']

    # --- Agregar información de precio y alineación estratégica ---
    # Unir con el dataframe de precios promedio
    df_lfm = df_lfm.merge(df_precios_productos, on='producto', how='left')

    # Calcular el valor esperado
    df_lfm['valor_esperado'] = df_lfm['score_hibrido'] * df_lfm['precio_promedio']

    # Necesitamos la alineación por producto. Podemos obtenerla del df_original agrupando por producto
    alineacion_productos = b2b.drop_duplicates('producto')[['producto', 'alineación con portafolio estratégico b2b']]
    df_lfm = df_lfm.merge(alineacion_productos, on='producto', how='left')

    return df_lfm.sort_values('score_hibrido', ascending=False).head(top_n)
