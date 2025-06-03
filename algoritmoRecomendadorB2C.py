#Librerias necesarias
import pandas as pd
import numpy as np
import os
from joblib import dump


#Librerias de las recomendaciones
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.cross_validation import random_train_test_split

# Cargar el dataset
pd2 = pd.read_csv('./b2c_nuevo.csv' , sep=",", encoding="utf-8")
b2c = pd2.copy()
b2c = b2c.dropna(subset=['id', 'producto'])

# !!!!!!!!!!!!! LIGTHFM !!!!!!!!!!!!!

# -------------------------
# PREPARAR DATOS
# -------------------------

top_municipios = b2c['municipio'].value_counts().nlargest(50).index
b2c['municipio'] = b2c['municipio'].where(b2c['municipio'].isin(top_municipios), 'otros')

top_asesores = b2c['asesor'].value_counts().nlargest(50).index
b2c['asesor'] = b2c['asesor'].where(b2c['asesor'].isin(top_asesores), 'otros')

top_puntos = b2c['punto de venta'].value_counts().nlargest(50).index
b2c['punto de venta'] = b2c['punto de venta'].where(b2c['punto de venta'].isin(top_puntos), 'otros')

# Inicializar dataset
dataset = Dataset()

# USER features
user_features_list = (
    [f"cluster:{x}" for x in b2c['cluster'].astype(str).unique()] +
    [f"municipio:{x}" for x in b2c['municipio'].astype(str).unique()] +
    [f"asesor:{x}" for x in b2c['asesor'].astype(str).unique()] +
    [f"punto:{x}" for x in b2c['punto de venta'].astype(str).unique()] +
    b2c.columns[b2c.columns.str.startswith("zona_")].tolist()
)

# ITEM features
item_features_list = (
    [f"categoria_macro:{x}" for x in b2c['categoria_macro'].astype(str).unique()] +
    [f"subcategoria:{x}" for x in b2c['subcategoria'].astype(str).unique()] +
    [f"color:{x}" for x in b2c['color'].astype(str).unique()]
)

# Fit mappings
dataset.fit(
    users=b2c['id'].unique(),
    items=b2c['producto'].unique(),
    user_features=user_features_list,
    item_features=item_features_list
)

# Interacciones implícitas (puedes usar .value o .cantidad como peso si quieres ponderar)
interactions, _ = dataset.build_interactions([
    (row['id'], row['producto']) for _, row in b2c.iterrows()
])


# -------------------------
# CONSTRUIR FEATURES
# -------------------------

def construir_features_usuario(df):
    features = []
    for _, row in df.iterrows():
        f = [
            f"cluster:{row['cluster']}",
            f"municipio:{row['municipio']}",
            f"asesor:{row['asesor']}",
            f"punto:{row['punto de venta']}"
        ]
        for zona in df.columns[df.columns.str.startswith("zona_")]:
            if row[zona] == 1:
                f.append(zona)
        features.append((row['id'], f))
    return features

def construir_features_producto(df):
    features = []
    for _, row in df.iterrows():
        f = [
            f"categoria_macro:{row['categoria_macro']}",
            f"subcategoria:{row['subcategoria']}",
            f"color:{row['color']}"
        ]
        features.append((row['producto'], f))
    return features

user_features = dataset.build_user_features(construir_features_usuario(b2c))
item_features = dataset.build_item_features(construir_features_producto(b2c))

# -------------------------
# ENTRENAMIENTO Y EVALUACIÓN
# -------------------------

# Quitar duplicados de (id, producto) antes de construir interacciones
b2c_interacciones = b2c[['id', 'producto']].drop_duplicates()

interactions, _ = dataset.build_interactions([
    (row['id'], row['producto']) for _, row in b2c_interacciones.iterrows()
])

# Dividir en train y test
train, test = random_train_test_split(interactions, test_percentage=0.2, random_state=42)

# Entrenar modelo
model = LightFM(loss='warp-kos', no_components=16, random_state=20)
model.fit(train, user_features=user_features, item_features=item_features, epochs=15, num_threads=4)

# GUARDAR LIGTHFM MODEL

 # Crear carpeta si no existe
os.makedirs("modelos", exist_ok=True)

# Define the path to save the model
model_path = './modelos/lightfm_model_b2c.pkl'

# Save the trained LightFM model
dump(model, model_path)

print(f"Model saved successfully to {model_path}")

dump(user_features, './modelos/b2c_user_features.pkl')
dump(item_features, './modelos/b2c_item_features.pkl')

# Optional: Save the dataset object and mappings as well if needed for loading
dataset_path = './modelos/lightfm_dataset_b2c.pkl'
dump(dataset, dataset_path)
print(f"Dataset saved successfully to {dataset_path}")

# !!!!!!!!!!!!! XGBOOST !!!!!!!!!!!!!

positivos = b2c[['id', 'producto']].drop_duplicates().copy()
positivos['interaccion'] = 1

from tqdm import tqdm  # barra de progreso

usuarios = b2c['id'].unique()
productos = b2c['producto'].unique()
compras = b2c.groupby('id')['producto'].apply(set).to_dict()

negativos = []
rng = np.random.default_rng(seed=42)

# Número de negativos por usuario
k_negativos = 5

for u in tqdm(usuarios):
    comprados = compras.get(u, set())
    candidatos = list(set(productos) - comprados)

    if len(candidatos) == 0:
        continue  # este usuario ya compró todo

    muestra = rng.choice(candidatos, size=min(k_negativos, len(candidatos)), replace=False)
    negativos.extend([(u, p, 0) for p in muestra])

# Convertir a DataFrame
negativos = pd.DataFrame(negativos, columns=['id', 'producto', 'interaccion'])

# Positivos
positivos = b2c[['id', 'producto']].drop_duplicates().copy()
positivos['interaccion'] = 1

# Unir todo
df_tabular = pd.concat([positivos, negativos], axis=0, ignore_index=True)

features_usuario = b2c.drop_duplicates('id').set_index('id')[
    ['edad', 'edad_promedio',
       'ingreso_laboral_promedio', 'GINI', 'IPUG',
       'cluster']
]

features_producto = b2c.drop_duplicates('producto').set_index('producto')[
    ['categoria_macro', 'subcategoria', 'color', 'precio', 'alineación con portafolio estratégico']
]

df_tabular = df_tabular.merge(features_usuario, on='id', how='left')
df_tabular = df_tabular.merge(features_producto, on='producto', how='left')

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# --- 1. Selección de columnas útiles (excluye id, producto, interaccion) ---
X = df_tabular.drop(columns=['id', 'producto', 'interaccion'])
y = df_tabular['interaccion']

# --- 2. Tratar nulos y reducir cardinalidad ---
X.fillna("desconocido", inplace=True)

# Reducir cardinalidad de 'color' si tiene muchos valores
if X['color'].nunique() > 20:
    top_colores = X['color'].value_counts().nlargest(20).index
    X['color'] = X['color'].where(X['color'].isin(top_colores), 'otros')

# --- 3. Convertir categóricas a tipo 'category' ---
categoricas = ['cluster', 'categoria_macro', 'subcategoria', 'color']
for col in categoricas:
    X[col] = X[col].astype('category')

# --- 4. Split y entrenamiento ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

modelo_xgb = XGBClassifier(
    tree_method='hist',
    enable_categorical=True,
    use_label_encoder=False,
    eval_metric='logloss',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)

modelo_xgb.fit(X_train, y_train)

# Define the path to save the XGBoost model
model_xgb_path = './modelos/xgb_model_b2c.pkl'

# Save the trained XGBoost model
dump(modelo_xgb, model_xgb_path)

print(f"XGBoost model saved successfully to {model_xgb_path}")

# Guardar features para uso en el recomendador
dump(features_usuario.reset_index(), './modelos/b2c_user_features_df.pkl')
dump(features_producto.reset_index(), './modelos/b2c_item_features_df.pkl')



