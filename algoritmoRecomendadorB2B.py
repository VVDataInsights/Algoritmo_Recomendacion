import pandas as pd
import numpy as np
from joblib import dump
from sklearn.preprocessing import LabelEncoder
from lightfm import LightFM
from lightfm.data import Dataset
import os

# Cargar el dataset
pd2 = pd.read_csv('./b2b_nuevo.csv' , sep=",", encoding="utf-8")
b2b = pd2.copy()

categorical_cols = b2b.select_dtypes(include=['object', 'category']).columns
print(categorical_cols)

cat_cols = ['id_b2b', 'municipio', 'zona', 'categoria_b2b_macro',
       'categoria_b2b', 'subcategoria_b2b', 'producto']

label_encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    b2b[f'{col}_enc'] = le.fit_transform(b2b[col].astype(str))
    label_encoders[col] = le

# Crear carpeta si no existe
os.makedirs("modelos", exist_ok=True)

# !!!!!!!!!!!!! LIGTHFM !!!!!!!!!!!!!
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k, recall_at_k, auc_score
import numpy as np

b2b[cat_cols] = b2b[cat_cols].astype(str)

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

# Entrenamiento
model_lfm = LightFM(loss='warp')
model_lfm.fit(interactions,
          user_features=user_features,
          item_features=item_features,
          epochs=10,
          num_threads=4)

# Define the path to save the model
model_path = './modelos/lightfm_model_b2b.pkl'

# Save the trained LightFM model
dump(model_lfm, model_path)

print(f"Model saved successfully to {model_path}")

# Optional: Save the dataset object and mappings as well if needed for loading
dataset_path = './modelos/lightfm_dataset_b2b.pkl'
dump(dataset, dataset_path)
print(f"Dataset saved successfully to {dataset_path}")


# !!!!!!!!!!!!! XGBOOST !!!!!!!!!!!!!

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import itertools

# --- Preprocesamiento ---
b2b[cat_cols] = b2b[cat_cols].astype(str)
b2b['comprado'] = 1

# --- Construcción de universo completo cliente-producto ---
clientes = b2b['id_b2b'].unique()
productos = b2b['producto'].unique()

combinaciones = pd.DataFrame(list(itertools.product(clientes, productos)), columns=['id_b2b', 'producto'])

# Marcar los comprados reales
comprados = b2b[['id_b2b', 'producto']].drop_duplicates()
comprados['comprado'] = 1

# Merge para crear dataset con comprados y no comprados
df_all = combinaciones.merge(comprados, on=['id_b2b', 'producto'], how='left')
df_all['comprado'] = df_all['comprado'].fillna(0)

# Filtrar: muestreamos mismos positivos y mismos negativos para balancear
positivos_df = df_all[df_all['comprado'] == 1]
negativos_df = df_all[df_all['comprado'] == 0].sample(n=len(positivos_df), random_state=42)
df_final = pd.concat([positivos_df, negativos_df], ignore_index=True)

# --- Enriquecer con info de cliente y producto ---
cliente_info = b2b.drop_duplicates('id_b2b').set_index('id_b2b')[[
    'municipio', 'zona', 'valor_total', 'Total de unidades', 'Total de edificaciones en obra'
]]
producto_info = b2b.drop_duplicates('producto').set_index('producto')[[
    'categoria_b2b_macro', 'categoria_b2b', 'subcategoria_b2b'
]]

df_final = df_final.join(cliente_info, on='id_b2b')
df_final = df_final.join(producto_info, on='producto')

# --- Codificación ---
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df_final[col] = le.fit_transform(df_final[col])
    encoders[col] = le

# --- Split en entrenamiento y holdout balanceado ---
positivos_df = df_final[df_final['comprado'] == 1]
negativos_df = df_final[df_final['comprado'] == 0]

train_pos, test_pos = train_test_split(positivos_df, test_size=0.2, random_state=42)
train_neg, test_neg = train_test_split(negativos_df, test_size=0.2, random_state=42)

train_df = pd.concat([train_pos, train_neg]).sample(frac=1, random_state=42).reset_index(drop=True)
test_df = pd.concat([test_pos, test_neg]).sample(frac=1, random_state=42).reset_index(drop=True)

X_train = train_df.drop(columns=['comprado'])
y_train = train_df['comprado']
X_test = test_df.drop(columns=['comprado'])
y_test = test_df['comprado']

# --- Modelo con regularización ---
model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    max_depth=4,
    learning_rate=0.05,
    n_estimators=300,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=5,
    reg_lambda=10,
    random_state=42
)

# Define the path to save the XGBoost model
model_xgb_path = './modelos/xgb_model_b2b.pkl'

# Save the trained XGBoost model
dump(model, model_xgb_path)

print(f"XGBoost model saved successfully to {model_xgb_path}")

# Save the encoders if needed for loading and predicting
encoders_path = './modelos/encoders_b2b.pkl'
dump(encoders, encoders_path)
print(f"Encoders saved successfully to {encoders_path}")





