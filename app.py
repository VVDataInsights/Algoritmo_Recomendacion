import dash
from dash import html, dcc, Input, Output, State, dash_table
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from funciones.recomendadorB2B import recomendar_hibrido_b2b
from funciones.recomendadorB2C import recomendar_hibrido

# Cargar datos para obtener clientes B2B y B2C
try:
    b2b_data = pd.read_csv("./b2b_nuevo.csv")
    unique_clients_b2b = sorted(b2b_data['id_b2b'].unique())
except:
    unique_clients_b2b = ['B2B_01', 'B2B_02', 'B2B_03']

try:
    b2c_data = pd.read_csv("./b2c_nuevo.csv")
    unique_clients_b2c = sorted(b2c_data['id'].unique())
except:
    unique_clients_b2c = ['B2C_01', 'B2C_02', 'B2C_03']

app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Recomendador Híbrido B2B & B2C"
server = app.server

# Paleta de colores Corona - Escala de azules institucionales
COLORS = {
    'primary': '#1e3a8a',          # Azul Corona principal
    'secondary': '#1e40af',        # Azul medio
    'accent': '#3b82f6',           # Azul brillante
    'success': '#0ea5e9',          # Azul cielo
    'warning': '#0284c7',          # Azul información
    'danger': '#0369a1',           # Azul oscuro
    'background': '#f0f9ff',       # Azul muy claro
    'card': '#ffffff',             # Blanco
    'text': '#1e293b',             # Gris azulado oscuro
    'text_light': '#64748b',       # Gris azulado
    'border': '#cbd5e1',           # Gris azulado claro
    'dark': '#0f172a',             # Azul muy oscuro
    'b2b_primary': '#1e3a8a',      # Azul Corona B2B
    'b2c_primary': '#3b82f6',      # Azul brillante B2C
    'light_blue': '#dbeafe',       # Azul muy claro
    'medium_blue': '#93c5fd',      # Azul medio claro
    'gradient_start': '#1e3a8a',   # Inicio gradiente
    'gradient_end': '#3b82f6'      # Final gradiente
}


# Métricas de rendimiento
metricas_xgb_b2b = {'Precision': 0.80, 'Recall': 0.81, 'F1': 0.79, 'AUC': 0.8735}
metricas_lfm_b2b = {'Precision': 0.9333, 'Recall': 0.0048, 'F1': 0.008, 'AUC': 0.8779}
metricas_hibrido_b2b = {'Promedio Score': 0.7443, 'Max Score': 0.8067, 'Min Score': 0.7112, 'Desvest Score': 0.0292}

# Métricas simuladas para B2C
metricas_xgb_b2c = {'Precision': 0.84, 'Recall': 0.86, 'F1': 0.85, 'AUC': 0.9138}
metricas_lfm_b2c = {'Precision': 0.0119, 'Recall': 0.0657, 'F1': 0.005, 'AUC': 0.9365}
metricas_hibrido_b2c = {'Promedio Score': 0.9717, 'Max Score': 0.9845, 'Min Score': 0.9631, 'Desvest Score': 0.0072}

# Estilos CSS personalizados con paleta azul Corona
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
        <style>
            * {
                font-family: 'Inter', sans-serif;
            }
            body {
                margin: 0;
                background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
                min-height: 100vh;
            }
            .main-container {
                background: #f0f9ff;
                min-height: 100vh;
                padding: 20px;
            }
            .header-container {
                background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 50%, #3b82f6 100%);
                padding: 40px 20px;
                margin: -20px -20px 30px -20px;
                border-radius: 0 0 20px 20px;
                box-shadow: 0 10px 30px rgba(30, 58, 138, 0.4);
            }
            .tabs-container {
                margin-bottom: 30px;
            }
            .tabs-content {
                background: white;
                border-radius: 0 15px 15px 15px;
                box-shadow: 0 4px 20px rgba(30, 58, 138, 0.1);
                border: 1px solid #cbd5e1;
            }
            .control-panel {
                background: white;
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0 4px 20px rgba(30, 58, 138, 0.1);
                margin-bottom: 30px;
                border: 1px solid #cbd5e1;
                border-left: 4px solid #1e3a8a;
            }
            .metrics-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }
            .metric-card {
                background: white;
                padding: 25px;
                border-radius: 12px;
                box-shadow: 0 4px 15px rgba(30, 58, 138, 0.1);
                text-align: center;
                border: 1px solid #cbd5e1;
                border-top: 4px solid #1e3a8a;
                transition: transform 0.2s ease, box-shadow 0.2s ease;
            }
            .metric-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(30, 58, 138, 0.15);
            }
            .chart-container {
                background: white;
                padding: 25px;
                border-radius: 15px;
                box-shadow: 0 4px 20px rgba(30, 58, 138, 0.1);
                margin-bottom: 30px;
                border: 1px solid #cbd5e1;
                border-left: 4px solid #3b82f6;
            }
            .table-container {
                background: white;
                padding: 25px;
                border-radius: 15px;
                box-shadow: 0 4px 20px rgba(30, 58, 138, 0.1);
                border: 1px solid #cbd5e1;
                border-left: 4px solid #0ea5e9;
            }
            .control-item {
                margin-bottom: 25px;
            }
            .control-label {
                font-weight: 600;
                color: #1e293b;
                margin-bottom: 8px;
                display: block;
                font-size: 14px;
            }
            .run-button {
                background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
                color: white;
                border: none;
                padding: 15px 30px;
                border-radius: 8px;
                font-weight: 600;
                font-size: 16px;
                cursor: pointer;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(30, 58, 138, 0.3);
                width: 100%;
            }
            .run-button:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(30, 58, 138, 0.4);
                background: linear-gradient(135deg, #1e40af 0%, #0ea5e9 100%);
            }
            .run-button.b2c {
                background: linear-gradient(135deg, #3b82f6 0%, #0ea5e9 100%);
                box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
            }
            .run-button.b2c:hover {
                box-shadow: 0 8px 25px rgba(59, 130, 246, 0.4);
                background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);
            }

            /* Personalización de pestañas */
            .tab-parent {
                border-bottom: 2px solid #1e3a8a !important;
            }

            /* Personalización de dropdowns y sliders */
            .Select-control {
                border-color: #cbd5e1 !important;
            }
            .Select-control:hover {
                border-color: #1e3a8a !important;
            }
            .Select--is-focused .Select-control {
                border-color: #1e3a8a !important;
                box-shadow: 0 0 0 1px #1e3a8a !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

def create_metric_card(title, value, color, icon):
    return html.Div([
        html.Div([
            html.I(className=f"fas {icon}", style={
                'fontSize': '24px', 
                'color': color,
                'marginBottom': '10px'
            }),
            html.H3(f"{value:.3f}", style={
                'margin': '0', 
                'color': COLORS['text'],
                'fontSize': '28px',
                'fontWeight': '700'
            }),
            html.P(title, style={
                'margin': '5px 0 0 0', 
                'color': COLORS['text_light'],
                'fontSize': '14px',
                'fontWeight': '500'
            })
        ])
    ], className="metric-card")

def create_control_panel(model_type, clients_list):
    button_class = "run-button b2c" if model_type == "B2C" else "run-button"
    return html.Div([
        html.H3([
            html.I(className="fas fa-cogs", style={'marginRight': '10px', 'color': COLORS['primary']}),
            f"Configuración del Análisis {model_type}"
        ], style={
            'color': COLORS['text'], 
            'marginBottom': '25px',
            'fontSize': '20px',
            'fontWeight': '600'
        }),
        
        html.Div([
            html.Label(f"Cliente {model_type}", className="control-label"),
            dcc.Dropdown(
                id=f"cliente_id_{model_type.lower()}",
                options=[{'label': f"{c}", 'value': c} for c in clients_list],
                value=clients_list[0] if clients_list else None,
                style={'marginBottom': '15px'},
                placeholder=f"Selecciona un cliente {model_type}..."
            )
        ], className="control-item"),

        html.Div([
            html.Label("Número de Recomendaciones (Top N)", className="control-label"),
            dcc.Slider(
                id=f"topn_{model_type.lower()}",
                min=1, max=30, step=1, value=10,
                marks={i: str(i) for i in range(5, 31, 5)},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], className="control-item"),

        html.Div([
            html.Label("Factor de Combinación Alpha (0=XGB, 1=LFM)", className="control-label"),
            dcc.Slider(
                id=f"alpha_{model_type.lower()}",
                min=0.0, max=1.0, step=0.05, value=0.5,
                marks={i/10: f"{i/10:.1f}" for i in range(0, 11, 2)},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], className="control-item"),

        html.Button([
            html.I(className="fas fa-play", style={'marginRight': '8px'}),
            f"Ejecutar Análisis {model_type}"
        ], id=f"run_button_{model_type.lower()}", n_clicks=0, className=button_class)
    ], className="control-panel")

def create_metrics_section(model_type, metrics):
    primary_color = COLORS['b2c_primary'] if model_type == "B2C" else COLORS['b2b_primary']
    return html.Div([
        html.H2([
            html.I(className="fas fa-chart-line", style={'marginRight': '10px', 'color': primary_color}),
            f"Métricas de Rendimiento del Modelo Híbrido {model_type}"
        ], style={
            'color': COLORS['text'],
            'marginBottom': '20px',
            'fontSize': '24px',
            'fontWeight': '600'
        }),
        html.Div([
            create_metric_card("Promedio Score", metrics['Promedio Score'], COLORS['primary'], "fa-bullseye"),
            create_metric_card("Max Score", metrics['Max Score'], COLORS['success'], "fa-search"),
            create_metric_card("Min Score", metrics['Min Score'], COLORS['warning'], "fa-balance-scale"),
            create_metric_card("Desvest Score", metrics['Desvest Score'], COLORS['accent'], "fa-trophy")
        ], className="metrics-grid")
    ], style={'marginBottom': '30px'})

# Layout principal
app.layout = html.Div([
    # LOGO ESTÁTICO
    html.Img(
        src='/assets/logoCorona.png',
        style={
            'position': 'fixed',
            'top': '40px',
            'left': '20px',
            'height': '60px',
            'zIndex': '1000',
            'backgroundColor': 'white',
            'padding': '5px',
            'borderRadius': '10px',
            'boxShadow': '0 4px 12px rgba(30, 58, 138, 0.2)',
            'border': '2px solid #dbeafe'
        }
    ),

    # Header
    html.Div([
        html.Div([
            html.H1([
                html.I(className="fas fa-brain", style={'marginRight': '15px'}),
                "Sistema de Recomendación Híbrido B2B & B2C"
            ], style={
                'textAlign': 'center', 
                'color': 'white',
                'margin': '0 0 10px 0',
                'fontSize': '36px',
                'fontWeight': '700',
                'textShadow': '2px 2px 4px rgba(0,0,0,0.3)'
            }),
            html.P("Optimiza tus recomendaciones con algoritmos híbridos avanzados de Machine Learning",
                   style={
                       'textAlign': 'center', 
                       'color': 'rgba(255,255,255,0.95)',
                       'margin': '0',
                       'fontSize': '18px',
                       'fontWeight': '400'
                   })
        ])
    ], className="header-container"),

    # Pestañas
    html.Div([
        dcc.Tabs(
            id="main_tabs",
            value="b2b",
            children=[
                dcc.Tab(
                    label="B2B - Business to Business",
                    value="b2b",
                    style={
                        'padding': '12px 24px',
                        'fontWeight': '600',
                        'fontSize': '16px',
                        'borderRadius': '15px 15px 0 0',
                        'backgroundColor': '#f0f9ff',
                        'border': '1px solid #cbd5e1'
                    },
                    selected_style={
                        'backgroundColor': COLORS['b2b_primary'],
                        'color': 'white',
                        'borderBottom': 'none',
                        'borderTop': '3px solid #0f172a'
                    }
                ),
                dcc.Tab(
                    label="B2C - Business to Consumer",
                    value="b2c",
                    style={
                        'padding': '12px 24px',
                        'fontWeight': '600',
                        'fontSize': '16px',
                        'borderRadius': '15px 15px 0 0',
                        'backgroundColor': '#f0f9ff',
                        'border': '1px solid #cbd5e1'
                    },
                    selected_style={
                        'backgroundColor': COLORS['b2c_primary'],
                        'color': 'white',
                        'borderBottom': 'none',
                        'borderTop': '3px solid #0f172a'
                    }
                )
            ],
            style={
                'height': '60px'
            }
        ),
        html.Div(id="tabs_content", className="tabs-content")
    ], className="tabs-container")

], className="main-container")

@app.callback(
    Output("tabs_content", "children"),
    Input("main_tabs", "value")
)
def render_tab_content(active_tab):
    if active_tab == "b2b":
        return html.Div([
            # Métricas B2B
            create_metrics_section("B2B", metricas_hibrido_b2b),
            
            # Panel de control B2B
            create_control_panel("B2B", unique_clients_b2b),

            # Gráficos B2B
            html.Div([
                html.H3([
                    html.I(className="fas fa-chart-bar", style={'marginRight': '10px', 'color': COLORS['primary']}),
                    "Comparación de Modelos"
                ], style={
                    'color': COLORS['text'], 
                    'marginBottom': '20px',
                    'fontSize': '20px',
                    'fontWeight': '600'
                }),
                dcc.Graph(id="comparison_graph_b2b")
            ], className="chart-container"),

            html.Div([
                html.H3([
                    html.I(className="fas fa-sort-amount-down", style={'marginRight': '10px', 'color': COLORS['primary']}),
                    "Ranking de Productos"
                ], style={
                    'color': COLORS['text'], 
                    'marginBottom': '20px',
                    'fontSize': '20px',
                    'fontWeight': '600'
                }),
                dcc.Graph(id="score_graph_b2b")
            ], className="chart-container"),

            html.Div([
                html.H3([
                    html.I(className="fas fa-scatter-chart", style={'marginRight': '10px', 'color': COLORS['primary']}),
                    "Análisis de Valor vs Alineación Estratégica"
                ], style={
                    'color': COLORS['text'], 
                    'marginBottom': '20px',
                    'fontSize': '20px',
                    'fontWeight': '600'
                }),
                dcc.Graph(id="value_graph_b2b")
            ], className="chart-container"),

            # Tabla de resultados B2B
            html.Div([
                html.H3([
                    html.I(className="fas fa-table", style={'marginRight': '10px', 'color': COLORS['primary']}),
                    "Resultados Detallados"
                ], style={
                    'color': COLORS['text'], 
                    'marginBottom': '20px',
                    'fontSize': '20px',
                    'fontWeight': '600'
                }),
                html.Div(id="tabla_resultados_b2b")
            ], className="table-container")
        ], style={'padding': '30px'})
    
    else:  # B2C
        return html.Div([
            # Métricas B2C
            create_metrics_section("B2C", metricas_hibrido_b2c),
            
            # Panel de control B2C
            create_control_panel("B2C", unique_clients_b2c),

            # Gráficos B2C
            html.Div([
                html.H3([
                    html.I(className="fas fa-chart-bar", style={'marginRight': '10px', 'color': COLORS['b2c_primary']}),
                    "Comparación de Modelos"
                ], style={
                    'color': COLORS['text'], 
                    'marginBottom': '20px',
                    'fontSize': '20px',
                    'fontWeight': '600'
                }),
                dcc.Graph(id="comparison_graph_b2c")
            ], className="chart-container"),

            html.Div([
                html.H3([
                    html.I(className="fas fa-sort-amount-down", style={'marginRight': '10px', 'color': COLORS['b2c_primary']}),
                    "Ranking de Productos"
                ], style={
                    'color': COLORS['text'], 
                    'marginBottom': '20px',
                    'fontSize': '20px',
                    'fontWeight': '600'
                }),
                dcc.Graph(id="score_graph_b2c")
            ], className="chart-container"),

            html.Div([
                html.H3([
                    html.I(className="fas fa-scatter-chart", style={'marginRight': '10px', 'color': COLORS['b2c_primary']}),
                    "Análisis de Valor vs Alineación Estratégica"
                ], style={
                    'color': COLORS['text'], 
                    'marginBottom': '20px',
                    'fontSize': '20px',
                    'fontWeight': '600'
                }),
                dcc.Graph(id="value_graph_b2c")
            ], className="chart-container"),

            # Tabla de resultados B2C
            html.Div([
                html.H3([
                    html.I(className="fas fa-table", style={'marginRight': '10px', 'color': COLORS['b2c_primary']}),
                    "Resultados Detallados"
                ], style={
                    'color': COLORS['text'], 
                    'marginBottom': '20px',
                    'fontSize': '20px',
                    'fontWeight': '600'
                }),
                html.Div(id="tabla_resultados_b2c")
            ], className="table-container")
        ], style={'padding': '30px'})

# Callbacks para B2B
@app.callback(
    [Output("comparison_graph_b2b", "figure"),
     Output("score_graph_b2b", "figure"),
     Output("value_graph_b2b", "figure"),
     Output("tabla_resultados_b2b", "children")],
    Input("run_button_b2b", "n_clicks"),
    [State("cliente_id_b2b", "value"),
     State("topn_b2b", "value"),
     State("alpha_b2b", "value")]
)
def actualizar_dashboard_b2b(n_clicks, cliente_id, topn, alpha):
    return actualizar_dashboard_general("b2b", n_clicks, cliente_id, topn, alpha, "B2B", recomendar_hibrido_b2b)

# Callbacks para B2C
@app.callback(
    [Output("comparison_graph_b2c", "figure"),
     Output("score_graph_b2c", "figure"),
     Output("value_graph_b2c", "figure"),
     Output("tabla_resultados_b2c", "children")],
    Input("run_button_b2c", "n_clicks"),
    [State("cliente_id_b2c", "value"),
     State("topn_b2c", "value"),
     State("alpha_b2c", "value")]
)
def actualizar_dashboard_b2c(n_clicks, cliente_id, topn, alpha):
    return actualizar_dashboard_general("b2c", n_clicks, cliente_id, topn, alpha, "B2C", recomendar_hibrido)

def actualizar_dashboard_general(typeData, n_clicks, cliente_id, topn, alpha, model_type, recomendador_func):
    if n_clicks == 0 or not cliente_id:
        # Gráficos vacíos con mensaje
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            text=f"Haz clic en 'Ejecutar Análisis {model_type}' para generar los resultados",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color=COLORS['text_light'])
        )
        empty_fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis={'visible': False},
            yaxis={'visible': False}
        )
        
        return empty_fig, empty_fig, empty_fig, html.Div([
            html.P(f"👆 Configura los parámetros y ejecuta el análisis {model_type} para ver los resultados", 
                   style={'textAlign': 'center', 'color': COLORS['text_light'], 'fontSize': '16px'})
        ])

    if typeData == "b2b":
        data = b2b_data
    else:
        data = b2c_data
    # Obtener recomendaciones
    df = recomendador_func(data, cliente_id=cliente_id, top_n=topn, alpha=alpha)
    if isinstance(df, str):
        error_fig = go.Figure()
        error_fig.add_annotation(
            text=f"Error: {df}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color=COLORS['danger'])
        )
        error_fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis={'visible': False},
            yaxis={'visible': False}
        )
        return error_fig, error_fig, error_fig, html.Div([
            html.P(f"{df}", style={'textAlign': 'center', 'color': COLORS['danger']})
        ])

    # Normalizar scores para ambos modelos si es necesario
    if 'score_lfm_norm' not in df.columns:
        if df['score_lfm'].nunique() > 1:
            min_lfm = df['score_lfm'].min()
            max_lfm = df['score_lfm'].max()
            df['score_lfm_norm'] = (df['score_lfm'] - min_lfm) / (max_lfm - min_lfm)
        else:
            df['score_lfm_norm'] = 0.5

    # Calcular score híbrido si no existe
    if 'score_hibrido' not in df.columns:
        df['score_hibrido'] = alpha * df['score_lfm_norm'] + (1 - alpha) * df['score_xgb']

    # Gráfico 1: Comparación de modelos con colores Corona azules
    df_melted = df.melt(
        id_vars='producto', 
        value_vars=['score_lfm_norm', 'score_xgb', 'score_hibrido'],
        var_name='modelo', 
        value_name='score'
    )
    
    # Paleta de azules Corona para los modelos
    model_colors = {
        'score_lfm_norm': COLORS['primary'],
        'score_xgb': COLORS['success'],
        'score_hibrido': COLORS['secondary']
    }

    fig1 = px.bar(
        df_melted, 
        x='producto', 
        y='score', 
        color='modelo',
        barmode='group',
        color_discrete_map=model_colors,
        title=f"Comparación de Scores por Modelo - {model_type}"
    )
    
    fig1.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Inter", size=12),
        title_font=dict(size=16, color=COLORS['text']),
        xaxis=dict(title="Productos", gridcolor='#f1f5f9'),
        yaxis=dict(title="Score Normalizado", gridcolor='#f1f5f9'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Gráfico 2: Ranking de productos
    fig2 = px.bar(
        df.sort_values("score_hibrido"), 
        x="score_hibrido", 
        y="producto", 
        orientation='h',
        color="score_hibrido",
        color_continuous_scale="Viridis",
        title=f"Ranking de Productos por Score Híbrido - {model_type}"
    )
    
    fig2.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Inter", size=12),
        title_font=dict(size=16, color=COLORS['text']),
        xaxis=dict(title="Score Híbrido", gridcolor='#f1f5f9'),
        yaxis=dict(title="Productos", gridcolor='#f1f5f9')
    )

    # Gráfico 3: Scatter plot - adaptable según las columnas disponibles
    # Para B2B: alineación con portafolio estratégico b2b, valor_esperado
    # Para B2C: alineación estratégica, valor_esperado
    x_col = 'alineación con portafolio estratégico b2b' if model_type == "B2B" else 'alineación estratégica'
    x_title = "Alineación con Portafolio Estratégico" if model_type == "B2B" else "Alineación Estratégica"

    if x_col in df.columns and 'valor_esperado' in df.columns:
        fig3 = px.scatter(
            df,
            x=x_col,
            y='valor_esperado',
            size=np.abs(df['score_hibrido']),
            color='score_hibrido',
            hover_data=['producto'],
            size_max=50,
            color_continuous_scale="Plasma",
            title=f"Valor Esperado vs {x_title} - {model_type}"
        )

        fig3.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Inter", size=12),
            title_font=dict(size=16, color=COLORS['text']),
            xaxis=dict(title=x_title, gridcolor='#f1f5f9'),
            yaxis=dict(title="Valor Esperado", gridcolor='#f1f5f9')
        )
    else:
        fig3 = go.Figure()
        fig3.add_annotation(
            text="Datos de valor esperado no disponibles",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color=COLORS['text_light'])
        )
        fig3.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis={'visible': False},
            yaxis={'visible': False}
        )

    # Tabla estilizada - adaptable a las columnas disponibles
    tabla = dash_table.DataTable(
        columns = [
            {"name": "Producto", "id": "producto"},
            {"name": "Score LFM Norm", "id": "score_lfm_norm", "type": "numeric", "format": {"specifier": ".3f"}},
            {"name": "Score XGB", "id": "score_xgb", "type": "numeric", "format": {"specifier": ".3f"}},
            {"name": "Score Híbrido", "id": "score_hibrido", "type": "numeric", "format": {"specifier": ".3f"}},
            {"name": "Alineación Estratégica", "id": "alineación estratégica" if model_type == "B2C" else "alineación con portafolio estratégico b2b"},
            {"name": "Valor Esperado", "id": "valor_esperado", "type": "numeric", "format": {"specifier": ".2f"}}
        ],
        data=df.round(6).to_dict('records'),
        page_size=topn,
        style_cell={
            'textAlign': 'left',
            'fontFamily': 'Inter',
            'fontSize': '14px',
            'padding': '12px'
        },
        style_header={
            'backgroundColor': COLORS['primary'],
            'color': 'white',
            'fontWeight': '600',
            'textAlign': 'center'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': '#f8fafc'
            }
        ],
        style_data={
            'border': '1px solid #e2e8f0'
        }
    )
    return fig1, fig2, fig3, tabla

if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
