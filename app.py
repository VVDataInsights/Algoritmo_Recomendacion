import dash
from dash import html, dcc, Input, Output, State, dash_table
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from funciones.recomendador import recomendar_hibrido_b2b

# Cargar datos para obtener clientes 
try:
    b2b_data = pd.read_csv("./b2b_nuevo.csv")
    unique_clients = sorted(b2b_data['id_b2b'].unique())
except:
    unique_clients = ['B2B_01', 'B2B_02', 'B2B_03']

app = dash.Dash(__name__)
app.title = "Recomendador B2B Híbrido"
server = app.server

# Paleta de colores moderna y profesional
COLORS = {
    'primary': '#667eea',
    'secondary': '#764ba2',
    'accent': '#f093fb',
    'success': '#4facfe',
    'warning': '#43e97b',
    'danger': '#fa709a',
    'background': '#f8fafc',
    'card': '#ffffff',
    'text': '#2d3748',
    'text_light': '#718096',
    'border': '#e2e8f0',
    'dark': '#1a202c'
}

# Métricas de rendimiento
metricas_xgb_b2b = {'Precision': 0.8542, 'Recall': 0.7834, 'F1': 0.8172, 'NDCG': 0.8901}
metricas_lfm_b2b = {'Precision': 0.7923, 'Recall': 0.8134, 'F1': 0.8027, 'NDCG': 0.8234}
metricas_hibrido_b2b = {'Precision': 0.8734, 'Recall': 0.8456, 'F1': 0.8593, 'NDCG': 0.9123}

# Estilos CSS personalizados
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
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .main-container {
                background: #f8fafc;
                min-height: 100vh;
                padding: 20px;
            }
            .header-container {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 40px 20px;
                margin: -20px -20px 30px -20px;
                border-radius: 0 0 20px 20px;
                box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
            }
            .control-panel {
                background: white;
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
                margin-bottom: 30px;
                border: 1px solid #e2e8f0;
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
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
                text-align: center;
                border: 1px solid #e2e8f0;
                transition: transform 0.2s ease, box-shadow 0.2s ease;
            }
            .metric-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(0, 0, 0, 0.12);
            }
            .chart-container {
                background: white;
                padding: 25px;
                border-radius: 15px;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
                margin-bottom: 30px;
                border: 1px solid #e2e8f0;
            }
            .table-container {
                background: white;
                padding: 25px;
                border-radius: 15px;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
                border: 1px solid #e2e8f0;
            }
            .control-item {
                margin-bottom: 25px;
            }
            .control-label {
                font-weight: 600;
                color: #2d3748;
                margin-bottom: 8px;
                display: block;
                font-size: 14px;
            }
            .run-button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 15px 30px;
                border-radius: 8px;
                font-weight: 600;
                font-size: 16px;
                cursor: pointer;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
                width: 100%;
            }
            .run-button:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
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
            'boxShadow': '0 4px 12px rgba(0,0,0,0.1)'
        }
    ),

    # Header
    html.Div([
        html.Div([
            html.H1([
                html.I(className="fas fa-brain", style={'marginRight': '15px'}),
                "Sistema de Recomendación Híbrido B2B"
            ], style={
                'textAlign': 'center', 
                'color': 'white',
                'margin': '0 0 10px 0',
                'fontSize': '36px',
                'fontWeight': '700'
            }),
            html.P("Optimiza tus recomendaciones con algoritmos híbridos avanzados de Machine Learning",
                   style={
                       'textAlign': 'center', 
                       'color': 'rgba(255,255,255,0.9)',
                       'margin': '0',
                       'fontSize': '18px',
                       'fontWeight': '400'
                   })
        ])
    ], className="header-container"),

    # Métricas de rendimiento
    html.Div([
        html.H2([
            html.I(className="fas fa-chart-line", style={'marginRight': '10px'}),
            "Métricas de Rendimiento del Modelo Híbrido"
        ], style={
            'color': COLORS['text'], 
            'marginBottom': '20px',
            'fontSize': '24px',
            'fontWeight': '600'
        }),
        html.Div([
            create_metric_card("Precision", metricas_hibrido_b2b['Precision'], COLORS['primary'], "fa-bullseye"),
            create_metric_card("Recall", metricas_hibrido_b2b['Recall'], COLORS['success'], "fa-search"),
            create_metric_card("F1-Score", metricas_hibrido_b2b['F1'], COLORS['warning'], "fa-balance-scale"),
            create_metric_card("NDCG", metricas_hibrido_b2b['NDCG'], COLORS['danger'], "fa-trophy")
        ], className="metrics-grid")
    ], style={'marginBottom': '30px'}),

    # Panel de control
    html.Div([
        html.H3([
            html.I(className="fas fa-cogs", style={'marginRight': '10px'}),
            "Configuración del Análisis"
        ], style={
            'color': COLORS['text'], 
            'marginBottom': '25px',
            'fontSize': '20px',
            'fontWeight': '600'
        }),
        
        html.Div([
            html.Label("Cliente B2B", className="control-label"),
            dcc.Dropdown(
                id="cliente_id",
                options=[{'label': f"{c}", 'value': c} for c in unique_clients],
                value=unique_clients[0],
                style={'marginBottom': '15px'},
                placeholder="Selecciona un cliente..."
            )
        ], className="control-item"),

        html.Div([
            html.Label("Número de Recomendaciones (Top N)", className="control-label"),
            dcc.Slider(
                id="topn",
                min=1, max=30, step=1, value=10,
                marks={i: str(i) for i in range(5, 31, 5)},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], className="control-item"),

        html.Div([
            html.Label("Factor de Combinación Alpha (0=XGB, 1=LFM)", className="control-label"),
            dcc.Slider(
                id="alpha",
                min=0.0, max=1.0, step=0.05, value=0.5,
                marks={i/10: f"{i/10:.1f}" for i in range(0, 11, 2)},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], className="control-item"),

        html.Button([
            html.I(className="fas fa-play", style={'marginRight': '8px'}),
            "Ejecutar Análisis"
        ], id="run_button", n_clicks=0, className="run-button")
    ], className="control-panel"),

    # Gráficos
    html.Div([
        html.H3([
            html.I(className="fas fa-chart-bar", style={'marginRight': '10px'}),
            "Comparación de Modelos"
        ], style={
            'color': COLORS['text'], 
            'marginBottom': '20px',
            'fontSize': '20px',
            'fontWeight': '600'
        }),
        dcc.Graph(id="comparison_graph")
    ], className="chart-container"),

    html.Div([
        html.H3([
            html.I(className="fas fa-sort-amount-down", style={'marginRight': '10px'}),
            "Ranking de Productos"
        ], style={
            'color': COLORS['text'], 
            'marginBottom': '20px',
            'fontSize': '20px',
            'fontWeight': '600'
        }),
        dcc.Graph(id="score_graph")
    ], className="chart-container"),

    html.Div([
        html.H3([
            html.I(className="fas fa-scatter-chart", style={'marginRight': '10px'}),
            "Análisis de Valor vs Alineación Estratégica"
        ], style={
            'color': COLORS['text'], 
            'marginBottom': '20px',
            'fontSize': '20px',
            'fontWeight': '600'
        }),
        dcc.Graph(id="value_graph")
    ], className="chart-container"),

    # Tabla de resultados
    html.Div([
        html.H3([
            html.I(className="fas fa-table", style={'marginRight': '10px'}),
            "Resultados Detallados"
        ], style={
            'color': COLORS['text'], 
            'marginBottom': '20px',
            'fontSize': '20px',
            'fontWeight': '600'
        }),
        html.Div(id="tabla_resultados")
    ], className="table-container")

], className="main-container")

@app.callback(
    [Output("comparison_graph", "figure"),
     Output("score_graph", "figure"),
     Output("value_graph", "figure"),
     Output("tabla_resultados", "children")],
    Input("run_button", "n_clicks"),
    [State("cliente_id", "value"),
     State("topn", "value"),
     State("alpha", "value")]
)
def actualizar_dashboard(n_clicks, cliente_id, topn, alpha):
    if n_clicks == 0 or not cliente_id:
        # Gráficos vacíos con mensaje
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            text="Haz clic en 'Ejecutar Análisis' para generar los resultados",
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
            html.P("👆 Configura los parámetros y ejecuta el análisis para ver los resultados", 
                   style={'textAlign': 'center', 'color': COLORS['text_light'], 'fontSize': '16px'})
        ])

    # Obtener recomendaciones
    df = recomendar_hibrido_b2b(cliente_id=cliente_id, top_n=topn, alpha=alpha)
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

    # Gráfico 1: Comparación de modelos
    df['score_lfm_norm'] = (df['score_lfm'] - df['score_lfm'].min()) / (df['score_lfm'].max() - df['score_lfm'].min())
    df['score_hibrido_norm'] = alpha * df['score_lfm_norm'] + (1 - alpha) * df['score_xgb']
    
    df_melted = df.melt(
        id_vars='producto', 
        value_vars=['score_lfm_norm', 'score_xgb', 'score_hibrido_norm'],
        var_name='modelo', 
        value_name='score'
    )
    
    model_colors = {
        'score_lfm_norm': COLORS['primary'],
        'score_xgb': COLORS['secondary'], 
        'score_hibrido_norm': COLORS['accent']
    }
    
    fig1 = px.bar(
        df_melted, 
        x='producto', 
        y='score', 
        color='modelo',
        barmode='group',
        color_discrete_map=model_colors,
        title="Comparación de Scores por Modelo"
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
        title="Ranking de Productos por Score Híbrido"
    )
    
    fig2.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Inter", size=12),
        title_font=dict(size=16, color=COLORS['text']),
        xaxis=dict(title="Score Híbrido", gridcolor='#f1f5f9'),
        yaxis=dict(title="Productos", gridcolor='#f1f5f9')
    )

    # Gráfico 3: Scatter plot valor vs alineación
    fig3 = px.scatter(
        df, 
        x='alineación con portafolio estratégico b2b', 
        y='valor_esperado',
        size=np.abs(df['score_hibrido']), 
        color='score_hibrido',
        hover_data=['producto'], 
        size_max=50,
        color_continuous_scale="Plasma",
        title="Valor Esperado vs Alineación Estratégica"
    )
    
    fig3.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Inter", size=12),
        title_font=dict(size=16, color=COLORS['text']),
        xaxis=dict(title="Alineación con Portafolio Estratégico", gridcolor='#f1f5f9'),
        yaxis=dict(title="Valor Esperado", gridcolor='#f1f5f9')
    )

    # Tabla estilizada
    tabla = dash_table.DataTable(
        columns=[
            {"name": "Producto", "id": "producto"},
            {"name": "Score LFM Norm", "id": "score_lfm_norm", "type": "numeric", "format": {"specifier": ".3f"}},
            {"name": "Score XGB", "id": "score_xgb", "type": "numeric", "format": {"specifier": ".3f"}},
            {"name": "Score Híbrido", "id": "score_hibrido", "type": "numeric", "format": {"specifier": ".3f"}},
            {"name": "Precio Promedio", "id": "precio_promedio", "type": "numeric", "format": {"specifier": ",.0f"}},
            {"name": "Valor Esperado", "id": "valor_esperado", "type": "numeric", "format": {"specifier": ",.0f"}},
            {"name": "Alineación Estratégica", "id": "alineación con portafolio estratégico b2b", "type": "numeric", "format": {"specifier": ".6f"}}
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