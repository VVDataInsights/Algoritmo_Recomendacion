import dash
from dash import html, dcc, Input, Output, State, dash_table
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from funciones.recomendador import recomendar_hibrido_b2b

# Configuración de la app
app = dash.Dash(__name__)
app.title = "Recomendador B2B Híbrido"
server = app.server

# Estilos personalizados
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'success': '#C73E1D',
    'background': '#F5F7FA',
    'card': '#FFFFFF',
    'text': '#2C3E50',
    'light_gray': '#ECF0F1'
}

# Estilos CSS personalizados
external_stylesheets = ['https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Layout principal
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("Sistema de Recomendación Híbrido B2B", 
                style={
                    'textAlign': 'center',
                    'color': COLORS['primary'],
                    'fontFamily': 'Inter, sans-serif',
                    'fontWeight': '700',
                    'fontSize': '2.5rem',
                    'marginBottom': '0.5rem'
                }),
        html.P("Optimiza tus recomendaciones con algoritmos híbridos avanzados",
               style={
                   'textAlign': 'center',
                   'color': COLORS['text'],
                   'fontFamily': 'Inter, sans-serif',
                   'fontSize': '1.1rem',
                   'opacity': '0.8'
               })
    ], style={
        'backgroundColor': COLORS['card'],
        'padding': '2rem',
        'marginBottom': '2rem',
        'borderRadius': '15px',
        'boxShadow': '0 4px 15px rgba(0,0,0,0.1)'
    }),

    # Panel de control
    html.Div([
        html.Div([
            html.H3("Configuración", style={
                'color': COLORS['primary'],
                'fontFamily': 'Inter, sans-serif',
                'fontWeight': '600',
                'marginBottom': '1.5rem'
            }),
            
            # Cliente ID
            html.Div([
                html.Label("ID del Cliente:", style={
                    'fontWeight': '500',
                    'color': COLORS['text'],
                    'fontFamily': 'Inter, sans-serif',
                    'marginBottom': '0.5rem',
                    'display': 'block'
                }),
                dcc.Input(
                    id="cliente_id",
                    type="text",
                    value="B2B_01",
                    placeholder="Ingresa el ID del cliente...",
                    style={
                        'width': '100%',
                        'padding': '12px',
                        'border': f'2px solid {COLORS["light_gray"]}',
                        'borderRadius': '8px',
                        'fontSize': '16px',
                        'fontFamily': 'Inter, sans-serif',
                        'transition': 'border-color 0.3s ease'
                    }
                )
            ], style={'marginBottom': '1.5rem'}),

            # Top-N Slider
            html.Div([
                html.Label("Número de productos recomendados:", style={
                    'fontWeight': '500',
                    'color': COLORS['text'],
                    'fontFamily': 'Inter, sans-serif',
                    'marginBottom': '0.5rem',
                    'display': 'block'
                }),
                html.Div(id="topn-value", style={
                    'fontSize': '18px',
                    'fontWeight': '600',
                    'color': COLORS['primary'],
                    'marginBottom': '0.5rem'
                }),
                dcc.Slider(
                    id="topn",
                    min=1,
                    max=30,
                    step=1,
                    value=10,
                    marks={i: {'label': str(i), 'style': {'fontSize': '12px'}} for i in range(1, 31, 5)},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], style={'marginBottom': '1.5rem'}),

            # Alpha Slider
            html.Div([
                html.Label("Peso del modelo LightFM (α):", style={
                    'fontWeight': '500',
                    'color': COLORS['text'],
                    'fontFamily': 'Inter, sans-serif',
                    'marginBottom': '0.5rem',
                    'display': 'block'
                }),
                html.Div(id="alpha-value", style={
                    'fontSize': '18px',
                    'fontWeight': '600',
                    'color': COLORS['secondary'],
                    'marginBottom': '0.5rem'
                }),
                dcc.Slider(
                    id="alpha",
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    value=0.5,
                    marks={i/10: {'label': f'{i/10:.1f}', 'style': {'fontSize': '12px'}} for i in range(0, 11, 2)},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], style={'marginBottom': '2rem'}),

            # Botón de ejecutar
            html.Button(
                "Ejecutar Recomendación",
                id="run_button",
                n_clicks=0,
                style={
                    'width': '100%',
                    'padding': '15px',
                    'backgroundColor': COLORS['primary'],
                    'color': 'white',
                    'border': 'none',
                    'borderRadius': '10px',
                    'fontSize': '16px',
                    'fontWeight': '600',
                    'fontFamily': 'Inter, sans-serif',
                    'cursor': 'pointer',
                    'transition': 'all 0.3s ease',
                    'boxShadow': '0 4px 15px rgba(46, 134, 171, 0.3)'
                }
            ),

            # Botón de descarga
            html.Div([
                html.A(
                    "Descargar Resultados CSV",
                    id="download-link",
                    download="recomendaciones.csv",
                    href="",
                    target="_blank",
                    style={
                        'display': 'none',
                        'width': '100%',
                        'padding': '12px',
                        'backgroundColor': COLORS['success'],
                        'color': 'white',
                        'textDecoration': 'none',
                        'borderRadius': '8px',
                        'fontSize': '14px',
                        'fontWeight': '500',
                        'fontFamily': 'Inter, sans-serif',
                        'textAlign': 'center',
                        'marginTop': '1rem',
                        'display': 'block'
                    }
                )
            ])
        ], style={
            'backgroundColor': COLORS['card'],
            'padding': '2rem',
            'borderRadius': '15px',
            'boxShadow': '0 4px 15px rgba(0,0,0,0.1)',
            'height': 'fit-content'
        })
    ], style={
        'width': '30%',
        'display': 'inline-block',
        'verticalAlign': 'top',
        'marginRight': '2%'
    }),

    # Panel de resultados
    html.Div([
        # Resumen ejecutivo
        html.Div(id="summary", style={'marginBottom': '2rem'}),
        
        # Gráficos
        html.Div([
            dcc.Graph(id="score_graph", style={'marginBottom': '2rem'}),
            dcc.Graph(id="comparison_graph", style={'marginBottom': '2rem'}),
            dcc.Graph(id="value_graph")
        ]),
        
        # Tabla de resultados
        html.Div(id="tabla_resultados", style={'marginTop': '2rem'})
        
    ], style={
        'width': '68%',
        'display': 'inline-block',
        'verticalAlign': 'top'
    })

], style={
    'backgroundColor': COLORS['background'],
    'minHeight': '100vh',
    'padding': '2rem',
    'fontFamily': 'Inter, sans-serif'
})

# Global para almacenar la última recomendación
latest_df = pd.DataFrame()

# Callbacks para actualizar los valores mostrados de los sliders
@app.callback(
    Output("topn-value", "children"),
    Input("topn", "value")
)
def update_topn_display(value):
    return f"Top {value} productos"

@app.callback(
    Output("alpha-value", "children"),
    Input("alpha", "value")
)
def update_alpha_display(value):
    return f"α = {value:.2f} ({'LightFM' if value > 0.5 else 'XGBoost' if value < 0.5 else 'Equilibrado'})"

# Callback principal
@app.callback(
    [Output("score_graph", "figure"),
     Output("comparison_graph", "figure"),
     Output("value_graph", "figure"),
     Output("tabla_resultados", "children"),
     Output("summary", "children"),
     Output("download-link", "href"),
     Output("download-link", "style")],
    Input("run_button", "n_clicks"),
    [State("cliente_id", "value"),
     State("topn", "value"),
     State("alpha", "value")]
)
def actualizar_dashboard(n_clicks, cliente_id, topn, alpha):
    global latest_df
    
    if n_clicks == 0 or not cliente_id:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="Ejecuta una recomendación para ver los resultados",
            showlegend=False,
            height=400
        )
        return empty_fig, empty_fig, empty_fig, html.Div(), html.Div(), "", {'display': 'none'}

    try:
        # Ejecutar recomendación
        df = recomendar_hibrido_b2b(cliente_id=cliente_id, top_n=topn, alpha=alpha)
        
        if isinstance(df, str):
            error_div = html.Div([
                html.H4("Error", style={'color': COLORS['success']}),
                html.P(df)
            ], style={
                'backgroundColor': '#fee',
                'padding': '1rem',
                'borderRadius': '8px',
                'border': f'1px solid {COLORS["success"]}'
            })
            return go.Figure(), go.Figure(), go.Figure(), error_div, html.Div(), "", {'display': 'none'}

        # Procesar datos
        df['alineación con portafolio estratégico b2b'] = pd.to_numeric(
            df.get('alineación con portafolio estratégico b2b', pd.Series(dtype='float64')),
            errors='coerce'
        )

        # Normalizar scores para comparación
        min_lfm, max_lfm = df['score_lfm'].min(), df['score_lfm'].max()
        if max_lfm != min_lfm:
            df['score_lfm_norm'] = (df['score_lfm'] - min_lfm) / (max_lfm - min_lfm)
        else:
            df['score_lfm_norm'] = 0.5

        # Crear resumen ejecutivo
        valor_total = df['valor_esperado'].sum()
        alineacion_prom = df['alineación con portafolio estratégico b2b'].mean()
        
        resumen = html.Div([
            html.H3(f"Resumen Ejecutivo - Cliente: {cliente_id}", style={
                'color': COLORS['primary'],
                'fontFamily': 'Inter, sans-serif',
                'fontWeight': '600',
                'marginBottom': '1rem'
            }),
            html.Div([
                html.Div([
                    html.H4(f"${valor_total:,.0f}", style={
                        'color': COLORS['success'],
                        'fontSize': '2rem',
                        'fontWeight': '700',
                        'margin': '0'
                    }),
                    html.P("Valor Esperado Total", style={
                        'color': COLORS['text'],
                        'margin': '0',
                        'fontSize': '0.9rem'
                    })
                ], style={
                    'textAlign': 'center',
                    'backgroundColor': '#fff',
                    'padding': '1.5rem',
                    'borderRadius': '10px',
                    'boxShadow': '0 2px 10px rgba(0,0,0,0.1)',
                    'width': '48%',
                    'display': 'inline-block',
                    'marginRight': '4%'
                }),
                html.Div([
                    html.H4(f"{alineacion_prom:.3f}", style={
                        'color': COLORS['secondary'],
                        'fontSize': '2rem',
                        'fontWeight': '700',
                        'margin': '0'
                    }),
                    html.P("Alineación Estratégica Promedio", style={
                        'color': COLORS['text'],
                        'margin': '0',
                        'fontSize': '0.9rem'
                    })
                ], style={
                    'textAlign': 'center',
                    'backgroundColor': '#fff',
                    'padding': '1.5rem',
                    'borderRadius': '10px',
                    'boxShadow': '0 2px 10px rgba(0,0,0,0.1)',
                    'width': '48%',
                    'display': 'inline-block'
                })
            ])
        ], style={
            'backgroundColor': COLORS['card'],
            'padding': '2rem',
            'borderRadius': '15px',
            'boxShadow': '0 4px 15px rgba(0,0,0,0.1)'
        })

        # Gráfico 1: Scores híbridos
        df_sorted = df.sort_values("score_hibrido")
        fig1 = px.bar(
            df_sorted,
            x="score_hibrido",
            y="producto",
            orientation='h',
            title="Score Híbrido por Producto",
            color="score_hibrido",
            color_continuous_scale="Blues"
        )
        fig1.update_layout(
            height=500,
            font=dict(family="Inter, sans-serif"),
            title_font_size=20,
            title_font_color=COLORS['primary']
        )

        # Gráfico 2: Comparación de modelos
        df_melted = df.melt(
            id_vars='producto',
            value_vars=['score_lfm_norm', 'score_xgb', 'score_hibrido'],
            var_name='modelo',
            value_name='score'
        )
        df_melted['modelo'] = df_melted['modelo'].map({
            'score_lfm_norm': 'LightFM (Normalizado)',
            'score_xgb': 'XGBoost',
            'score_hibrido': 'Híbrido'
        })
        
        fig2 = px.bar(
            df_melted,
            x='producto',
            y='score',
            color='modelo',
            barmode='group',
            title="Comparación de Scores por Modelo",
            color_discrete_map={
                'LightFM (Normalizado)': COLORS['primary'],
                'XGBoost': COLORS['secondary'],
                'Híbrido': COLORS['accent']
            }
        )
        fig2.update_layout(
            height=500,
            font=dict(family="Inter, sans-serif"),
            title_font_size=20,
            title_font_color=COLORS['primary'],
            xaxis_tickangle=-45
        )

        # Gráfico 3: Valor esperado vs Alineación
        fig3 = px.scatter(
            df,
            x='alineación con portafolio estratégico b2b',
            y='valor_esperado',
            size='score_hibrido',
            hover_data=['producto'],
            title="Valor Esperado vs Alineación Estratégica",
            color='score_hibrido',
            color_continuous_scale="Viridis"
        )
        fig3.update_layout(
            height=500,
            font=dict(family="Inter, sans-serif"),
            title_font_size=20,
            title_font_color=COLORS['primary']
        )

        # Tabla de resultados mejorada
        tabla = html.Div([
            html.H3("Resultados Detallados", style={
                'color': COLORS['primary'],
                'fontFamily': 'Inter, sans-serif',
                'fontWeight': '600',
                'marginBottom': '1rem'
            }),
            dash_table.DataTable(
                columns=[
                    {"name": "Producto", "id": "producto"},
                    {"name": "Score LightFM", "id": "score_lfm", "type": "numeric", "format": {"specifier": ".4f"}},
                    {"name": "Score XGBoost", "id": "score_xgb", "type": "numeric", "format": {"specifier": ".4f"}},
                    {"name": "Score Híbrido", "id": "score_hibrido", "type": "numeric", "format": {"specifier": ".4f"}},
                    {"name": "Precio Promedio", "id": "precio_promedio", "type": "numeric", "format": {"specifier": "$,.0f"}},
                    {"name": "Valor Esperado", "id": "valor_esperado", "type": "numeric", "format": {"specifier": "$,.0f"}},
                    {"name": "Alineación Estratégica", "id": "alineación con portafolio estratégico b2b", "type": "numeric", "format": {"specifier": ".6f"}}
                ],
                data=df.to_dict('records'),
                style_table={
                    'overflowX': 'auto',
                    'backgroundColor': COLORS['card'],
                    'borderRadius': '10px'
                },
                style_cell={
                    'textAlign': 'left',
                    'fontFamily': 'Inter, sans-serif',
                    'fontSize': '14px',
                    'padding': '12px'
                },
                style_header={
                    'backgroundColor': COLORS['primary'],
                    'color': 'white',
                    'fontWeight': '600'
                },
                style_data_conditional=[
                    {
                        'if': {'row_index': 0},
                        'backgroundColor': '#E8F4FD',
                        'color': 'black',
                    }
                ],
                page_size=15,
                sort_action="native"
            )
        ], style={
            'backgroundColor': COLORS['card'],
            'padding': '2rem',
            'borderRadius': '15px',
            'boxShadow': '0 4px 15px rgba(0,0,0,0.1)'
        })

        # Preparar descarga
        latest_df = df.copy()
        csv_string = latest_df.to_csv(index=False, encoding='utf-8')
        csv_string = "data:text/csv;charset=utf-8," + csv_string

        download_style = {
            'display': 'block',
            'width': '100%',
            'padding': '12px',
            'backgroundColor': COLORS['success'],
            'color': 'white',
            'textDecoration': 'none',
            'borderRadius': '8px',
            'fontSize': '14px',
            'fontWeight': '500',
            'fontFamily': 'Inter, sans-serif',
            'textAlign': 'center',
            'marginTop': '1rem'
        }

        return fig1, fig2, fig3, tabla, resumen, csv_string, download_style

    except Exception as e:
        error_fig = go.Figure()
        error_fig.add_annotation(
            text=f"Error: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        error_fig.update_layout(height=400)
        
        return error_fig, error_fig, error_fig, html.Div(), html.Div(), "", {'display': 'none'}


if __name__ == "__main__":
    app.run_server(debug=True, port=8050)