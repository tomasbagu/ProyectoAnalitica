"""
Dashboard Interactivo - Análisis de Rendimiento de Jugadores de Pádel
=====================================================================

Este dashboard lee los datos preprocesados del archivo CSV generado por el notebook
y presenta la información de manera visual e interactiva.

IMPORTANTE: Antes de ejecutar este dashboard, debes correr todo el notebook 
Proyecto.ipynb para generar el archivo 'datos_dashboard.csv'

Ejecutar: python dashboard.py
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import warnings
import os
from dotenv import load_dotenv
import google.generativeai as genai

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURACIÓN DE GEMINI AI
# =============================================================================

# Cargar variables de entorno desde .env
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("⚠️ GEMINI_API_KEY no encontrada. Crea un archivo .env con la clave.")

def generar_consejo_ia(datos_jugador, nombre_jugador):
    """
    Genera un consejo personalizado usando Gemini AI basado en los datos del jugador.
    """
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Extraer métricas del jugador
        velocidad_prom = datos_jugador['player_speed_mps_mean'].mean() if 'player_speed_mps_mean' in datos_jugador.columns else 0
        aceleracion_prom = datos_jugador['player_acceleration_mps2_mean'].mean() if 'player_acceleration_mps2_mean' in datos_jugador.columns else 0
        desplazamiento_total = datos_jugador['player_displacement_m_sum'].sum() if 'player_displacement_m_sum' in datos_jugador.columns else 0
        nivel_rendimiento = datos_jugador['nivel_rendimiento'].iloc[-1] if 'nivel_rendimiento' in datos_jugador.columns else "N/A"
        estado_declarado = datos_jugador['ESTADO_FISICO_first'].iloc[-1] if 'ESTADO_FISICO_first' in datos_jugador.columns else "N/A"
        evaluacion = datos_jugador['evaluacion'].iloc[-1] if 'evaluacion' in datos_jugador.columns else "N/A"
        edad = datos_jugador['EDAD_first'].iloc[0] if 'EDAD_first' in datos_jugador.columns else "N/A"
        nivel_padel = datos_jugador['NIVEL_ACTUAL_PADEL_first'].iloc[0] if 'NIVEL_ACTUAL_PADEL_first' in datos_jugador.columns else "N/A"
        num_partidos = len(datos_jugador)
        
        prompt = f"""
        Eres un entrenador experto de pádel y preparador físico. Analiza los siguientes datos de un jugador y genera consejos personalizados específicos para mejorar su rendimiento.

        DATOS DEL JUGADOR:
        - Nombre: {nombre_jugador}
        - Edad: {edad} años
        - Nivel de pádel declarado: {nivel_padel}
        - Estado físico declarado: {estado_declarado}
        - Nivel de rendimiento real (basado en análisis de video): {nivel_rendimiento}
        - Evaluación: {evaluacion}
        - Partidos analizados: {num_partidos}
        
        MÉTRICAS DE RENDIMIENTO (promedios por partido):
        - Velocidad promedio: {velocidad_prom:.2f} m/s
        - Aceleración promedio: {aceleracion_prom:.2f} m/s²
        - Desplazamiento total acumulado: {desplazamiento_total:.1f} metros
        
        CONTEXTO:
        - Si la evaluación es "Sobreestimó": el jugador cree estar en mejor forma de lo que realmente muestra
        - Si la evaluación es "Subestimó": el jugador tiene mejor rendimiento del que cree
        - Si la evaluación es "Declaró correctamente": hay coherencia entre percepción y realidad
        
        Genera un consejo personalizado de máximo 4-5 oraciones que incluya:
        1. Una observación sobre su rendimiento actual
        2. Un ejercicio o rutina específica para mejorar
        3. Un consejo táctico para pádel
        
        Responde en español y de forma motivadora pero realista.
        """
        
        response = model.generate_content(prompt)
        return response.text
    
    except Exception as e:
        return f"⚠️ No se pudo generar el consejo con IA: {str(e)}"

# =============================================================================
# CARGA DE DATOS DESDE CSV
# =============================================================================

def cargar_datos_csv():
    """Carga los datos preprocesados desde el CSV generado por el notebook."""
    csv_path = "datos_dashboard.csv"
    
    if not os.path.exists(csv_path):
        print("="*60)
        print("❌ ERROR: No se encontró el archivo 'datos_dashboard.csv'")
        print("="*60)
        print("\n📋 Para generar este archivo:")
        print("   1. Abre el notebook 'Proyecto.ipynb'")
        print("   2. Ejecuta TODAS las celdas del notebook")
        print("   3. La última celda generará el archivo CSV")
        print("   4. Vuelve a ejecutar este dashboard")
        print("="*60)
        raise FileNotFoundError(
            "El archivo 'datos_dashboard.csv' no existe. "
            "Ejecuta primero el notebook Proyecto.ipynb completo."
        )
    
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    print(f"✅ Datos cargados desde '{csv_path}'")
    print(f"📊 Registros: {len(df)} | Jugadores: {df['player_name_clean'].nunique()}")
    
    return df

# =============================================================================
# CARGAR DATOS
# =============================================================================

print("\n" + "="*60)
print("🎾 CARGANDO DASHBOARD DE PÁDEL")
print("="*60)

try:
    matches = cargar_datos_csv()
    jugadores_lista = sorted(matches['player_name_clean'].unique().tolist())
    datos_cargados = True
except FileNotFoundError as e:
    print(str(e))
    matches = pd.DataFrame()
    jugadores_lista = []
    datos_cargados = False

# =============================================================================
# CONFIGURACIÓN DEL DASHBOARD
# =============================================================================

app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "Dashboard Pádel Analytics"

# CSS personalizado para el dropdown
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .Select-value-label, .Select-placeholder {
                color: white !important;
            }
            .Select-control {
                background-color: #2a2a4a !important;
            }
            .Select-menu-outer {
                background-color: #2a2a4a !important;
            }
            .VirtualizedSelectOption {
                color: white !important;
            }
            #selector-jugador .Select-value-label {
                color: white !important;
            }
            #selector-jugador input {
                color: white !important;
            }
            .Select-input > input {
                color: white !important;
            }
            .Select--single > .Select-control .Select-value {
                color: white !important;
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

# Colores personalizados
COLORS = {
    'background': '#1a1a2e',
    'card': '#16213e',
    'primary': '#0f3460',
    'accent': '#e94560',
    'text': '#ffffff',
    'success': '#4ecca3',
    'warning': '#ffc107',
    'danger': '#ff6b6b'
}

# =============================================================================
# LAYOUT DEL DASHBOARD
# =============================================================================

if not datos_cargados:
    # Layout de error si no hay datos
    app.layout = dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("❌ Error: Datos no disponibles", className="text-center mt-5"),
                html.Hr(),
                html.P("No se encontró el archivo 'datos_dashboard.csv'", className="text-center lead"),
                html.Div([
                    html.H4("📋 Pasos para solucionar:", className="mt-4"),
                    html.Ol([
                        html.Li("Abre el notebook 'Proyecto.ipynb' en VS Code"),
                        html.Li("Ejecuta TODAS las celdas del notebook (Run All)"),
                        html.Li("Verifica que se generó el archivo 'datos_dashboard.csv'"),
                        html.Li("Vuelve a ejecutar este dashboard"),
                    ], className="lead")
                ], className="mt-4")
            ], width=8, className="mx-auto")
        ])
    ], fluid=True, style={'backgroundColor': COLORS['background'], 'minHeight': '100vh', 'paddingTop': '50px'})
else:
    app.layout = dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H1("🎾 Dashboard de Análisis de Rendimiento - Pádel", 
                       className="text-center mb-2", style={'color': COLORS['accent']}),
                html.P("Análisis de métricas físicas, clustering y comparación rendimiento vs estado declarado",
                      className="text-center text-muted")
            ])
        ], className="mb-4 mt-3"),
        
        # Selector de jugador
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("🏃 Seleccionar Jugador", style={'backgroundColor': COLORS['primary']}),
                    dbc.CardBody([
                        dcc.Dropdown(
                            id='selector-jugador',
                            options=[{'label': j, 'value': j} for j in jugadores_lista],
                            value=jugadores_lista[0] if jugadores_lista else None,
                            placeholder="Selecciona un jugador...",
                            style={'backgroundColor': '#2a2a4a', 'color': '#ffffff'},
                            className='dropdown-white-text'
                        )
                    ])
                ], style={'backgroundColor': COLORS['card']})
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("📊 Resumen General", style={'backgroundColor': COLORS['primary']}),
                    dbc.CardBody([
                        html.Div(id='resumen-general')
                    ])
                ], style={'backgroundColor': COLORS['card']})
            ], width=6)
        ], className="mb-4"),
        
        # Fila de KPIs del jugador
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Nivel de Rendimiento", className="text-muted"),
                        html.H3(id='kpi-rendimiento', className="text-center")
                    ])
                ], style={'backgroundColor': COLORS['card']})
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Estado Declarado", className="text-muted"),
                        html.H3(id='kpi-estado', className="text-center")
                    ])
                ], style={'backgroundColor': COLORS['card']})
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Evaluación", className="text-muted"),
                        html.H3(id='kpi-evaluacion', className="text-center")
                    ])
                ], style={'backgroundColor': COLORS['card']})
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Partidos Jugados", className="text-muted"),
                        html.H3(id='kpi-partidos', className="text-center")
                    ])
                ], style={'backgroundColor': COLORS['card']})
            ], width=3),
        ], className="mb-4"),
        
        # Recomendación personalizada
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("💡 Recomendación Personalizada", style={'backgroundColor': COLORS['primary']}),
                    dbc.CardBody([
                        html.P(id='recomendacion-texto', className="lead")
                    ])
                ], style={'backgroundColor': COLORS['card']})
            ])
        ], className="mb-4"),
        
        # Consejo IA con Gemini
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.Span("🤖 Consejo Personalizado con IA ", style={'marginRight': '10px'}),
                        html.Span("powered by Gemini", style={'fontSize': '12px', 'opacity': '0.7'})
                    ], style={'backgroundColor': COLORS['primary']}),
                    dbc.CardBody([
                        dbc.Button(
                            "✨ Generar Consejo con IA",
                            id='btn-generar-consejo',
                            color="danger",
                            className="mb-3",
                            style={'width': '100%'}
                        ),
                        dcc.Loading(
                            id="loading-consejo",
                            type="circle",
                            children=[
                                html.Div(
                                    id='consejo-ia-texto',
                                    className="lead",
                                    style={
                                        'whiteSpace': 'pre-wrap',
                                        'backgroundColor': 'rgba(233, 69, 96, 0.1)',
                                        'padding': '15px',
                                        'borderRadius': '10px',
                                        'borderLeft': f'4px solid {COLORS["accent"]}',
                                        'minHeight': '100px'
                                    }
                                )
                            ]
                        )
                    ])
                ], style={'backgroundColor': COLORS['card']})
            ])
        ], className="mb-4"),
        
        # Gráficos principales
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("🎯 Clustering UMAP - Posición del Jugador", 
                                  style={'backgroundColor': COLORS['primary']}),
                    dbc.CardBody([
                        dcc.Graph(id='grafico-umap')
                    ])
                ], style={'backgroundColor': COLORS['card']})
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("📈 Evolución del Rendimiento", 
                                  style={'backgroundColor': COLORS['primary']}),
                    dbc.CardBody([
                        dcc.Graph(id='grafico-evolucion')
                    ])
                ], style={'backgroundColor': COLORS['card']})
            ], width=6)
        ], className="mb-4"),
        
        # Métricas detalladas
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("📊 Métricas Físicas Promedio", 
                                  style={'backgroundColor': COLORS['primary']}),
                    dbc.CardBody([
                        dcc.Graph(id='grafico-metricas')
                    ])
                ], style={'backgroundColor': COLORS['card']})
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("🏆 Comparación con Promedios Generales", 
                                  style={'backgroundColor': COLORS['primary']}),
                    dbc.CardBody([
                        dcc.Graph(id='grafico-radar')
                    ])
                ], style={'backgroundColor': COLORS['card']})
            ], width=6)
        ], className="mb-4"),
        
        # Información del perfil
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("👤 Perfil del Jugador", style={'backgroundColor': COLORS['primary']}),
                    dbc.CardBody([
                        html.Div(id='perfil-jugador')
                    ])
                ], style={'backgroundColor': COLORS['card']})
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("📋 Historial de Partidos", style={'backgroundColor': COLORS['primary']}),
                    dbc.CardBody([
                        html.Div(id='historial-partidos', style={'maxHeight': '300px', 'overflowY': 'auto'})
                    ])
                ], style={'backgroundColor': COLORS['card']})
            ], width=8)
        ], className="mb-4"),
        
        # Distribución general
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("📊 Distribución de Evaluaciones - Todos los Jugadores", 
                                  style={'backgroundColor': COLORS['primary']}),
                    dbc.CardBody([
                        dcc.Graph(id='grafico-distribucion')
                    ])
                ], style={'backgroundColor': COLORS['card']})
            ])
        ], className="mb-4"),
        
        # Footer
        dbc.Row([
            dbc.Col([
                html.Hr(),
                html.P("Proyecto Final - Analítica de Datos | Universidad de La Sabana | 2024", 
                      className="text-center text-muted")
            ])
        ])
        
    ], fluid=True, style={'backgroundColor': COLORS['background'], 'minHeight': '100vh'})

# =============================================================================
# CALLBACKS
# =============================================================================

if datos_cargados:
    @callback(
        [Output('resumen-general', 'children'),
         Output('kpi-rendimiento', 'children'),
         Output('kpi-estado', 'children'),
         Output('kpi-evaluacion', 'children'),
         Output('kpi-partidos', 'children'),
         Output('recomendacion-texto', 'children'),
         Output('grafico-umap', 'figure'),
         Output('grafico-evolucion', 'figure'),
         Output('grafico-metricas', 'figure'),
         Output('grafico-radar', 'figure'),
         Output('perfil-jugador', 'children'),
         Output('historial-partidos', 'children'),
         Output('grafico-distribucion', 'figure')],
        [Input('selector-jugador', 'value')]
    )
    def actualizar_dashboard(jugador_seleccionado):
        if not jugador_seleccionado:
            return ["Selecciona un jugador"] + ["--"] * 4 + [""] + [go.Figure()] * 5 + ["", "", go.Figure()]
        
        # Filtrar datos del jugador
        datos_jugador = matches[matches['player_name_clean'] == jugador_seleccionado].copy()
        datos_jugador = datos_jugador.sort_values('partido_num')
        
        if len(datos_jugador) == 0:
            return ["Sin datos"] + ["--"] * 4 + [""] + [go.Figure()] * 5 + ["", "", go.Figure()]
        
        # KPIs
        nivel_rendimiento = datos_jugador['nivel_rendimiento'].iloc[-1] if 'nivel_rendimiento' in datos_jugador.columns else "N/A"
        estado_declarado = datos_jugador['ESTADO_FISICO_first'].iloc[-1] if 'ESTADO_FISICO_first' in datos_jugador.columns else "N/A"
        evaluacion = datos_jugador['evaluacion'].iloc[-1] if 'evaluacion' in datos_jugador.columns else "N/A"
        recomendacion = datos_jugador['recomendacion'].iloc[-1] if 'recomendacion' in datos_jugador.columns else "Sin recomendación"
        num_partidos = len(datos_jugador)
        
        # Colores para evaluación
        color_eval = COLORS['success'] if evaluacion == "Declaró correctamente" else (
            COLORS['warning'] if evaluacion == "Sobreestimó" else COLORS['danger']
        )
        
        # Resumen general
        resumen = html.Div([
            html.P(f"Total jugadores: {len(jugadores_lista)}"),
            html.P(f"Total registros: {len(matches)}")
        ])
        
        # ===================== GRÁFICO UMAP =====================
        fig_umap = go.Figure()
        
        # Todos los jugadores (fondo)
        if 'UMAP1' in matches.columns and 'UMAP2' in matches.columns:
            fig_umap.add_trace(go.Scatter(
                x=matches['UMAP1'],
                y=matches['UMAP2'],
                mode='markers',
                marker=dict(
                    size=10,
                    color=matches['cluster_umap'] if 'cluster_umap' in matches.columns else 'blue',
                    colorscale='Viridis',
                    opacity=0.4
                ),
                name='Otros jugadores',
                hoverinfo='skip'
            ))
            
            # Jugador seleccionado (destacado)
            fig_umap.add_trace(go.Scatter(
                x=datos_jugador['UMAP1'],
                y=datos_jugador['UMAP2'],
                mode='markers+text',
                marker=dict(
                    size=18,
                    color=COLORS['accent'],
                    symbol='star',
                    line=dict(width=2, color='white')
                ),
                text=[f"P{i+1}" for i in range(len(datos_jugador))],
                textposition='top center',
                name=jugador_seleccionado
            ))
        
        fig_umap.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            showlegend=True,
            legend=dict(x=0, y=1)
        )
        
        # ===================== GRÁFICO EVOLUCIÓN =====================
        fig_evolucion = go.Figure()
        
        if 'nivel_num' in datos_jugador.columns:
            fig_evolucion.add_trace(go.Scatter(
                x=datos_jugador['partido_num'],
                y=datos_jugador['nivel_num'],
                mode='lines+markers',
                line=dict(color=COLORS['accent'], width=3),
                marker=dict(size=12, symbol='circle'),
                name='Rendimiento'
            ))
        
        fig_evolucion.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Número de Partido",
            yaxis_title="Nivel de Rendimiento",
            yaxis=dict(
                tickmode='array',
                tickvals=[0, 1, 2],
                ticktext=['Alto', 'Bajo', 'Medio']
            )
        )
        
        # ===================== GRÁFICO MÉTRICAS =====================
        metricas_cols = [c for c in datos_jugador.columns if any(x in c for x in ['speed', 'acceleration', 'displacement', 'distance']) and datos_jugador[c].dtype in ['float64', 'int64']]
        metricas_mostrar = metricas_cols[:5]  # Máximo 5 métricas
        
        if metricas_mostrar:
            valores_jugador = datos_jugador[metricas_mostrar].mean().values
            nombres_metricas = [m.replace('_mean', '').replace('_sum', '').replace('_first', '').replace('_', ' ').title()[:20] 
                              for m in metricas_mostrar]
            
            fig_metricas = go.Figure(go.Bar(
                x=nombres_metricas,
                y=valores_jugador,
                marker_color=COLORS['accent']
            ))
        else:
            fig_metricas = go.Figure()
        
        fig_metricas.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Métrica",
            yaxis_title="Valor"
        )
        
        # ===================== GRÁFICO RADAR =====================
        fig_radar = go.Figure()
        
        if metricas_mostrar:
            valores_general = matches[metricas_mostrar].mean().values
            valores_jugador_prom = datos_jugador[metricas_mostrar].mean().values
            
            # Normalizar para radar (evitar división por cero)
            valores_jugador_norm = np.where(valores_general != 0, 
                                            valores_jugador_prom / valores_general * 100, 
                                            100)
            
            fig_radar.add_trace(go.Scatterpolar(
                r=valores_jugador_norm,
                theta=nombres_metricas,
                fill='toself',
                name=jugador_seleccionado,
                line_color=COLORS['accent']
            ))
            
            fig_radar.add_trace(go.Scatterpolar(
                r=[100] * len(nombres_metricas),
                theta=nombres_metricas,
                fill='toself',
                name='Promedio general',
                line_color='rgba(255,255,255,0.3)'
            ))
        
        fig_radar.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            polar=dict(bgcolor='rgba(0,0,0,0)')
        )
        
        # ===================== PERFIL DEL JUGADOR =====================
        edad = datos_jugador['EDAD_first'].iloc[0] if 'EDAD_first' in datos_jugador.columns else "N/A"
        estatura = datos_jugador['ESTATURA_first'].iloc[0] if 'ESTATURA_first' in datos_jugador.columns else "N/A"
        nivel = datos_jugador['NIVEL_ACTUAL_PADEL_first'].iloc[0] if 'NIVEL_ACTUAL_PADEL_first' in datos_jugador.columns else "N/A"
        frecuencia = datos_jugador['FRECUENCIA_DEPORTE_first'].iloc[0] if 'FRECUENCIA_DEPORTE_first' in datos_jugador.columns else "N/A"
        
        perfil = html.Div([
            html.H5(jugador_seleccionado, style={'color': '#ffffff'}),
            html.Hr(),
            html.P(f"📅 Edad: {edad} años"),
            html.P(f"📏 Estatura: {estatura} cm"),
            html.P(f"🎾 Nivel: {nivel}"),
            html.P(f"📆 Frecuencia: {frecuencia}"),
        ])
        
        # ===================== HISTORIAL DE PARTIDOS =====================
        historial = dbc.Table([
            html.Thead(html.Tr([
                html.Th("Partido"),
                html.Th("Rendimiento"),
                html.Th("Estado Declarado"),
                html.Th("Evaluación")
            ])),
            html.Tbody([
                html.Tr([
                    html.Td(row['partido']),
                    html.Td(row['nivel_rendimiento'] if 'nivel_rendimiento' in row else "N/A"),
                    html.Td(row['ESTADO_FISICO_first'] if 'ESTADO_FISICO_first' in row else "N/A"),
                    html.Td(row['evaluacion'] if 'evaluacion' in row else "N/A")
                ]) for _, row in datos_jugador.iterrows()
            ])
        ], bordered=True, hover=True, responsive=True, striped=True, 
           className="table-dark")
        
        # ===================== DISTRIBUCIÓN GENERAL =====================
        if 'evaluacion' in matches.columns:
            dist_eval = matches['evaluacion'].value_counts()
            colors_pie = []
            for label in dist_eval.index:
                if label == "Declaró correctamente":
                    colors_pie.append(COLORS['success'])
                elif label == "Sobreestimó":
                    colors_pie.append(COLORS['warning'])
                else:
                    colors_pie.append(COLORS['danger'])
            
            fig_dist = go.Figure(go.Pie(
                labels=dist_eval.index,
                values=dist_eval.values,
                marker_colors=colors_pie,
                hole=0.4
            ))
        else:
            fig_dist = go.Figure()
        
        fig_dist.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        # Formatear KPIs con colores
        kpi_rendimiento = html.Span(nivel_rendimiento, style={
            'color': COLORS['success'] if 'Alto' in str(nivel_rendimiento) else (
                COLORS['warning'] if 'Medio' in str(nivel_rendimiento) else COLORS['danger']
            )
        })
        kpi_estado = html.Span(estado_declarado, style={'color': COLORS['text']})
        kpi_evaluacion = html.Span(evaluacion, style={'color': color_eval})
        kpi_partidos = html.Span(str(num_partidos), style={'color': COLORS['accent']})
        
        return (resumen, kpi_rendimiento, kpi_estado, kpi_evaluacion, kpi_partidos,
                recomendacion, fig_umap, fig_evolucion, fig_metricas, fig_radar,
                perfil, historial, fig_dist)
    
    @callback(
        Output('consejo-ia-texto', 'children'),
        [Input('btn-generar-consejo', 'n_clicks')],
        [State('selector-jugador', 'value')],
        prevent_initial_call=True
    )
    def generar_consejo_callback(n_clicks, jugador_seleccionado):
        if not jugador_seleccionado:
            return "Selecciona un jugador primero."
        
        datos_jugador = matches[matches['player_name_clean'] == jugador_seleccionado].copy()
        
        if len(datos_jugador) == 0:
            return "No hay datos disponibles para este jugador."
        
        consejo = generar_consejo_ia(datos_jugador, jugador_seleccionado)
        return consejo

# =============================================================================
# EJECUCIÓN
# =============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🎾 DASHBOARD DE ANÁLISIS DE PÁDEL")
    print("="*60)
    if datos_cargados:
        print(f"📊 Jugadores cargados: {len(jugadores_lista)}")
        print(f"📋 Registros totales: {len(matches)}")
    else:
        print("⚠️  DATOS NO DISPONIBLES - Ejecuta el notebook primero")
    print("="*60)
    print("\n🚀 Iniciando servidor...")
    print("📍 Abre tu navegador en: http://127.0.0.1:8050")
    print("\n(Presiona Ctrl+C para detener el servidor)\n")
    
    app.run(debug=True, host='127.0.0.1', port=8050)
