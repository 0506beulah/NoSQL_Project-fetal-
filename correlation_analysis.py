import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import plotly.express as px


# Load Dataset for Fetal Health Analysis (f_health.csv)
df_fetal = pd.read_csv("f_health.csv")

# Define the correlation analysis layout
correlation_analysis_layout = dbc.Container([
    html.H1("Correlation Analysis for Fetal Health", style={'textAlign': 'center', 'color': 'pink'}),
    html.P("Enter Fetal Movement and Uterine Contractions to analyze the correlation.", style={'textAlign': 'center', 'color': 'white'}),

    # Input fields for fetal movement and uterine contractions
    dbc.Row([
        dbc.Col([
            html.Label("Fetal Movement", style={'color': 'pink', 'fontWeight': 'bold'}),
            dcc.Input(id='fetal-movement', type='number', placeholder="Enter Fetal Movement", 
                      style={'margin': '5px', 'width': '100%', 'padding': '10px', 'borderRadius': '8px', 'backgroundColor': '#333', 'color': 'pink'})
        ], width=6),
        dbc.Col([
            html.Label("Uterine Contractions", style={'color': 'pink', 'fontWeight': 'bold'}),
            dcc.Input(id='uterine-contractions', type='number', placeholder="Enter Uterine Contractions", 
                      style={'margin': '5px', 'width': '100%', 'padding': '10px', 'borderRadius': '8px', 'backgroundColor': '#333', 'color': 'pink'})
        ], width=6)
    ], justify="center"),

    # Button to trigger analysis
    dbc.Button("Analyze", id="analyze-btn", color="primary", className="mb-3", 
               style={'backgroundColor': '#555', 'color': 'white', 'borderRadius': '8px'}),
    
    # Output area for prediction
    html.Div(id="output-correlation", style={'color': 'white', 'textAlign': 'center', 'fontSize': '20px', 'marginTop': '20px'}),
    
    # Back to Home Button (Updated)
    # Back to Home Button at top-left corner
    dbc.Button("Back to Home", href="/", color="primary", className="mb-3", 
               style={'backgroundColor': 'black', 'color': 'pink', 'borderRadius': '5px', 'margin-bottom': '200px'}),

    # Correlation Heatmap Placeholder
    dcc.Graph(id='correlation-heatmap', style={'height': '400px'}),
], style={'backgroundColor': '#111', 'height': '100vh', 'display': 'flex', 'flexDirection': 'column'})

# Output Div    