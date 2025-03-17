import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from dash.dependencies import Input, Output
from sklearn.metrics import classification_report

# Load Dataset for Maternal Health Analysis (replace with your dataset)
df_maternal = pd.read_csv("m_health.csv")

# Print column names to ensure 'maternal_health' exists
print("Columns in maternal health dataset:", df_maternal.columns)

# Drop unwanted columns (adapt this as per your dataset)
drop_columns_maternal = ["unwanted_column1", "unwanted_column2"]
df_maternal.drop(columns=drop_columns_maternal, inplace=True, errors='ignore')

# Handle missing values
df_maternal.dropna(inplace=True)

# Maternal Health Layout
maternal_health_layout = dbc.Container([
    html.H1("Maternal Health Analysis", style={'textAlign': 'center', 'color': 'white'}),
    html.P("Enter the parameters to analyze maternal health.", style={'textAlign': 'center', 'color': 'white'}),

    # Layout for inputs and buttons
    dbc.Row(
        [
            dbc.Col(
                [
                    html.Label(col, style={'color': 'pink', 'fontWeight': 'bold'}),
                    dcc.Input(id=col, type='number', placeholder=f"Enter {col}",
                              style={'margin': '17px', 'width': '100%', 'padding': '17px',
                                     'borderRadius': '8px', 'backgroundColor': '#333', 'color': 'white'})
                ],
                width=6
            ) for col in df_maternal.drop(columns='maternal_health').columns  # Exclude the target column
        ],
        justify="center",
        style={'flex': '1', 'display': 'flex', 'alignItems': 'center', 'height': '100%'}
    ),

    # Back to Home Button at top-left corner
    dbc.Button("Back to Home", href="/", color="primary", className="mb-3", 
               style={'backgroundColor': '#555', 'color': 'white', 'borderRadius': '8px', 'position': 'absolute', 'top': '10px', 'left': '10px'}),

    html.Br(),
    dbc.Button("Analyze", id="analyze-btn-maternal", color="primary", className="mb-3", 
               style={'backgroundColor': '#555', 'color': 'white', 'borderRadius': '8px'}),

    html.Br(),
    html.Div(id="output-maternal", style={'color': 'white', 'textAlign': 'center', 'fontSize': '20px', 'marginTop': '20px'}), 
], style={'backgroundColor': '#111', 'height': '100vh', 'display': 'flex', 'flexDirection': 'column'})

