import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

# Initialize Dash app (this will be imported in app.py)

# Hardcoded admin credentials (For now)
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "password"

# Layout for Login Page
login_layout = dbc.Container([
    html.H2("Admin Login", style={'textAlign': 'center', 'color': 'pink'}),
    
    dbc.Row([
        dbc.Col([
            html.Label("Username", style={'color': 'white'}),
            dcc.Input(id="username", type="text", placeholder="Enter Username", style={'width': '100%'}),
            html.Br(),
            html.Label("Password", style={'color': 'white'}),
            dcc.Input(id="password", type="password", placeholder="Enter Password", style={'width': '100%'}),
            html.Br(),
            dbc.Button("Login", id="login-btn", color="primary", style={'marginTop': '10px', 'width': '100%'}),
            html.Div(id="login-output", style={'color': 'red', 'textAlign': 'center', 'marginTop': '10px'})
        ], width=4)
    ], justify="center")
], style={'height': '100vh', 'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center'})




