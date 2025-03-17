'''import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

# old code
# Main Dashboard Layout
main_dashboard_layout = dbc.Container([
    
    dbc.Row([  # Remove 'no_gutters' argument here
        # Left Division (Navigation Buttons)
        dbc.Col([
            html.H1("Main Dashboard", style={'textAlign': 'center', 'color': 'white'}),
            html.P("Welcome to the Main Dashboard.", 
                   style={'textAlign': 'center', 'color': 'white'}),

            # Navigation Buttons
            dbc.Button("Go to Fetal Health Analysis", href="/fetal_health", color="primary", className="mb-3", 
                       style={'backgroundColor': '#555', 'color': 'white', 'borderRadius': '8px', 'width': '100%'}),
            dbc.Button("Another Feature", href="/another_feature", color="secondary", className="mb-3", 
                       style={'backgroundColor': '#555', 'color': 'white', 'borderRadius': '8px', 'width': '100%'})
        ], width=3, style={'padding': '20px'}),

        # Center Division (Information and Images)
        dbc.Col([
            html.Div([
                # Images Section
                html.H3("Fetal and Maternal Health", style={'color': 'white', 'textAlign': 'center'}),
                html.Div([
                    html.Img(src="/assets/pic1.png", style={'width': '50%', 'margin': '10px'}),
                ], style={'display': 'flex', 'justifyContent': 'center', 'flexWrap': 'wrap'}),
                
                # Information/Quote Section
                html.Div([
                    html.P("Fetal Health Awareness:", style={'color': 'white', 'fontWeight': 'bold', 'textAlign': 'center'}),
                    html.P("""
                        Maternal and fetal health are intertwined. It's essential for mothers to get regular prenatal care,
                        eat nutritious food, stay active, and avoid harmful substances. Fetal health can be monitored through 
                        regular checkups and screenings.
                    """, style={'color': 'white', 'textAlign': 'center'}),
                    html.H5('"A mother is the first teacher of a child, and a healthy mother means a healthy future for the baby."', 
                            style={'color': 'white', 'textAlign': 'center', 'fontStyle': 'italic'})
                ], style={'marginTop': '20px'})
            ])
        ], width=9, style={'padding': '20px'})
    ], style={'minHeight': '100vh'})  # Ensure the row takes the full height of the screen
], fluid=True)

if __name__ == '__main__':
    app.run_server(debug=True)

# oldcode


# new code
# Main Dashboard Layout
main_dashboard_layout = dbc.Container([
    html.H1("Main Dashboard", style={'textAlign': 'center', 'color': 'pink', 'fontWeight': 'bold'}),
    html.P("Welcome to the Main Dashboard. Navigate to the Fetal Health Analysis page by clicking the button below.", 
           style={'textAlign': 'center', 'color': 'white'}),
    
    # Row for the navigation button and the central content (image and information)
    dbc.Row([
        dbc.Col(
            dbc.Button("Go to Fetal Health Analysis", href="/fetal_health", color="danger", className="mb-3", 
                       style={'backgroundColor': 'pink', 'color': 'white', 'borderRadius': '8px', 'width': '100%'}),
            width=6
        ),
        dbc.Col([
            html.Div([
                html.H3("Fetal and Maternal Health Awareness", style={'color': 'pink', 'textAlign': 'center'}),
                html.Img(src="/assets/pic1.png", style={'width': '50%', 'margin': '10px', 'borderRadius': '8px'}),
                html.P("Fetal health and maternal care are crucial aspects of prenatal well-being. "
                       "Ensure regular check-ups and a healthy lifestyle during pregnancy. "
                       "Your health and the health of your baby matter!", 
                       style={'color': 'white', 'textAlign': 'center', 'fontSize': '16px'}),
            ])
        ], width=6)
    ], justify="center"),
], style={'backgroundColor': 'black', 'padding': '20px', 'height': '100vh'})

if __name__ == '__main__':
    app.run_server(debug=True)
'''
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
# Add this to your app layout to enable routing
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),  # Add this component for URL routing
    html.Div(id='page-content')  # This is where page content will be rendered
])

# Main Dashboard Layout
main_dashboard_layout = dbc.Container([
    html.H1("IN UTERO EXAMINATION", style={'textAlign': 'center', 'color': 'pink', 'fontWeight': 'bold', 'paddingTop': '20px'}),  # Heading

    html.P("Welcome to the Main Dashboard", 
           style={'textAlign': 'center', 'color': 'white', 'fontSize': '18px'}),

    # Row for the navigation button and the central content (image and information)
    dbc.Row([
        # Left Division (Navigation Button)
        dbc.Col(
            [
            dbc.Button("Go to Fetal Health Analysis", href="/fetal_health", color="danger", className="mb-3", 
                       style={'backgroundColor': 'pink', 'color': 'black', 'borderRadius': '8px', 'width': '100%'}),
            dbc.Button("Go to Maternal Health Analysis", href="/maternal_health", color="danger", className="mb-3", 
                           style={'backgroundColor': 'pink', 'color': 'black', 'borderRadius': '8px', 'width': '100%'}),
            dbc.Button("Go to Correlation Analysis", id = "correlation-analysis-btn", color = "danger", href = '/correlation_analysis', className = "mb-3",
            style={'backgroundColor': 'pink', 'color': 'black', 'borderRadius': '8px','width': '100%'}),
            ],
            width=10,  # Take up 3/12 of the width
            style={'padding': '17px'}
        ),
        

        # Center Division (Information and Image)
        dbc.Col([
            html.Div([
                html.H3("Fetal and Maternal Health ", style={'color': 'pink', 'textAlign': 'center'}),

                # Image Section
                html.Img(src="/assets/pic1.png", style={'width': '50%', 'margin': '10px', 'borderRadius': '20px','marginLeft': '220px'}),

                # Information Section
                html.P("Fetal health and maternal care are crucial aspects of prenatal well-being. "
                       "Ensure regular check-ups and a healthy lifestyle during pregnancy. "
                       "Your health and the health of your baby matter!", 
                       style={'color': 'pink', 'textAlign': 'center', 'fontSize': '16px'}),

                html.H5('"A mother is the first teacher of a child, and a healthy mother means a healthy future for the baby."', 
                        style={'color': 'pink', 'textAlign': 'center', 'fontStyle': 'italic'})
            ])
        ], width=8, style={'padding': '20px'})
    ], justify="center", align="right"),  # Ensures both sections are centered vertically and horizontally
], style={'backgroundColor': 'black', 'padding': '20px', 'height': '100vh'})

if __name__ == '__main__':
    app.run_server(debug=True)
