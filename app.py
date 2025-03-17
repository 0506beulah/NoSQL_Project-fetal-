import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash import Dash, dcc, html, callback, Input, Output, ctx
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from main_dashboard import main_dashboard_layout
from fetal_health import fetal_health_layout
from maternal_health import maternal_health_layout
from correlation_analysis import correlation_analysis_layout
import plotly.graph_objects as go


# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY], suppress_callback_exceptions=True)
server = app.server

# Load Dataset for Fetal Health Analysis
df = pd.read_csv("f_health.csv")

# Drop unwanted columns
drop_columns = ["histogram_min", "histogram_max", "histogram_number_of_peaks", "histogram_number_of_zeroes", 
                "histogram_mode", "histogram_mean", "histogram_median", "histogram_variance", "histogram_tendency"]
df.drop(columns=drop_columns, inplace=True, errors='ignore')

# Check for missing values and handle them
df.dropna(inplace=True)

# Convert categorical columns (if any)
if df["fetal_health"].dtype == "object":
    label_encoder = LabelEncoder()
    df["fetal_health"] = label_encoder.fit_transform(df["fetal_health"])

# Split features and target variable
X = df.drop(columns=['fetal_health'])
y = df['fetal_health']

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

# Evaluate model
print(classification_report(y_test, model.predict(X_test)))

# Layout of the app with URL routing
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),  # URL management for navigation
    html.Div(id='page-content')  # Placeholder for the page content
])

# Fetal Health Layout with analysis functionality
fetal_health_page_layout = dbc.Container([
    html.H1("Fetal Health Analysis", style={'textAlign': 'center', 'color': 'pink'}),
    html.P("Enter the parameters to analyze fetal health.", style={'textAlign': 'center', 'color': 'white'}),

    # Layout for inputs and buttons
    dbc.Row(
        [
            dbc.Col(
                [
                    html.Label(col, style={'color': 'pink', 'fontWeight': 'bold'}),
                    dcc.Input(id=col, type='number', placeholder=f"Enter {col}",
                              style={'margin': 'px', 'width': '100%', 'padding': '10px',
                                     'borderRadius': '17px', 'backgroundColor': '#333', 'color': 'pink'})
                ],
                width=6
            ) for col in X.columns
        ],
        justify="center",
        style={'flex': '1', 'display': 'flex', 'alignItems': 'center', 'height': '100%'}
    ),

    # Back to Home Button at top-left corner
    dbc.Button("Back to Home", href="/", color="primary", className="mb-3", 
               style={'backgroundColor': '#555', 'color': 'white', 'borderRadius': '3px', 'position': 'absolute', 'top': '10px', 'left': '10px'}),

    html.Br(),
    dbc.Button("Analyze", id="analyze-btn", color="primary", className="mb-3", 
               style={'backgroundColor': '#555', 'color': 'white', 'borderRadius': '8px'}),

    html.Br(),
    html.Div(id="output", style={'color': 'white', 'textAlign': 'center', 'fontSize': '20px', 'marginTop': '20px'}),
], style={'backgroundColor': '#111', 'height': '100vh', 'display': 'flex', 'flexDirection': 'column'})

# Load Dataset for Maternal Health Analysis (replace with your dataset)
df_maternal = pd.read_csv("m_health.csv")

# Drop unwanted columns (adapt this as per your dataset)
drop_columns_maternal = ["unwanted_column1", "unwanted_column2"]
df_maternal.drop(columns=drop_columns_maternal, inplace=True, errors='ignore')

# Handle missing values
df_maternal.dropna(inplace=True)

# Convert categorical columns (if any)
if df_maternal["maternal_health"].dtype == "object":
    label_encoder = LabelEncoder()
    df_maternal["maternal_health"] = label_encoder.fit_transform(df_maternal["maternal_health"])

# Split features and target variable
X_maternal = df_maternal.drop(columns=['maternal_health'])
y_maternal = df_maternal['maternal_health']

# Scale the data
scaler_maternal = StandardScaler()
X_scaled_maternal = scaler_maternal.fit_transform(X_maternal)

# Split into training and test sets
X_train_maternal, X_test_maternal, y_train_maternal, y_test_maternal = train_test_split(X_scaled_maternal, y_maternal, test_size=0.2, random_state=42)

# Train the model
model_maternal = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
model_maternal.fit(X_train_maternal, y_train_maternal)

# Evaluate model (Optional)
print("Maternal Health Classification Report:\n", classification_report(y_test_maternal, model_maternal.predict(X_test_maternal)))

# Maternal Health Layout
maternal_health_page_layout = dbc.Container([
    html.H1("Maternal Health Analysis", style={'textAlign': 'center', 'color': 'pink'}),
    html.P("Enter the parameters to analyze maternal health.", style={'textAlign': 'center', 'color': 'white'}),

    # Layout for inputs and buttons
    dbc.Row(
        [
            dbc.Col(
                [
                    html.Label(col, style={'color': 'pink', 'fontWeight': 'bold'}),
                    dcc.Input(id=col, type='number', placeholder=f"Enter {col}",
                              style={'margin': '17px', 'width': '100%', 'padding': '10px',
                                     'borderRadius': '17px', 'backgroundColor': '#333', 'color': 'pink'})
                ],
                width=7
            ) for col in X_maternal.columns
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
# Callback for Fetal Health Analysis
@app.callback(
    Output("output", "children"),
    Input("analyze-btn", "n_clicks"),
    [Input(col, "value") for col in X.columns]
)

def analyze_fetal_health(n_clicks, *values):
    if n_clicks is None:
        return " "

    input_data = np.array(values).reshape(1, -1)
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    risk_level = model.predict_proba(input_scaled)[0][1] * 100  # Probability of risk

    # Adjusting output based on prediction
    if prediction == 1:
        status = "Normal"
        color = 'green'
    else:
        status = "At Risk"
        color = 'red'

    return html.H3(f"Fetal Health: {status} (Risk Level: {risk_level:.2f}%)", style={'color': color, 'textAlign': 'center'})

@app.callback(
    Output("output-maternal", "children"),
    Input("analyze-btn-maternal", "n_clicks"),
    [Input(col, "value") for col in X_maternal.columns]
)
def analyze_maternal_health(n_clicks, *values):
    if n_clicks is None:
        return " "

    # Create an array of the input values
    input_data = np.array(values).reshape(1, -1)
    if np.isnan(input_data).any():
        return html.H3("Error: Please fill in all fields before analyzing.", 
                       style={'color': 'red', 'textAlign': 'center'})
    input_scaled = scaler_maternal.transform(input_data)

    # Predict the risk based on the input values
    prediction = model_maternal.predict(input_scaled)[0]
    risk_level = model_maternal.predict_proba(input_scaled)[0][1] * 100  # Probability of risk

    # Find the corresponding row in the dataset (this can be approximate matching)
    # Find the closest match by comparing the input values with rows in the dataset
    diff = np.abs(X_maternal.values - input_data)  # Absolute difference
    min_diff_row = np.argmin(diff.sum(axis=1))  # Find the row with the smallest difference

    # Get the maternal_health value from the closest match in the dataset
    maternal_health_value = df_maternal.iloc[min_diff_row]['maternal_health']

    # Determine the risk based on the maternal_health value
    if maternal_health_value == 0:
        risk_status = "Low Risk"
        risk_color = 'green'
    elif maternal_health_value == 1:
        risk_status = "Mid Risk"
        risk_color = 'yellow'
    else:
        risk_status = "High Risk"
        risk_color = 'red'

    # Return the result with risk information
    return html.H3(f"Maternal Health: {risk_status} (Risk Level: {risk_level:.2f}%)", 
                   style={'color': risk_color, 'textAlign': 'center'})


#correlation analysis
correlation_analysis_layout = dbc.Container([
    html.H1("Correlation Analysis for Fetal Health", style={'textAlign': 'center', 'color': 'pink'}),
    html.P("Enter Fetal Movement and Uterine Contractions to analyze the correlation.",
           style={'textAlign': 'center', 'color': 'white'}),

    # Row for input fields
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

    # Correlation Heatmap Placeholder
    dcc.Graph(id='correlation-heatmap', style={'height': '400px'}),
], style={'backgroundColor': '#111', 'height': '100vh', 'display': 'flex', 'flexDirection': 'column'})


# Prediction Function
def get_predictions(fetal_movement, uterine_contractions):
    # Define derived values (blood pressure and gestational diabetes)
    # For simplicity, we assume simple thresholds (you can use more complex calculations here)
    blood_pressure = uterine_contractions * 0.5  # Example: Linear relation to contractions
    gestational_diabetes = fetal_movement * 0.3  # Example: Linear relation to movement
    
    # Prediction logic
    predictions = {}
    if blood_pressure > 120:  # High blood pressure threshold
        predictions['fetal_growth'] = 'Low Fetal Growth\n'
        predictions['preterm_birth'] = 'Preterm Birth Risk\n'
        predictions['fetal_distress'] = 'Risk of Fetal Distress\n'
    else:
        predictions['fetal_growth'] = 'Normal Growth\n'
        predictions['preterm_birth'] = 'Normal Birth\n'
        predictions['fetal_distress'] = 'No Risk of Fetal Distress\n'
    
    if gestational_diabetes > 50:  # High maternal diabetes threshold
        predictions['premature_birth'] = 'Premature Birth Risk\n'
        predictions['birth_defects'] = 'Risk of Birth Defects\n'
    else:
        predictions['premature_birth'] = 'Normal Birth\n'
        predictions['birth_defects'] = 'No Risk of Birth Defects\n'

    return predictions, blood_pressure, gestational_diabetes

# Callback to update results


@app.callback(
    [Output('output-correlation', 'children'),
     Output('correlation-heatmap', 'figure')],
    [Input('analyze-btn', 'n_clicks')],
    [Input('fetal-movement', 'value'),
     Input('uterine-contractions', 'value')]
)
def analyze_data(n_clicks, fetal_movement, uterine_contractions):
    if n_clicks is None:
        return '', go.Figure()

    # Check if inputs are valid
    if fetal_movement is None or uterine_contractions is None:
        return 'Please enter values for both Fetal Movement and Uterine Contractions.', go.Figure()

    # Get predictions and derived values
    predictions, blood_pressure, gestational_diabetes = get_predictions(fetal_movement, uterine_contractions)

    # Prepare prediction output text
    prediction_text = (
        f"Predictions based on inputs:\n\n"
        f"Fetal Growth: {predictions['fetal_growth']}\n"
        f"Preterm Birth: {predictions['preterm_birth']}\n"
        f"Fetal Distress: {predictions['fetal_distress']}\n"
        f"Premature Birth Risk: {predictions['premature_birth']}\n"
        f"Birth Defects Risk: {predictions['birth_defects']}\n"
    )

    # **Create a better correlation heatmap**
    # Generate a more varied dataset
    correlation_matrix = np.array([
        [1.0, np.random.uniform(0.5, 1.0) * blood_pressure / 150],
        [np.random.uniform(0.5, 1.0) * gestational_diabetes / 100, 1.0]
    ])

    # Define axis labels
    labels = ['Blood Pressure', 'Gestational Diabetes']

    # Create a heatmap with varied values
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix,
        x=labels,
        y=labels,
        colorscale='Viridis',
        colorbar=dict(title='Correlation'),
        zmin=0, zmax=1  # Normalized scale for better differentiation
    ))

    return prediction_text, fig
#login page
dashboard_layout = dbc.Container([
    html.H1("Dashboard", style={'textAlign': 'center', 'color': 'pink'}),
    dbc.Button("Logout", id="logout-btn", color="danger", style={'position': 'absolute', 'top': '10px', 'right': '10px'}),
    html.Div("Welcome to the Admin Dashboard!", style={'textAlign': 'center', 'color': 'white'})
], style={'height': '100vh', 'display': 'flex', 'flexDirection': 'column'})





# Callback to render different pages based on URL
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == '/fetal_health':
        return fetal_health_page_layout  # Navigate to fetal health analysis page
    elif pathname == '/maternal_health':
        return maternal_health_page_layout
    elif pathname == '/correlation_analysis':
        return correlation_analysis_layout 
    elif pathname == "/dashboard":
        return dashboard_layout  # Navigate to maternal health analysis page
    else:
        return main_dashboard_layout  # Default to the main dashboard page





if __name__ == '__main__':
    app.run_server(debug=True)# Callback to render different pages based on URL

