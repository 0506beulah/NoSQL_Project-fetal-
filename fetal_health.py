'''import dash
from dash import dcc, html
import dash_bootstrap_components as dbc

# Load the dataset for Fetal Health Analysis
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load Dataset
df = pd.read_csv("f_health.csv")

# Drop unwanted columns and handle missing values
drop_columns = ["histogram_min", "histogram_max", "histogram_number_of_peaks", "histogram_number_of_zeroes", 
                "histogram_mode", "histogram_mean", "histogram_median", "histogram_variance", "histogram_tendency"]
df.drop(columns=drop_columns, inplace=True, errors='ignore')
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

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

# Fetal Health Analysis Layout
fetal_health_page_layout = dbc.Container([
    html.H1("Fetal Health Analysis", style={'textAlign': 'center', 'color': 'pink'}),
    html.P("Enter the parameters to analyze fetal health.", style={'textAlign': 'center', 'color': 'white'}),

    dbc.Row(
        [
            dbc.Col(
                [
                    html.Label(col, style={'color': 'pink', 'fontWeight': 'bold'}),
                    dcc.Input(id=col, type='number', placeholder=f"Enter {col}", 
                              style={'margin': '5px', 'width': '100%', 'padding': '10px', 
                                     'borderRadius': '8px', 'backgroundColor': '#333', 'color': 'white'})
                ],
                width=6
            ) for col in X.columns
        ],
        justify="center",
        style={'flex': '1', 'display': 'flex', 'alignItems': 'center', 'height': '100%'}
    ),
    
    html.Br(),
    dbc.Button("Analyze", id="analyze-btn", color="primary", className="mb-3", 
               style={'backgroundColor': 'black', 'color': 'pink', 'borderRadius': '8px'}),
    
    #html.Br(),
    html.Div(id="output"),

    #html.Br(),
    dbc.Button("Back to Home", href="/", color="primary", className="mb-3", 
               style={'backgroundColor': 'black', 'color': 'pink', 'borderRadius': '5px',  'margin-bottom': '200px'}),
    
], style={'backgroundColor': '#111', 'height': '100vh', 'display': 'flex', 'flexDirection': 'column'})
'''
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Load Dataset
df = pd.read_csv("f_health.csv")
df.columns = df.columns.str.strip()
# Drop unwanted columns and handle missing values
drop_columns = ["histogram_min", "histogram_max", "histogram_number_of_peaks", "histogram_number_of_zeroes", 
                "histogram_mode", "histogram_mean", "histogram_median", "histogram_variance", "histogram_tendency", 
                "histogram_width"]  # Added histogram_width

df.drop(columns = drop_columns, inplace = True, errors = 'ignore')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]  
print("Before Dropping NA, Shape of DataFrame:", df.shape)  
print("Before Dropping NA, Shape of DataFrame:", df.shape)
print(df.head())  # Show a few rows to check if data exists

# Drop rows with too many missing values
df.dropna(thresh=len(df.columns) - 2, inplace=True)  
df.dropna(inplace=True)
print("After Dropping NA, Shape of DataFrame:", df.shape)  # Check if it became empty
print("Remaining Columns:", df.columns)# Add histogram_width to the list if missing
  # Removes any leading/trailing spaces in column names




# Convert categorical columns (if any)
if df["fetal_health"].dtype == "object":
    label_encoder = LabelEncoder()
    df["fetal_health"] = label_encoder.fit_transform(df["fetal_health"])

# Split features and target variable
'''X = df.drop(columns=['fetal_health'])
y = df['fetal_health']
'''

X = df.drop(columns=['fetal_health'])  # Keep X for model training
y = df['fetal_health']  # Target variable
print("Final Features Used for Input:", X.columns)

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print("X_train Shape:", X_train.shape)
print("X_test Shape:", X_test.shape)

model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

# Fetal Health Analysis Layout
fetal_health_layout = dbc.Container([
    html.H1("Fetal Health Analysis", style={'textAlign': 'center', 'color': 'pink'}),
    html.P("Enter the parameters to analyze fetal health.", style={'textAlign': 'center', 'color': 'white'}),

    dbc.Row(
        [
            dbc.Col(
                [
                    html.Label(col, style={'color': 'pink', 'fontWeight': 'bold'}),
                    dcc.Input(id=col.replace(" ", "_").replace("-", "_"), type='number', placeholder=f"Enter {col}", 
                              style={'margin': '5px', 'width': '100%', 'padding': '10px', 
                                     'borderRadius': '8px', 'backgroundColor': '#333', 'color': 'white'})
                ],
                width=6
            ) for col in X.columns
        ],
        justify="center",
        style={'flex': '1', 'display': 'flex', 'alignItems': 'center', 'height': '100%'}
    ),
    
    html.Br(),
    dbc.Button("Analyze", id="analyze-btn", color="primary", className="mb-3", 
               style={'backgroundColor': 'black', 'color': 'pink', 'borderRadius': '8px'}),

    # Output Div
    html.Div(id="output", style={'color': 'pink', 'textAlign': 'center', 'fontSize': '20px', 'marginTop': '20px'}),

    dbc.Button("Back to Home", href="/", color="primary", className="mb-3", 
               style={'backgroundColor': 'black', 'color': 'pink', 'borderRadius': '5px', 'margin-bottom': '200px'}),

], style={'backgroundColor': '#111', 'height': '100vh', 'display': 'flex', 'flexDirection': 'column'})