import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# --- COLOR CONFIGURATION ---
# Using RGBA for transparency (0.2 opacity)
# This allows 'True' and 'Pred' blocks to overlap, creating a darker color where they agree.
STATE_COLORS = {
    'Core Sleep': 'rgba(0, 0, 255, 0.2)',    # Blue
    'Deep Sleep': 'rgba(128, 0, 128, 0.2)',  # Purple
    'REM Sleep':  'rgba(0, 255, 255, 0.2)',  # Cyan
    'REM':        'rgba(0, 255, 255, 0.2)',  # Cyan (alt name)
    'Awake':      'rgba(255, 165, 0, 0.2)',  # Orange
    'inBed':      'rgba(128, 128, 128, 0.2)', # Grey
    'notInBed':   'rgba(50, 50, 50, 0.2)',    # Dark Grey
    'Disturbance':'rgba(255, 0, 0, 0.2)',     # Red
    'default':    'rgba(0, 0, 0, 0.1)'
}

def load_true_sleep_data(classification_path):
    """
    Attempts to find and load the matching true_sleep_data CSV.
    """
    dirname = os.path.dirname(classification_path)
    filename = os.path.basename(classification_path)
    
    # Extract night_id (e.g., from classification-211125.csv)
    try:
        # Splits by '-' and takes the last part, removing extension
        night_id = filename.rsplit('-', 1)[-1].replace('.csv', '')
    except IndexError:
        return None

    true_filename = f"true_sleep_data-{night_id}.csv"
    true_path = os.path.join(dirname, true_filename)

    if not os.path.exists(true_path):
        return None

    try:
        df = pd.read_csv(true_path)
        
        # Exclude Disturbances (as requested)
        df = df[df['sleep_state'] != 'Disturbance'].copy()
        
        # Parse Dates
        df['start_dt'] = pd.to_datetime(df['date'] + ' ' + df['time'])
        df['end_dt'] = pd.to_datetime(df['date'] + ' ' + df['end_time'])
        
        # Handle Midnight Crossover
        mask = df['end_dt'] < df['start_dt']
        df.loc[mask, 'end_dt'] += pd.Timedelta(days=1)
        
        return df
    except Exception as e:
        print(f"Warning: Failed to load true data ({e})")
        return None

def plot_sleep_analysis(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found {file_path}")
        return

    print(f"Loading {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    if 'datetime' not in df.columns:
        print("Error: 'datetime' column missing.")
        return

    # Parse main datetime
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce', format='mixed')
    df = df.sort_values('datetime')

    # Calculate average sampling interval to fill gaps in "Predicted" blocks
    try:
        median_interval = df['datetime'].diff().median()
        if pd.isnull(median_interval): median_interval = pd.Timedelta(seconds=5)
    except:
        median_interval = pd.Timedelta(seconds=5)

    # Identify Probability Columns
    prob_cols = df.select_dtypes(include=['float64', 'float32']).columns.tolist()

    # --- START PLOTTING ---
    
    # 1. Base Plot: Probability Lines
    df_long = df.melt(id_vars=['datetime'], value_vars=prob_cols, var_name='State', value_name='Probability')
    
    fig = px.line(
        df_long, 
        x='datetime', 
        y='Probability', 
        color='State',
        title=f'Sleep Analysis: {os.path.basename(file_path)}',
        template='plotly_dark',
        labels={'datetime': 'Time', 'Probability': 'Confidence'}
    )

    # 2. Add "True Data" Blocks (Background)
    df_true = load_true_sleep_data(file_path)
    if df_true is not None and not df_true.empty:
        for state in df_true['sleep_state'].unique():
            subset = df_true[df_true['sleep_state'] == state]
            color = STATE_COLORS.get(state, STATE_COLORS['default'])
            
            # Construct a single trace with multiple rectangles
            x_vals = []
            y_vals = []
            for _, row in subset.iterrows():
                x_vals.extend([row['start_dt'], row['end_dt'], row['end_dt'], row['start_dt'], None])
                y_vals.extend([0, 0, 1, 1, None]) # Full height (0 to 1)

            fig.add_trace(go.Scatter(
                x=x_vals, y=y_vals,
                fill='toself', fillcolor=color,
                mode='none', # No border line
                name=f"True {state}",
                legendgroup=f"true_{state}", # Group for toggling
                hoverinfo='text', text=f"True {state}",
                opacity=1.0 # Opacity handled by RGBA color
            ))

    # 3. Add "Predicted Decision" Blocks (Background)
    if 'classification' in df.columns:
        # Identify continuous segments of the same decision
        # Shift(1) compares current row with previous row
        df['change'] = df['classification'] != df['classification'].shift(1)
        df['group'] = df['change'].cumsum()
        
        # Group by these continuous segments
        # We take the min time as start, and max time (+ interval) as end
        groups = df.groupby(['group', 'classification'])['datetime'].agg(['min', 'max']).reset_index()
        groups['max'] = groups['max'] + median_interval

        unique_preds = groups['classification'].unique()
        
        for state in unique_preds:
            subset = groups[groups['classification'] == state]
            color = STATE_COLORS.get(state, STATE_COLORS['default'])
            
            x_vals = []
            y_vals = []
            for _, row in subset.iterrows():
                x_vals.extend([row['min'], row['max'], row['max'], row['min'], None])
                y_vals.extend([0, 0, 1, 1, None])

            fig.add_trace(go.Scatter(
                x=x_vals, y=y_vals,
                fill='toself', fillcolor=color,
                mode='none',
                name=f"Pred {state}",
                legendgroup=f"pred_{state}",
                hoverinfo='text', text=f"Pred {state}",
                visible=True # Show by default (can be 'legendonly')
            ))

    # Layout improvements
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Probability / State",
        hovermode="x unified",
        legend=dict(groupclick="toggleitem") # Clicking toggles the specific item
    )

    fig.show()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
    else:
        # Default for testing
        date = '211125'
        target_file = f'Data/{date}/classification_refined-{date}.csv' 
        if not os.path.exists(target_file):
            print("No file provided and default not found.")
            sys.exit(1)
            
    plot_sleep_analysis(target_file)

# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# import sys
# import os

# # Define colors for the Ground Truth blocks (background)
# # Using RGBA for transparency (0.2 opacity)
# TRUE_STATE_COLORS = {
#     'Core Sleep': 'rgba(0, 0, 255, 0.2)',    # Blue
#     'Deep Sleep': 'rgba(128, 0, 128, 0.2)',  # Purple
#     'REM Sleep':  'rgba(0, 255, 255, 0.2)',  # Cyan
#     'Awake':      'rgba(255, 165, 0, 0.2)',  # Orange
#     'inBed':      'rgba(128, 128, 128, 0.2)', # Grey
#     # Fallback for unknown states
#     'default':    'rgba(0, 0, 0, 0.1)'
# }

# def load_true_sleep_data(classification_path):
#     """
#     Attempts to find and load the matching true_sleep_data CSV.
#     Returns a DataFrame or None.
#     """
#     dirname = os.path.dirname(classification_path)
#     filename = os.path.basename(classification_path)
    
#     # Extract night_id. Assuming format like "classification-211125.csv" or "classification_refined-211125.csv"
#     # We take the part after the last hyphen and remove extension
#     try:
#         night_id = filename.rsplit('-', 1)[-1].replace('.csv', '')
#     except IndexError:
#         print("Warning: Could not extract night_id from filename. Skipping true data.")
#         return None

#     true_filename = f"true_sleep_data-{night_id}.csv"
#     true_path = os.path.join(dirname, true_filename)

#     if not os.path.exists(true_path):
#         print(f"Info: No true sleep data found at '{true_path}'.")
#         return None

#     try:
#         df = pd.read_csv(true_path)
        
#         # 1. Exclude Disturbances
#         df = df[df['sleep_state'] != 'Disturbance'].copy()
        
#         # 2. Parse Timestamps
#         # We need to construct full datetimes for start and end
#         # Assuming 'date', 'time', and 'end_time' columns exist
#         df['start_dt'] = pd.to_datetime(df['date'] + ' ' + df['time'])
#         df['end_dt'] = pd.to_datetime(df['date'] + ' ' + df['end_time'])
        
#         # 3. Handle Midnight Crossover
#         # If end_time is earlier than start_time, it implies the next day
#         mask = df['end_dt'] < df['start_dt']
#         df.loc[mask, 'end_dt'] += pd.Timedelta(days=1)
        
#         print(f"Loaded True Sleep Data: {len(df)} segments found.")
#         return df
        
#     except Exception as e:
#         print(f"Error loading true sleep data: {e}")
#         return None

# def plot_state_probabilities(file_path):
#     """
#     Reads a classification CSV and plots probabilities with true sleep data overlay.
#     """
#     if not os.path.exists(file_path):
#         print(f"Error: The file '{file_path}' was not found.")
#         return

#     # --- 1. Load Classification Data ---
#     try:
#         df = pd.read_csv(file_path)
#     except Exception as e:
#         print(f"Error reading CSV file: {e}")
#         return

#     if 'datetime' not in df.columns:
#         print("Error: 'datetime' column not found in the CSV.")
#         return

#     # Convert datetime column
#     df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce', format='mixed')

#     # Identify probability columns
#     prob_cols = df.select_dtypes(include=['float64', 'float32']).columns.tolist()
#     if not prob_cols:
#         print("No probability (float) columns found to plot.")
#         return

#     print(f"Plotting probabilities: {prob_cols}")

#     # Reshape for Plotly Express
#     df_long = df.melt(
#         id_vars=['datetime'], 
#         value_vars=prob_cols, 
#         var_name='State', 
#         value_name='Probability'
#     )

#     # --- 2. Create Base Figure ---
#     fig = px.line(
#         df_long, 
#         x='datetime', 
#         y='Probability', 
#         color='State',
#         title=f'State Probabilities vs True Data: {os.path.basename(file_path)}',
#         labels={'datetime': 'Time', 'Probability': 'Confidence Score'},
#         template='plotly_dark'
#     )

#     # --- 3. Load and Add True Data Overlay ---
#     df_true = load_true_sleep_data(file_path)
    
#     if df_true is not None and not df_true.empty:
#         unique_states = df_true['sleep_state'].unique()
        
#         for state in unique_states:
#             state_data = df_true[df_true['sleep_state'] == state]
#             color = TRUE_STATE_COLORS.get(state, TRUE_STATE_COLORS['default'])
            
#             # We construct a "Shape" trace for this state
#             # To make it a single toggleable legend item, we use one Scatter trace
#             # with multiple filled areas separated by None
#             x_vals = []
#             y_vals = []
            
#             for _, row in state_data.iterrows():
#                 # Define rectangle coordinates (Start, End, End, Start, None)
#                 x_vals.extend([row['start_dt'], row['end_dt'], row['end_dt'], row['start_dt'], None])
#                 y_vals.extend([0, 0, 1, 1, None]) # Cover full height 0 to 1
            
#             fig.add_trace(go.Scatter(
#                 x=x_vals,
#                 y=y_vals,
#                 fill='toself',
#                 fillcolor=color,
#                 mode='none', # No border lines
#                 name=f"True {state}", # Legend Name
#                 hoverinfo='text',
#                 text=f"True {state}", # Hover text
#                 opacity=1.0, # Opacity is handled in the color definition
#                 legendgroup=f"true_{state}", # Grouping (optional here since it's one trace)
#                 showlegend=True
#             ))

#     # --- 4. Final Layout Adjustments ---
#     fig.update_layout(
#         xaxis_title="Time",
#         yaxis_title="Probability",
#         legend_title="Legend",
#         hovermode="x unified"
#     )

#     fig.show()

# if __name__ == "__main__":
#     # --- CONFIGURATION ---
    
#     # Check for command line argument
#     if len(sys.argv) > 1:
#         target_file = sys.argv[1]
#     else:
#         # Default fallback
#         date = '211125'
#         target_file = f'Data/{date}/classification-{date}.csv' 
    
#     plot_state_probabilities(target_file)