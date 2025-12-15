#!/usr/bin/python3

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# define colors for sleep states using rgba for transparency
STATE_COLORS = {
    'Core Sleep': 'rgba(0, 0, 255, 0.2)',    
    'Deep Sleep': 'rgba(128, 0, 128, 0.2)',  
    'REM Sleep':  'rgba(0, 255, 255, 0.2)',  
    'REM':        'rgba(0, 255, 255, 0.2)',  
    'Awake':      'rgba(255, 165, 0, 0.2)',  
    'inBed':      'rgba(128, 128, 128, 0.2)', 
    'notInBed':   'rgba(50, 50, 50, 0.2)',    
    'Disturbance':'rgba(255, 0, 0, 0.2)',     
    'default':    'rgba(0, 0, 0, 0.1)'
}

def load_true_sleep_data(classification_path):
    # extract directory and filename
    dirname = os.path.dirname(classification_path)
    filename = os.path.basename(classification_path)
    
    # parse night id from filename
    try:
        night_id = filename.rsplit('-', 1)[-1].replace('.csv', '')
    except IndexError:
        return None

    # construct path for ground truth file
    true_filename = f"true_sleep_data-{night_id}.csv"
    true_path = os.path.join(dirname, true_filename)

    # return none if file does not exist
    if not os.path.exists(true_path):
        return None

    try:
        # load csv data
        df = pd.read_csv(true_path)
        
        # remove disturbance records
        df = df[df['sleep_state'] != 'Disturbance'].copy()
        
        # combine date and time into datetime objects
        df['start_dt'] = pd.to_datetime(df['date'] + ' ' + df['time'])
        df['end_dt'] = pd.to_datetime(df['date'] + ' ' + df['end_time'])
        
        # adjust end time for midnight crossovers
        mask = df['end_dt'] < df['start_dt']
        df.loc[mask, 'end_dt'] += pd.Timedelta(days=1)
        
        return df
    except Exception as e:
        print(f"Warning: Failed to load true data ({e})")
        return None

def plot_sleep_analysis(file_path):
    # validate file existence
    if not os.path.exists(file_path):
        print(f"Error: File not found {file_path}")
        return

    print(f"Loading {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # validate datetime column presence
    if 'datetime' not in df.columns:
        print("Error: 'datetime' column missing.")
        return

    # parse and sort by datetime
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce', format='mixed')
    df = df.sort_values('datetime')

    # calculate sampling interval to bridge gaps in plotting
    try:
        median_interval = df['datetime'].diff().median()
        if pd.isnull(median_interval): median_interval = pd.Timedelta(seconds=5)
    except:
        median_interval = pd.Timedelta(seconds=5)

    # identify probability columns
    prob_cols = df.select_dtypes(include=['float64', 'float32']).columns.tolist()

    # reshape dataframe for line plotting
    df_long = df.melt(id_vars=['datetime'], value_vars=prob_cols, var_name='State', value_name='Probability')
    
    # initialize line chart for probabilities
    fig = px.line(
        df_long, 
        x='datetime', 
        y='Probability', 
        color='State',
        title=f'Sleep Analysis: {os.path.basename(file_path)}',
        template='plotly_dark',
        labels={'datetime': 'Time', 'Probability': 'Confidence'}
    )

    # load ground truth data
    df_true = load_true_sleep_data(file_path)
    
    # overlay ground truth blocks
    if df_true is not None and not df_true.empty:
        for state in df_true['sleep_state'].unique():
            subset = df_true[df_true['sleep_state'] == state]
            color = STATE_COLORS.get(state, STATE_COLORS['default'])
            
            # build coordinates for background rectangles
            x_vals = []
            y_vals = []
            for _, row in subset.iterrows():
                x_vals.extend([row['start_dt'], row['end_dt'], row['end_dt'], row['start_dt'], None])
                y_vals.extend([0, 0, 1, 1, None]) 

            # add ground truth trace
            fig.add_trace(go.Scatter(
                x=x_vals, y=y_vals,
                fill='toself', fillcolor=color,
                mode='none', 
                name=f"True {state}",
                legendgroup=f"true_{state}", 
                hoverinfo='text', text=f"True {state}",
                opacity=1.0 
            ))

    # overlay predicted classification blocks
    if 'classification' in df.columns:
        # identify transitions between states
        df['change'] = df['classification'] != df['classification'].shift(1)
        df['group'] = df['change'].cumsum()
        
        # determine start and end times for continuous segments
        groups = df.groupby(['group', 'classification'])['datetime'].agg(['min', 'max']).reset_index()
        groups['max'] = groups['max'] + median_interval

        unique_preds = groups['classification'].unique()
        
        for state in unique_preds:
            subset = groups[groups['classification'] == state]
            color = STATE_COLORS.get(state, STATE_COLORS['default'])
            
            # build coordinates for prediction rectangles
            x_vals = []
            y_vals = []
            for _, row in subset.iterrows():
                x_vals.extend([row['min'], row['max'], row['max'], row['min'], None])
                y_vals.extend([0, 0, 1, 1, None])

            # add prediction trace
            fig.add_trace(go.Scatter(
                x=x_vals, y=y_vals,
                fill='toself', fillcolor=color,
                mode='none',
                name=f"Pred {state}",
                legendgroup=f"pred_{state}",
                hoverinfo='text', text=f"Pred {state}",
                visible=True 
            ))

    # configure axis labels and legend interaction
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Probability / State",
        hovermode="x unified",
        legend=dict(groupclick="toggleitem") 
    )

    # display plot
    fig.show()

if __name__ == "__main__":
    # handle command line arguments or set default
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
    else:
        date = '211125'
        target_file = f'Data/{date}/classification_refined-{date}.csv' 
        if not os.path.exists(target_file):
            print("No file provided and default not found.")
            sys.exit(1)
            
    plot_sleep_analysis(target_file)