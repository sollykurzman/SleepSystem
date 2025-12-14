import pandas as pd
import plotly.express as px
import sys
import os

def plot_state_probabilities(file_path):
    """
    Reads a classification CSV and plots the probabilities of each state over time.
    """
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found.")
        return

    # Load the data
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Check for datetime column
    if 'datetime' not in df.columns:
        print("Error: 'datetime' column not found in the CSV.")
        return

    # Convert datetime column to proper datetime objects
    # df['datetime'] = pd.to_datetime(df['datetime'])
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce', format='mixed')

    # Identify probability columns
    # We select all columns that are numeric (floats)
    # You can also customize this list manually if needed
    prob_cols = df.select_dtypes(include=['float64', 'float32']).columns.tolist()

    if not prob_cols:
        print("No probability (float) columns found to plot.")
        return

    print(f"Plotting the following states: {prob_cols}")

    # Reshape the DataFrame to 'long' format for Plotly Express
    # This stacks all probability columns into a single 'Probability' column
    # and creates a 'State' column to distinguish them.
    df_long = df.melt(
        id_vars=['datetime'], 
        value_vars=prob_cols, 
        var_name='State', 
        value_name='Probability'
    )

    # Create the interactive line chart
    fig = px.line(
        df_long, 
        x='datetime', 
        y='Probability', 
        color='State',
        title=f'State Probabilities Over Time: {os.path.basename(file_path)}',
        labels={'datetime': 'Time', 'Probability': 'Confidence Score'},
        template='plotly_dark'  # Optional: looks good for scientific data
    )

    # Improve hover information and layout
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Probability",
        legend_title="State",
        hovermode="x unified" # Shows all values for a specific time on hover
    )

    # Show the plot
    fig.show()

if __name__ == "__main__":
    # --- CONFIGURATION ---
    # You can change this path to point to your new files
    # Example: file_path = '/path/to/your/classification-new.csv'
    
    # Default to argument if provided, otherwise use a hardcoded path
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
    else:
        # REPLACE THIS with your default file path
        date = '211125'
        target_file = f'Data/{date}/classification_refined-{date}.csv' 
    
    plot_state_probabilities(target_file)