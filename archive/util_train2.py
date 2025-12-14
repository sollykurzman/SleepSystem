import pandas as pd
import numpy as np
import sys
import os
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight

# --- Configuration ---
DATA_ROOT = 'Data' # The root directory to search for subfolders
BUFFER_WINDOW = 12 # Window size for rolling features (approx 1 min context)

FEATURES_COLS = [
    'in_bed__inBed', 'asleep__Asleep', 'asleep__Awake', 
    'state__Core Sleep', 'state__Deep Sleep', 'state__REM Sleep'
]

# --- Helper Functions ---

def is_int(val):
    try:
        int(val)
        return True
    except ValueError:
        return False
    except TypeError:
        return False

def add_rolling_features(df):
    """
    Adds rolling mean and std dev to the dataframe to give the model context.
    """
    df_roll = df.copy()
    
    # Ensure base features exist
    for col in FEATURES_COLS:
        if col not in df_roll.columns:
            df_roll[col] = 0.0
        df_roll[col] = df_roll[col].fillna(0)

    # Add Rolling Features
    new_cols = []
    for col in FEATURES_COLS:
        # Mean: Smooths out flickering
        mean_col = f"{col}_roll_mean"
        df_roll[mean_col] = df_roll[col].rolling(window=BUFFER_WINDOW, center=True, min_periods=1).mean()
        new_cols.append(mean_col)
        
        # Std: Detects transitions
        std_col = f"{col}_roll_std"
        df_roll[std_col] = df_roll[col].rolling(window=BUFFER_WINDOW, center=True, min_periods=1).std().fillna(0)
        new_cols.append(std_col)
        
    return df_roll, FEATURES_COLS + new_cols

def load_sleep_state_data(night_id, base_dir):
    """
    Adapted from your snippet to load Ground Truth and User Input data.
    """
    csv_filepath = os.path.join(base_dir, str(night_id), f'true_sleep_data-{night_id}.csv')
    user_input_path = os.path.join(base_dir, str(night_id), f'inbed_data-{night_id}.csv')
    
    if not os.path.exists(csv_filepath):
        raise FileNotFoundError(f"True sleep data not found: {csv_filepath}")

    # Load True Data
    df = pd.read_csv(csv_filepath)
    # Exclude Disturbance from the lookup table as per requirement
    df = df[df['sleep_state'] != 'Disturbance'].reset_index(drop=True)
    
    df['start_datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')
    df['end_datetime'] = pd.to_datetime(df['date'] + ' ' + df['end_time'], errors='coerce')
    
    # Handle Midnight Crossover
    midnight_crossover = df['end_datetime'] < df['start_datetime']
    df.loc[midnight_crossover, 'end_datetime'] += pd.Timedelta(days=1)
    
    df = df.dropna(subset=['start_datetime', 'end_datetime'])
    
    # Load User Input Data (Optional)
    user_df = pd.DataFrame()
    if os.path.isfile(user_input_path):
        try:
            temp_user_df = pd.read_csv(user_input_path)
            if 'datetime' in temp_user_df.columns and 'sleep_state' in temp_user_df.columns:
                temp_user_df['start_datetime'] = pd.to_datetime(temp_user_df['datetime'], errors='coerce')
                temp_user_df = temp_user_df.dropna(subset=['start_datetime'])
                
                # Logic: State lasts until the next state starts
                temp_user_df['end_datetime'] = temp_user_df['start_datetime'].shift(-1)
                
                if not temp_user_df.empty:
                    # Assume last state lasts 24h? Or just 1 day offset as fallback
                    # Your snippet: user_df.loc[user_df.index[-1], 'end_datetime'] = user_df['start_datetime'].iloc[-1] + pd.Timedelta(days=1)
                    last_idx = temp_user_df.index[-1]
                    temp_user_df.loc[last_idx, 'end_datetime'] = temp_user_df.loc[last_idx, 'start_datetime'] + pd.Timedelta(days=1)
                
                user_crossover = temp_user_df['end_datetime'] < temp_user_df['start_datetime']
                temp_user_df.loc[user_crossover, 'end_datetime'] += pd.Timedelta(days=1)
                
                user_df = temp_user_df
        except Exception as e:
            print(f"Warning: Failed to load user data {user_input_path}: {e}")
    
    first_in_bed_time = df['start_datetime'].min()
    first_row_awake = df.iloc[0]['sleep_state'] == 'Awake' if not df.empty else False
    
    return df, user_df, first_in_bed_time, first_row_awake

def assign_labels_to_classification(class_df, night_id, base_dir):
    """
    Applies the hierarchy logic to label the classification dataframe.
    """
    # 1. Non-Integer ID case: Whole folder is one state
    if not is_int(night_id):
        class_df['label'] = str(night_id)
        return class_df

    # 2. Integer ID case: Load hierarchies
    try:
        watch_df, user_df, first_in_bed_time, first_row_awake = load_sleep_state_data(night_id, base_dir)
    except FileNotFoundError:
        return pd.DataFrame() # Skip if files missing

    # Optimization: Use IntervalIndex for faster lookup than row-by-row apply
    # Create Intervals for Watch Data
    if not watch_df.empty:
        watch_idx = pd.IntervalIndex.from_arrays(watch_df['start_datetime'], watch_df['end_datetime'], closed='left')
    else:
        watch_idx = pd.IntervalIndex([])

    # Create Intervals for User Data
    if not user_df.empty:
        user_idx = pd.IntervalIndex.from_arrays(user_df['start_datetime'], user_df['end_datetime'], closed='left')
    else:
        user_idx = pd.IntervalIndex([])

    def get_label(ts):
        # 1. Check Watch Data
        if not watch_df.empty:
            try:
                loc = watch_idx.get_loc(ts)
                return watch_df.iloc[loc]['sleep_state']
            except KeyError:
                pass
        
        # 2. Check User Data
        if not user_df.empty:
            try:
                loc = user_idx.get_loc(ts)
                return user_df.iloc[loc]['sleep_state']
            except KeyError:
                pass
            
        # 3. Check Pre-bed Logic
        if first_in_bed_time is not pd.NaT and ts < first_in_bed_time and first_row_awake:
            return 'notInBed'
            
        return None

    # Apply labeling
    class_df['label'] = class_df['datetime'].apply(get_label)
    
    return class_df

def load_training_data(root_dir):
    print(f"Scanning '{root_dir}' for training data...")
    all_data = []
    final_features = []
    
    for item in os.listdir(root_dir):
        sub_path = os.path.join(root_dir, item)
        if not os.path.isdir(sub_path): continue
            
        class_file = os.path.join(sub_path, f'classification-{item}.csv')
        if not os.path.exists(class_file): continue
            
        try:
            df_c = pd.read_csv(class_file)
            df_c['datetime'] = pd.to_datetime(df_c['datetime'], format='mixed')
            
            # Prepare features
            df_c, features = add_rolling_features(df_c)
            if not final_features: final_features = features
            
            # Label
            df_labeled = assign_labels_to_classification(df_c, item, root_dir)
            
            if not df_labeled.empty and 'label' in df_labeled.columns:
                df_labeled = df_labeled.dropna(subset=['label'])
                
                # --- FILTERING STEP ---
                # Exclude Disturbance, Awake, and notInBed
                # The model will ONLY see Core, Deep, and REM
                valid_states = ['Core Sleep', 'Deep Sleep', 'REM Sleep']
                df_labeled = df_labeled[df_labeled['label'].isin(valid_states)]
                
                if not df_labeled.empty:
                    all_data.append(df_labeled)
        except Exception as e:
            print(f"Skipping {item}: {e}")

    if not all_data: return pd.DataFrame(), []
    return pd.concat(all_data, ignore_index=True), final_features

# --- Main Execution ---

def main():
    # 1. Load and Prep Data
    df_train, feature_cols = load_training_data(DATA_ROOT)
    
    if df_train.empty:
        print("No training data found. Please check 'Data/' folder structure.")
        return

    print(f"\nTotal Training Samples: {len(df_train)}")
    print(f"Labels found: {df_train['label'].unique()}")

    X = df_train[feature_cols]
    y = df_train['label']

    # 2. Split and Evaluate
    # Stratify ensures we have representation of all classes in test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Calculate sample weights for class balance
    train_weights = compute_sample_weight('balanced', y_train)

    print("\n--- Training Advanced Gradient Boosting Model ---")
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train, sample_weight=train_weights)

    joblib.dump(model, 'ML/state_2_model.joblib')
    joblib.dump(feature_cols, 'ML/state_2_features.joblib')

    # --- Requested Evaluation Block ---
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nOverall Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    # Helper to get class names dynamically
    unique_labels = sorted(y_test.unique())
    print(classification_report(y_test, y_pred, target_names=unique_labels))

    print("\nFeature Importances:")
    importances = model.feature_importances_
    # Sort and display top 10
    indices = np.argsort(importances)[::-1][:10]
    for i in indices:
        print(f"  {feature_cols[i]}: {importances[i] * 100:.2f}%")
    # ----------------------------------

    # 3. Predict on New Files (if provided)
    if len(sys.argv) > 1:
        print("\n--- Predicting on New Files ---")
        # Retrain on full data for best performance
        full_weights = compute_sample_weight('balanced', y)
        model.fit(X, y, sample_weight=full_weights)
        
        target_files = sys.argv[1:]
        for f in target_files:
            if not os.path.exists(f): continue
            try:
                print(f"Predicting {f}...")
                df_new = pd.read_csv(f)
                df_new['datetime'] = pd.to_datetime(df_new['datetime'], format='mixed')
                
                # Apply same buffering
                df_processed, _ = add_rolling_features(df_new)
                X_new = df_processed[feature_cols] # Ensure column order match
                
                preds = model.predict(X_new)
                
                # Generate Interval Output
                results = []
                current_state = preds[0]
                start_time = df_new.iloc[0]['datetime']
                
                for i in range(1, len(preds)):
                    state = preds[i]
                    time = df_new.iloc[i]['datetime']
                    if state != current_state:
                        duration = (time - start_time).total_seconds() / 60.0
                        if duration > 0:
                            results.append({
                                'date': start_time.strftime('%Y-%m-%d'),
                                'time': start_time.strftime('%H:%M:%S'),
                                'sleep_state': current_state,
                                'end_time': time.strftime('%H:%M:%S'),
                                'duration_minutes': round(duration, 1)
                            })
                        current_state = state
                        start_time = time
                
                # Save
                out_df = pd.DataFrame(results)
                out_name = f"predicted_{os.path.basename(f)}"
                out_df.to_csv(out_name, index=False)
                print(f"Saved: {out_name}")
                
            except Exception as e:
                print(f"Error predicting {f}: {e}")

if __name__ == "__main__":
    main()
