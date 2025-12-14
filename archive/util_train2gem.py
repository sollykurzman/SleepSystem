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
DATA_ROOT = 'Data'
# Two windows: Short (1 min) and Long (5 mins)
WINDOW_SHORT = 12 
WINDOW_LONG = 60

FEATURES_COLS = [
    'in_bed__inBed', 'asleep__Asleep', 'asleep__Awake', 
    'state__Core Sleep', 'state__Deep Sleep', 'state__REM Sleep'
]

def is_int(val):
    try:
        int(val)
        return True
    except ValueError:
        return False
    except TypeError:
        return False

def add_advanced_features(df):
    """
    Adds rolling features (short & long) + ratios + time features.
    """
    df_eng = df.copy()
    
    # 1. Ensure base features exist
    for col in FEATURES_COLS:
        if col not in df_eng.columns:
            df_eng[col] = 0.0
        df_eng[col] = df_eng[col].fillna(0)
        
    generated_features = []

    # 2. Rolling Features (Short Window)
    for col in FEATURES_COLS:
        mean_col = f"{col}_roll_mean_short"
        std_col = f"{col}_roll_std_short"
        
        df_eng[mean_col] = df_eng[col].rolling(window=WINDOW_SHORT, center=True, min_periods=1).mean()
        df_eng[std_col] = df_eng[col].rolling(window=WINDOW_SHORT, center=True, min_periods=1).std().fillna(0)
        generated_features.extend([mean_col, std_col])

    # 3. Rolling Features (Long Window) - Captures trend
    for col in FEATURES_COLS:
        mean_col = f"{col}_roll_mean_long"
        # We generally don't need std for long window, mean is enough for trend
        df_eng[mean_col] = df_eng[col].rolling(window=WINDOW_LONG, center=True, min_periods=1).mean()
        generated_features.append(mean_col)

    # 4. Interaction / Ratios
    # REM vs Deep Ratio (Adding small epsilon 0.01 to avoid div by zero)
    # df_eng['ratio_rem_deep'] = df_eng['state__REM Sleep'] / (df_eng['state__Deep Sleep'] + 0.01)
    # df_eng['ratio_deep_core'] = df_eng['state__Deep Sleep'] / (df_eng['state__Core Sleep'] + 0.01)
    # generated_features.extend(['ratio_rem_deep', 'ratio_deep_core'])

    # 5. Time Feature (Normalized Time since start of file)
    # This helps distinguish Deep (early night) vs REM (late night)
    # We assume the dataframe is chronological per file.
    # Create a normalized step 0.0 -> 1.0
    # df_eng['relative_time'] = np.linspace(0, 1, len(df_eng))
    # generated_features.append('relative_time')

    # Return df and list of all input features (Base + Generated)
    return df_eng, FEATURES_COLS + generated_features

def load_sleep_state_data(night_id, base_dir):
    csv_filepath = os.path.join(base_dir, str(night_id), f'true_sleep_data-{night_id}.csv')
    user_input_path = os.path.join(base_dir, str(night_id), f'inbed_data-{night_id}.csv')
    
    if not os.path.exists(csv_filepath):
        raise FileNotFoundError(f"True sleep data not found: {csv_filepath}")

    df = pd.read_csv(csv_filepath)
    df = df[df['sleep_state'] != 'Disturbance'].reset_index(drop=True)
    
    df['start_datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')
    df['end_datetime'] = pd.to_datetime(df['date'] + ' ' + df['end_time'], errors='coerce')
    
    midnight_crossover = df['end_datetime'] < df['start_datetime']
    df.loc[midnight_crossover, 'end_datetime'] += pd.Timedelta(days=1)
    df = df.dropna(subset=['start_datetime', 'end_datetime'])
    
    # Simple User Data Load (Optional)
    user_df = pd.DataFrame()
    if os.path.isfile(user_input_path):
        try:
            udf = pd.read_csv(user_input_path)
            if 'datetime' in udf.columns and 'sleep_state' in udf.columns:
                udf['start_datetime'] = pd.to_datetime(udf['datetime'], errors='coerce')
                udf['end_datetime'] = udf['start_datetime'].shift(-1)
                user_df = udf
        except: pass

    first_in_bed_time = df['start_datetime'].min()
    first_row_awake = df.iloc[0]['sleep_state'] == 'Awake' if not df.empty else False
    
    return df, user_df, first_in_bed_time, first_row_awake

def assign_labels_to_classification(class_df, night_id, base_dir):
    if not is_int(night_id):
        class_df['label'] = str(night_id)
        return class_df

    try:
        watch_df, user_df, first_in_bed, first_awake = load_sleep_state_data(night_id, base_dir)
    except FileNotFoundError: return pd.DataFrame()

    if not watch_df.empty:
        idx = pd.IntervalIndex.from_arrays(watch_df['start_datetime'], watch_df['end_datetime'], closed='left')
    else: idx = pd.IntervalIndex([])

    def get_label(ts):
        if not watch_df.empty:
            try: return watch_df.iloc[idx.get_loc(ts)]['sleep_state']
            except KeyError: pass
        if first_in_bed is not pd.NaT and ts < first_in_bed and first_awake:
            return 'notInBed'
        return None

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
            
            # --- Feature Engineering applied PER FILE ---
            df_c, features = add_advanced_features(df_c)
            if not final_features: final_features = features
            
            # Labeling
            df_labeled = assign_labels_to_classification(df_c, item, root_dir)
            
            if not df_labeled.empty and 'label' in df_labeled.columns:
                df_labeled = df_labeled.dropna(subset=['label'])
                valid_states = ['Core Sleep', 'Deep Sleep', 'REM Sleep']
                df_labeled = df_labeled[df_labeled['label'].isin(valid_states)]
                
                if not df_labeled.empty:
                    all_data.append(df_labeled)
        except Exception as e:
            print(f"Skipping {item}: {e}")

    if not all_data: return pd.DataFrame(), []
    return pd.concat(all_data, ignore_index=True), final_features

def main():
    # 1. Load Data
    df_train, feature_cols = load_training_data(DATA_ROOT)
    
    if df_train.empty:
        print("No training data.")
        return

    print(f"\nTotal Training Samples: {len(df_train)}")
    X = df_train[feature_cols]
    y = df_train['label']

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Train
    print("\n--- Training Model with Advanced Features ---")
    weights = compute_sample_weight('balanced', y_train)
    
    # Increased estimators and depth slightly to handle more features
    model = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=6, random_state=42)
    model.fit(X_train, y_train, sample_weight=weights)

    # 4. Save
    # joblib.dump(model, 'ML/state_2_model.joblib')
    # joblib.dump(feature_cols, 'ML/state_2_features.joblib')

    # 5. Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nOverall Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    unique_labels = sorted(y_test.unique())
    print(classification_report(y_test, y_pred, target_names=unique_labels))
    
    print("\nTop 15 Feature Importances:")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]
    for i in indices:
        print(f"  {feature_cols[i]}: {importances[i] * 100:.2f}%")

if __name__ == "__main__":
    main()