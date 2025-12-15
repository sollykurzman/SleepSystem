#!/usr/bin/python3

import pandas as pd
import numpy as np
import sys
import os
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight

# data file configuration
DATA_ROOT = 'Data'
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
    df_eng = df.copy()
    
    # ensure all base feature columns exist
    for col in FEATURES_COLS:
        if col not in df_eng.columns:
            df_eng[col] = 0.0
        df_eng[col] = df_eng[col].fillna(0)
        
    generated_features = []

    # calculate rolling stats for short window
    for col in FEATURES_COLS:
        mean_col = f"{col}_roll_mean_short"
        std_col = f"{col}_roll_std_short"
        
        df_eng[mean_col] = df_eng[col].rolling(window=WINDOW_SHORT, center=True, min_periods=1).mean()
        df_eng[std_col] = df_eng[col].rolling(window=WINDOW_SHORT, center=True, min_periods=1).std().fillna(0)
        generated_features.extend([mean_col, std_col])

    # calculate rolling stats for long window to capture trends
    for col in FEATURES_COLS:
        mean_col = f"{col}_roll_mean_long"
        
        df_eng[mean_col] = df_eng[col].rolling(window=WINDOW_LONG, center=True, min_periods=1).mean()
        generated_features.append(mean_col)

    return df_eng, FEATURES_COLS + generated_features

def load_sleep_state_data(night_id, base_dir):
    # construct file paths
    csv_filepath = os.path.join(base_dir, str(night_id), f'true_sleep_data-{night_id}.csv')
    user_input_path = os.path.join(base_dir, str(night_id), f'inbed_data-{night_id}.csv')
    
    if not os.path.exists(csv_filepath):
        raise FileNotFoundError(f"True sleep data not found: {csv_filepath}")

    # load and clean watch data
    df = pd.read_csv(csv_filepath)
    df = df[df['sleep_state'] != 'Disturbance'].reset_index(drop=True)
    
    # parse datetime columns
    df['start_datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')
    df['end_datetime'] = pd.to_datetime(df['date'] + ' ' + df['end_time'], errors='coerce')
    
    # handle midnight crossover
    midnight_crossover = df['end_datetime'] < df['start_datetime']
    df.loc[midnight_crossover, 'end_datetime'] += pd.Timedelta(days=1)
    df = df.dropna(subset=['start_datetime', 'end_datetime'])
    
    # attempt to load user input data
    user_df = pd.DataFrame()
    if os.path.isfile(user_input_path):
        try:
            udf = pd.read_csv(user_input_path)
            
            # validate required columns
            if 'datetime' in udf.columns and 'sleep_state' in udf.columns:
                udf['start_datetime'] = pd.to_datetime(udf['datetime'], errors='coerce')
                udf = udf.dropna(subset=['start_datetime'])
                
                # infer end time from next row start time
                udf['end_datetime'] = udf['start_datetime'].shift(-1)
                
                # set default duration for last entry
                if not udf.empty:
                    udf.loc[udf.index[-1], 'end_datetime'] = udf['start_datetime'].iloc[-1] + pd.Timedelta(days=1)
                
                # handle crossover for user data
                user_crossover = udf['end_datetime'] < udf['start_datetime']
                udf.loc[user_crossover, 'end_datetime'] += pd.Timedelta(days=1)
                
                user_df = udf
        except Exception as e:
            print(f"Error loading user data for {night_id}: {e}")

    # identify initial bed state
    first_in_bed_time = df['start_datetime'].min()
    first_row_awake = df.iloc[0]['sleep_state'] == 'Awake' if not df.empty else False
    
    return df, user_df, first_in_bed_time, first_row_awake

def assign_labels_to_classification(class_df, night_id, base_dir):
    # return immediately if night id is invalid
    if not is_int(night_id):
        class_df['label'] = str(night_id)
        return class_df

    try:
        watch_df, user_df, first_in_bed, first_awake = load_sleep_state_data(night_id, base_dir)
    except FileNotFoundError: 
        return pd.DataFrame()

    # create interval index for fast watch data lookup
    if not watch_df.empty:
        idx_watch = pd.IntervalIndex.from_arrays(watch_df['start_datetime'], watch_df['end_datetime'], closed='left')
    else: 
        idx_watch = pd.IntervalIndex([])

    # create interval index for fast user data lookup
    if not user_df.empty:
        idx_user = pd.IntervalIndex.from_arrays(user_df['start_datetime'], user_df['end_datetime'], closed='left')
    else:
        idx_user = pd.IntervalIndex([])

    def get_label(ts):
        # check watch data first
        if not watch_df.empty:
            try: 
                return watch_df.iloc[idx_watch.get_loc(ts)]['sleep_state']
            except KeyError: 
                pass
        
        # fallback to user data
        if not user_df.empty:
            try:
                return user_df.iloc[idx_user.get_loc(ts)]['sleep_state']
            except KeyError:
                pass

        # check for not-in-bed condition prior to sleep
        if first_in_bed is not pd.NaT and ts < first_in_bed and first_awake:
            return 'notInBed'
            
        return None

    # apply labeling logic to all timestamps
    class_df['label'] = class_df['datetime'].apply(get_label)
    return class_df

def load_training_data(root_dir):
    print(f"Scanning '{root_dir}' for training data...")
    all_data = []
    final_features = []
    
    # iterate through night directories
    for item in os.listdir(root_dir):
        sub_path = os.path.join(root_dir, item)
        if not os.path.isdir(sub_path): continue
            
        # check for pre-refinement classification file
        class_file = os.path.join(sub_path, f'classification_prerefinement-{item}.csv')
        if not os.path.exists(class_file): continue
            
        try:
            df_c = pd.read_csv(class_file)
            df_c['datetime'] = pd.to_datetime(df_c['datetime'], format='mixed')
            
            # apply feature engineering
            df_c, features = add_advanced_features(df_c)
            if not final_features: final_features = features
            
            # assign ground truth labels
            df_labeled = assign_labels_to_classification(df_c, item, root_dir)
            
            # filter for valid sleep states
            if not df_labeled.empty and 'label' in df_labeled.columns:
                df_labeled = df_labeled.dropna(subset=['label'])
                valid_states = ['Core Sleep', 'Deep Sleep', 'REM Sleep', 'notInBed', 'Awake']
                df_labeled = df_labeled[df_labeled['label'].isin(valid_states)]
                
                if not df_labeled.empty:
                    all_data.append(df_labeled)
        except Exception as e:
            print(f"Skipping {item}: {e}")

    if not all_data: return pd.DataFrame(), []
    return pd.concat(all_data, ignore_index=True), final_features

def main():
    # load and prepare dataset
    df_train, feature_cols = load_training_data(DATA_ROOT)
    
    if df_train.empty:
        print("No training data.")
        return

    print(f"\nTotal Training Samples: {len(df_train)}")
    X = df_train[feature_cols]
    y = df_train['label']

    # split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("\n--- Training Model with Advanced Features ---")
    
    # calculate weights to handle class imbalance
    weights = compute_sample_weight('balanced', y_train)
    
    # initialize and train classifier
    model = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=6, random_state=42)
    model.fit(X_train, y_train, sample_weight=weights)

    # save trained model and feature list
    joblib.dump(model, 'ML/state_2_model.joblib')
    joblib.dump(feature_cols, 'ML/state_2_features.joblib')

    # evaluate model performance
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nOverall Accuracy: {accuracy * 100:.2f}%")
    
    # display detailed classification report
    print("\nClassification Report:")
    unique_labels = sorted(y_test.unique())
    print(classification_report(y_test, y_pred, target_names=unique_labels))
    
    # display feature importance ranking
    print("\nTop 15 Feature Importances:")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]
    for i in indices:
        print(f"  {feature_cols[i]}: {importances[i] * 100:.2f}%")

if __name__ == "__main__":
    main()