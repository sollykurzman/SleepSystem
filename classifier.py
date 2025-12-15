#!/usr/bin/python3

import time
import os
import threading
from datetime import datetime
from collections import deque
import queue
import pandas as pd
import joblib
import numpy as np

import store
from util_format import process_window, add_history_features
from util_train2 import add_advanced_features 

# configuration settings
ML_INTERVAL = 2.0
MODEL_PATH = "ML/"
WINDOW_SECONDS = 30
BUFFER_SIZE = 60 

# load in-bed detection model and encoder
in_bed_model = joblib.load(MODEL_PATH + "in_bed_model.joblib")
in_bed_encoder = joblib.load(MODEL_PATH + "in_bed_encoder.joblib")

# load asleep/awake detection model and encoder
asleep_model = joblib.load(MODEL_PATH + "asleep_model.joblib")
asleep_encoder = joblib.load(MODEL_PATH + "asleep_encoder.joblib")

# load base sleep state model and encoder
state_model = joblib.load(MODEL_PATH + "state_model.joblib")
state_encoder = joblib.load(MODEL_PATH + "state_encoder.joblib")

# attempt to load refined secondary model
try:
    state_2_model = joblib.load(MODEL_PATH + "state_2_model.joblib")
    state_2_features = joblib.load(MODEL_PATH + "state_2_features.joblib")
except Exception as e:
    print(f"Warning: Could not load refined model ({e}). Ensure USE_REFINED_MODEL is False.")
    state_2_model = None
    state_2_features = []

# thread-safe buffer for feature history
class HistoryBuffer:
    def __init__(self, max_length):
        self.buffer = deque(maxlen=max_length)
        self.lock = threading.Lock()

    def add(self, x):
        with self.lock:
            self.buffer.append(x)

    def get(self):
        with self.lock:
            return list(self.buffer)

# initialize buffers
history_buffer = HistoryBuffer(max_length=12)
probability_buffer = deque(maxlen=BUFFER_SIZE)
classification_queue = queue.Queue()

def predict_with_model(model, encoder, X):
    # filter input for required features
    required = model.feature_names_in_
    X = X[required]

    # get prediction label
    label_idx = model.predict(X)[0]
    label = encoder.inverse_transform([label_idx])[0]

    # get prediction probabilities
    proba = model.predict_proba(X)[0]
    classes = encoder.inverse_transform(model.classes_)

    # map classes to probabilities
    probabilities = {
        cls: float(p) for cls, p in zip(classes, proba)
    }

    return label, probabilities

def flatten_probabilities(ib_proba, slp_proba, state_proba):
    # initialize flat dictionary
    flat = {}
    flat['in_bed__inBed'] = ib_proba.get('inBed', 0.0)
    flat['asleep__Asleep'] = slp_proba.get('Asleep', 0.0)
    flat['asleep__Awake'] = slp_proba.get('Awake', 0.0)
    flat['state__Core Sleep'] = state_proba.get('Core Sleep', 0.0)
    flat['state__Deep Sleep'] = state_proba.get('Deep Sleep', 0.0)
    
    # normalize naming for REM sleep
    rem_val = state_proba.get('REM Sleep', state_proba.get('REM', 0.0))
    flat['state__REM Sleep'] = rem_val 
    return flat

def classify_snippet(snippet, use_refined=True):
    # drop timestamp for prediction
    X = snippet.drop(columns=["timestamp"])

    # run base models for bed, sleep, and state
    ib_label, ib_proba = predict_with_model(in_bed_model, in_bed_encoder, X)
    slp_label, slp_proba = predict_with_model(asleep_model, asleep_encoder, X)
    state_label_base, state_proba_base = predict_with_model(state_model, state_encoder, X)

    # store probability history for refined model
    flat_probs = flatten_probabilities(ib_proba, slp_proba, state_proba_base)
    probability_buffer.append(flat_probs)

    # return immediately if user is not in bed
    if ib_label != "inBed":
        return ib_label, {"in_bed": ib_proba}

    # return immediately if user is awake
    if slp_label != "Asleep":
        return slp_label, {
            "in_bed": ib_proba,
            "asleep": slp_proba
        }

    # apply refined model if enabled and available
    if use_refined and state_2_model is not None:
        try:
            # convert buffer to dataframe
            df_probs = pd.DataFrame(list(probability_buffer))
            
            # generate advanced features from probability history
            df_refined, _ = add_advanced_features(df_probs)
            
            # align features with model expectations
            X_refined = df_refined.iloc[[-1]].reindex(columns=state_2_features, fill_value=0)
            
            # predict using refined model
            final_label = state_2_model.predict(X_refined)[0]
            final_proba_array = state_2_model.predict_proba(X_refined)[0]
            
            # format final probabilities
            state_2_probs_dict = {cls: float(p) for cls, p in zip(state_2_model.classes_, final_proba_array)}

            #assign in-bed probabilities
            p_not_in_bed = state_2_probs_dict.get('notInBed', 0.0)
            refined_ib_proba = {
                "notInBed": p_not_in_bed,
                "inBed": 1.0 - p_not_in_bed
            }

            #assign asleep probabilities
            p_awake = state_2_probs_dict.get('Awake', 0.0)
            p_asleep = (
                state_2_probs_dict.get('Core Sleep', 0.0) + 
                state_2_probs_dict.get('Deep Sleep', 0.0) + 
                state_2_probs_dict.get('REM Sleep', 0.0)
            )

            refined_slp_proba = {
                "Awake": p_awake,
                "Asleep": p_asleep
            }
            
            #return state and probabilities
            return final_label, {
                "in_bed": refined_ib_proba,
                "asleep": refined_slp_proba,
                "state": state_2_probs_dict 
            }
        except Exception as e:
            # ignore errors and fallback to base model
            pass

    # fallback to base model result
    return state_label_base, {
        "in_bed": ib_proba,
        "asleep": slp_proba,
        "state": state_proba_base 
    }

def write_classification_row(night_id, result, csv_path=None):
    # set default path if none provided
    if csv_path is None:
        csv_path = f"Data/{night_id}/classification-{night_id}.csv"

    # define columns structure
    ALL_COLUMNS = {
        "in_bed": ["inBed", "notInBed"],
        "asleep": ["Asleep", "Awake"],
        "state": ["Core Sleep", "REM Sleep", "Deep Sleep"]
    }

    # prepare row data
    row = {
        "datetime": result["datetime"],
        "classification": result["classification"]
    }

    # flatten nested probabilities into columns
    for stage, classes in ALL_COLUMNS.items():
        probs = result["probabilities"].get(stage, {})
        for cls in classes:
            row[f"{stage}__{cls}"] = probs.get(cls, None)

    # create dataframe
    df_out = pd.DataFrame([row])
    
    # check if header is needed
    write_header = not os.path.exists(csv_path)

    # append to csv
    df_out.to_csv(csv_path, mode="a", header=write_header, index=False)

def classify_once(night_id=None, write=True, timestamp=None, use_refined=True):
    # get latest data snapshot
    times, volts = store.live_buffer.get_snapshot()
    if times is None:
        return None

    df = pd.DataFrame({
        "datetime": times,
        "voltage": volts
    })

    # process raw data into feature window
    snippet = process_window(
        df["datetime"].values,
        df["voltage"].values,
        df["datetime"].iloc[0],
        WINDOW_SECONDS
    )

    if snippet is None:
        return None

    # update history buffer
    history = history_buffer.get()
    history_buffer.add(snippet)

    # combine history and current snippet
    snippet = pd.DataFrame(history + [snippet])
    snippet = snippet.drop(columns=["sleep_state"], errors="ignore")
    
    # calculate history-based features
    snippet = add_history_features(snippet)

    # run classification logic
    label, probabilities = classify_snippet(snippet, use_refined=use_refined)

    if timestamp is None:
        timestamp = datetime.now()

    # structure the result
    result = {
        "datetime": timestamp,
        "classification": label,
        "probabilities": probabilities
    }

    # push result to queue
    classification_queue.put(result)

    # save to disk if required
    if night_id and write:
        write_classification_row(night_id, result)

    return result

def classification_worker(night_id=None, until=None, use_refined=True):
    print("Classification worker started")

    # loop until end time reached
    while until is None or datetime.now() < until:
        time.sleep(ML_INTERVAL)
        classify_once(night_id, use_refined=use_refined)

def start_classifier(night_id=None, until=None):
    # spawn worker thread
    t = threading.Thread(
        target=classification_worker,
        args=(night_id, until),
        daemon=True
    )
    t.start()

def reset_state():
    # clear all buffers and queues
    history_buffer.buffer.clear()
    probability_buffer.clear()
    with classification_queue.mutex:
        classification_queue.queue.clear()
    print("Classifier state reset.")