import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from joblib import load
from pynput import keyboard # Requires the 'keyboard' library for keystroke capture
import getpass  # For silent password-like input

keystroke_data = []

# Load the trained model
rf_model = load('rf_model.joblib')

# Load user credentials
user_credentials = pd.read_csv('user_credentials.csv')

#  Listener callback functions
def on_press(key):
    try:
        # Check for valid character press and store the press time
        if hasattr(key, 'char') and key.char:
            press_time = time.time()
            keystroke_data.append(('press', key.char, press_time))
    except AttributeError:
        pass  # Ignore for special keys

def on_release(key):
    try:
        # Check for valid character release and store the release time
        if hasattr(key, 'char') and key.char:
            release_time = time.time()
            keystroke_data.append(('release', key.char, release_time))
    except AttributeError:
        pass

# Function to record keystroke timings for a specific string without retaining display
def record_keystrokes(password):
    global keystroke_data
    # print(f"Please type the string '{password}' and press Enter when done:")
    all_timings = []
    # input(f"Press Enter to start typing.")
    keystroke_data.clear()  # Clear previous attempts
    # Start capturing keystrokes
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        input(f"Please type the password '{password}' and press Enter when done.")
        listener.stop()

    # Calculate Hold, DD, and UD timings for each attempt
    timing_record = {}
    press_times = {}
    release_times = {}

    # Collect press and release times for each character
    for event, char, timestamp in keystroke_data:
        if event == 'press':
            press_times[char] = timestamp
        elif event == 'release':
            release_times[char] = timestamp

    # Calculate Hold time (H), Keydown-Keydown (DD), and Keyup-Keydown (UD) times
    for i in range(len(password)):
        char = password[i]
        if char in press_times and char in release_times:
            timing_record[f"H.char{i+1}"] = round(release_times[char] - press_times[char], 4)
        else:
            timing_record[f"H.char{i+1}"] = 0

    for i in range(len(password) - 1):
        if password[i] in press_times and password[i+1] in press_times:
            timing_record[f"DD.char{i+1}.char{i+2}"] = round(press_times[password[i+1]] - press_times[password[i]], 4)
        else:
            timing_record[f"DD.char{i+1}.char{i+2}"] = 0

        if password[i] in release_times and password[i+1] in press_times:
            timing_record[f"UD.char{i+1}.char{i+2}"] = round(press_times[password[i+1]] - release_times[password[i]], 4)
        else:
            timing_record[f"UD.char{i+1}.char{i+2}"] = 0
    
    return timing_record

# Load column names from training data (to ensure correct format for model prediction)
merged_df = pd.read_csv('Key_pattern.csv')
feature_columns = merged_df.drop(columns=['User']).columns

# Get username and password input
username = input("Enter your username: ")
password = input("Enter your password: ")

# Check if the username exists in the credentials file
if username in user_credentials['username'].values:
    stored_password = user_credentials.loc[user_credentials['username'] == username, 'password'].values[0]

    # Verify if the password matches the stored password
    if password == stored_password:
        print("Password matched.")

        # Record keystrokes for the designated string
        keystroke_features = record_keystrokes("CS@prjt21")

        # Convert the timing data to a DataFrame row
        keystroke_features_row = [keystroke_features.get(col, 0) for col in feature_columns]
        keystroke_df = pd.DataFrame([keystroke_features_row], columns=feature_columns)

        # Predict the username using the model
        predicted_username = rf_model.predict(keystroke_df)[0]
        print(f"Predicted username by the model: {predicted_username}")

        # Verify keystroke-based authentication
        if predicted_username == username:
            print("User authenticated successfully.")
        else:
            print("Username prediction does not match. Authentication failed.")
    else:
        print("Incorrect password. Authentication failed.")
else:
    print("Username not found. Authentication failed.")
