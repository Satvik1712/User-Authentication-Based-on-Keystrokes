import csv
import time
from getpass import getpass
from pynput import keyboard

# Keystroke dynamics storage (to capture key press and release events)
keystroke_data = []
user_password = ""

# Listener callback functions
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

# Function to record keystroke timings for the user-defined password
def record_keystroke_timings(password):
    global keystroke_data
    print(f"\nPlease type the password '{password}' 8 times for keystroke dynamics recording.\n")
    all_timings = []

    for attempt in range(8):
        input(f"Attempt {attempt + 1}: Press Enter to start typing.")
        
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

        all_timings.append(timing_record)

    return all_timings

# Function to write credentials to a CSV file
def save_credentials(username, password):
    with open("user_credentials.csv", mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write header if the file is empty
        if file.tell() == 0:
            writer.writerow(["username", "password"])
        writer.writerow([username, password])

# Function to save keystroke dynamics to a CSV file
def save_keystroke_dynamics(username, keystroke_data, filename="keystroke_dynamics.csv"):
    fieldnames = ["User"]
    
    # Dynamically generate fieldnames based on the password structure
    if keystroke_data:
        first_record = keystroke_data[0]
        for key in first_record.keys():
            fieldnames.append(key)
    
    with open(filename, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        # Write the header once if the file is empty
        if file.tell() == 0:
            writer.writeheader()
        
        # Write the rows
        for record in keystroke_data:
            record["User"] = username  # Append username to each record
            writer.writerow(record)

# Function to capture keystroke dynamics for the word CS@prjt21 and save to key_pattern.csv
def capture_key_pattern(username):
    global keystroke_data
    key_pattern = "CS@prjt21"
    
    print(f"\nPlease type the word '{key_pattern}' 8 times to capture its keystroke dynamics.\n")
    
    # Record keystroke dynamics for the fixed word
    keystroke_data.clear()  # Clear previous data
    
    keystroke_data = record_keystroke_timings(key_pattern)
    save_keystroke_dynamics(username, keystroke_data, filename="key_pattern.csv")
    print(f"Keystroke dynamics for the word '{key_pattern}' saved to key_pattern.csv.")

# Main function to get user credentials and keystroke dynamics
def main():
    global user_password
    
    print("Welcome to the Keystroke Dynamics Authentication System")
    username = input("Enter a username: ")
    
    # Ensure the user creates a password of exactly 8 characters
    while True:
        user_password = getpass("Enter a password (8 characters): ")
        if len(user_password) == 8:
            break
        else:
            print("Please enter a password of exactly 8 characters.")
    
    # Save credentials in user_credentials.csv
    save_credentials(username, user_password)
    print(f"Credentials for {username} saved.")
    
    # Capture and save keystroke dynamics based on the user's created password
    keystroke_data = record_keystroke_timings(user_password)
    save_keystroke_dynamics(username, keystroke_data)
    print(f"Keystroke dynamics for {username} saved.")

    # Capture keystroke dynamics for the word "CS@prjt21"
    capture_key_pattern(username)

if __name__ == "__main__":
    main()
