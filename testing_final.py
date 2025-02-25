import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier
from joblib import load
from pynput import keyboard
import dlib
import numpy as np
import cv2
import pickle
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import logging

# Load the trained keystroke dynamics model
rf_model = load('Keystrokes/rf_model.joblib')

# Load user credentials
user_credentials = pd.read_csv('Keystrokes/user_credentials.csv')

# Load column names from training data for keystroke features
merged_df = pd.read_csv('Keystrokes/Key_pattern.csv')
feature_columns = merged_df.drop(columns=['User']).columns

# Dlib initialization for face detection
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('Facedetection/dlib/shape_predictor_68_face_landmarks.dat')
face_reco_model = dlib.face_recognition_model_v1("Facedetection/dlib/dlib_face_recognition_resnet_model_v1.dat")

# Initialize keystroke data collection
keystroke_data = []

# Listener callback functions for keystrokes
def on_press(key):
    try:
        if hasattr(key, 'char') and key.char:
            press_time = time.time()
            keystroke_data.append(('press', key.char, press_time))
    except AttributeError:
        pass

def on_release(key):
    try:
        if hasattr(key, 'char') and key.char:
            release_time = time.time()
            keystroke_data.append(('release', key.char, release_time))
    except AttributeError:
        pass

# Function to record keystroke timings for password
def record_keystrokes(password):
    global keystroke_data
    keystroke_data.clear()
    input(f"Please type the password '{password}' and press Enter when done.")
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.stop()

    timing_record = {}
    press_times = {}
    release_times = {}

    for event, char, timestamp in keystroke_data:
        if event == 'press':
            press_times[char] = timestamp
        elif event == 'release':
            release_times[char] = timestamp

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

# Function to send email notification
def send_email_notification(recipient_email, captured_image_path):
    # Define email sender and receiver
    sender_email = "" # place the server email_id "csprjt21eac@gmail.com"
    sender_password = "" # place teh emailid password " ganx jypg qokb ovjj"
    subject = "Unauthorized Access Attempt Detected"
    body = "An unauthorized person attempted to access your account. Please be aware of potential unauthorized access. Please find the captured image attached."

    # Setup the MIME
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject

    # Attach the body with the msg instance
    msg.attach(MIMEText(body, 'plain'))

    # Open the file to be sent
    with open(captured_image_path, "rb") as attachment:
        # Create a MIMEBase object
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f"attachment; filename= {os.path.basename(captured_image_path)}")

        # Attach the instance 'part' to instance 'msg'
        msg.attach(part)

    # Create SMTP session for sending the mail
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()  # Enable security
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, msg.as_string())
            print(f"Notification email sent to {recipient_email}")
    except Exception as e:
        print(f"Failed to send email. Error: {e}")

# Face recognition class
class FaceRecognizer:
    def __init__(self):
        self.font = cv2.FONT_ITALIC
        with open("Facedetection/model.pkl", "rb") as f:
            model_data = pickle.load(f)
            self.face_features_known_list = model_data["face_features"]
            self.face_name_known_list = model_data["face_names"]
        if not os.path.exists("captured_images"):
            os.makedirs("captured_images")

    @staticmethod
    def return_euclidean_distance(feature_1, feature_2):
        return np.sqrt(np.sum(np.square(np.array(feature_1) - np.array(feature_2))))

    def process(self, stream):
        start_time = time.time()
        while stream.isOpened():
            flag, img_rd = stream.read()
            if not flag:
                break

            faces = detector(img_rd, 0)
            for k, d in enumerate(faces):
                shape = predictor(img_rd, d)
                face_descriptor = face_reco_model.compute_face_descriptor(img_rd, shape)

                distances = [self.return_euclidean_distance(face_descriptor, known_features)
                             for known_features in self.face_features_known_list]
                min_distance_index = np.argmin(distances)

                if distances[min_distance_index] < 0.4:
                    recognized_name = os.path.splitext(self.face_name_known_list[min_distance_index])[0]
                    print(f"Predicted username by face recognition model: {recognized_name}")
                    cv2.putText(img_rd, recognized_name, (d.left(), d.top() - 10), self.font, 0.8, (0, 255, 0), 2)
                    cv2.rectangle(img_rd, (d.left(), d.top()), (d.right(), d.bottom()), (0, 255, 0), 2)
                    captured_image_path = os.path.join("captured_images", f"{recognized_name}_captured.jpg")
                    cv2.imwrite(captured_image_path, img_rd)
                    return recognized_name
                else:
                    print("Unknown face detected")
                    cv2.putText(img_rd, "Unknown", (d.left(), d.top() - 10), self.font, 0.8, (0, 0, 255), 2)
                    cv2.rectangle(img_rd, (d.left(), d.top()), (d.right(), d.bottom()), (0, 0, 255), 2)

            cv2.imshow("Camera", img_rd)
            if time.time() - start_time > 10:
                print("Time exceeded 30 seconds for face adjustment.")
                return None

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        return None

    def run(self, keystroke_username, user_email):
        cap = cv2.VideoCapture(0)
        unauthorized_image_path = None
        start_time = time.time()
        while time.time() - start_time <= 10:  # Allow a 30-second window for retries
            recognized_name = self.process(cap)
            if recognized_name:
                print(f"User authenticated with face recognition: {recognized_name}")
                if recognized_name == keystroke_username:
                    print("Face recognition matches keystroke authentication. Access granted.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return True
                else:
                    print("Mismatch between keystroke and face recognition usernames.")
                    unauthorized_image_path = os.path.join("captured_images", f"{recognized_name}_unauthorized.jpg")
                    cv2.imwrite(unauthorized_image_path, cap.read()[1])
                    print("Unauthorized access attempt detected.")
            else:
                print("Face not recognized. Please adjust your face position.")
        
        # After 30 seconds, send email if no successful match
        if unauthorized_image_path:
            send_email_notification(user_email, unauthorized_image_path)
            print("Email notification sent after 30 seconds due to failed face authentication.")
        
        print("Face recognition failed after multiple attempts within 30 seconds.")
        cap.release()
        cv2.destroyAllWindows()
        return False

# Main authentication flow
def main():
    logging.basicConfig(level=logging.INFO)
    
    username = input("Enter your username: ")
    password = input("Enter your password: ")

    if username in user_credentials['username'].values:
        stored_password = user_credentials.loc[user_credentials['username'] == username, 'password'].values[0]
        user_email = user_credentials.loc[user_credentials['username'] == username, 'email'].values[0]
        
        if password == stored_password:
            print("Password matched.")
            keystroke_features = record_keystrokes("CS@prjt21")
            keystroke_features_row = [keystroke_features.get(col, 0) for col in feature_columns]
            keystroke_df = pd.DataFrame([keystroke_features_row], columns=feature_columns)
            predicted_username = rf_model.predict(keystroke_df)[0]
            print(f"Predicted username by keystroke dynamics model: {predicted_username}")
            if predicted_username == username:
                print("User authenticated successfully via keystrokes.")
                face_recognizer = FaceRecognizer()
                if face_recognizer.run(predicted_username, user_email):
                    print("Final authentication successful.")
                else:
                    print("Final authentication failed.")
            else:
                print("Keystroke prediction mismatch. Authentication failed.")
        else:
            print("Incorrect password. Authentication failed.")
    else:
        print("Username not found. Authentication failed.")

if __name__ == '__main__':
    main()
