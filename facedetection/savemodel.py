import dlib
import numpy as np
import cv2
import os
import pandas as pd
import time
import logging
import pickle

# Dlib / Use frontal face detector of Dlib
detector = dlib.get_frontal_face_detector()

# Dlib landmark / Get face landmarks
predictor = dlib.shape_predictor('dlib/shape_predictor_68_face_landmarks.dat')

# Dlib Resnet / Use Dlib resnet50 model to get 128D face descriptor
face_reco_model = dlib.face_recognition_model_v1("dlib/dlib_face_recognition_resnet_model_v1.dat")


class FaceRecognizer:
    def __init__(self):
        self.font = cv2.FONT_ITALIC

        # Variables for frame and FPS calculation
        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0
        self.start_time = time.time()

        # Load known faces and names
        self.face_features_known_list = []
        self.face_name_known_list = []

        # Create captured_images directory if it doesn't exist
        if not os.path.exists("captured_images"):
            os.makedirs("captured_images")

    # Load known faces from "features_all.csv" and save as a model
    def create_and_save_model(self):
        if os.path.exists("features_all.csv"):
            path_features_known_csv = "features_all.csv"
            csv_rd = pd.read_csv(path_features_known_csv, header=None)
            for i in range(csv_rd.shape[0]):
                features_someone_arr = []
                self.face_name_known_list.append(csv_rd.iloc[i][0])
                for j in range(1, 129):
                    if csv_rd.iloc[i][j] == '':
                        features_someone_arr.append('0')
                    else:
                        features_someone_arr.append(float(csv_rd.iloc[i][j]))
                self.face_features_known_list.append(features_someone_arr)
            logging.info("Faces in Database: %d", len(self.face_features_known_list))

            # Save the model to a pickle file
            model_data = {
                "face_features": self.face_features_known_list,
                "face_names": self.face_name_known_list
            }
            with open("model.pkl", "wb") as f:
                pickle.dump(model_data, f)
            print("Model saved as 'model.pkl'")
            return True
        else:
            logging.warning("'features_all.csv' not found!")
            return False


def main():
    logging.basicConfig(level=logging.INFO)
    face_recognizer = FaceRecognizer()
    face_recognizer.create_and_save_model()


if __name__ == '__main__':
    main()
