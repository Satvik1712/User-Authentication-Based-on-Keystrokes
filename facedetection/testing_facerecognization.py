import dlib
import numpy as np
import cv2
import pickle
import os
import logging

# Dlib / Use frontal face detector of Dlib
detector = dlib.get_frontal_face_detector()

# Dlib landmark / Get face landmarks
predictor = dlib.shape_predictor('dlib/shape_predictor_68_face_landmarks.dat')

# Dlib Resnet / Use Dlib resnet50 model to get 128D face descriptor
face_reco_model = dlib.face_recognition_model_v1("dlib/dlib_face_recognition_resnet_model_v1.dat")


class FaceRecognizer:
    def __init__(self):
        self.font = cv2.FONT_ITALIC

        # Load model from file
        with open("model.pkl", "rb") as f:
            model_data = pickle.load(f)
            self.face_features_known_list = model_data["face_features"]
            self.face_name_known_list = model_data["face_names"]

        # Create captured_images directory if it doesn't exist
        if not os.path.exists("captured_images"):
            os.makedirs("captured_images")

    @staticmethod
    # / Compute the Euclidean distance between two 128D features
    def return_euclidean_distance(feature_1, feature_2):
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        return dist

    # Face detection and recognition from input video stream
    def process(self, stream):
        while stream.isOpened():
            flag, img_rd = stream.read()
            if not flag:
                break

            # Detect faces in the current frame
            faces = detector(img_rd, 0)

            # Process detected faces
            for k, d in enumerate(faces):
                shape = predictor(img_rd, d)
                face_descriptor = face_reco_model.compute_face_descriptor(img_rd, shape)

                # Compare with known faces in the model
                distances = []
                for known_features in self.face_features_known_list:
                    e_distance = self.return_euclidean_distance(face_descriptor, known_features)
                    distances.append(e_distance)

                # Find the recognized face with the minimum distance
                min_distance_index = np.argmin(distances)
                if distances[min_distance_index] < 0.4:
                    # Remove the file extension (.jpg) from the recognized name
                    recognized_name = os.path.splitext(self.face_name_known_list[min_distance_index])[0]
                    print(f"Recognized: {recognized_name}")

                    # Display the recognized name on the frame
                    cv2.putText(img_rd, recognized_name, (d.left(), d.top() - 10), self.font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.rectangle(img_rd, (d.left(), d.top()), (d.right(), d.bottom()), (0, 255, 0), 2)

                    # Save the captured image
                    captured_image_path = os.path.join("captured_images", f"{recognized_name}_captured.jpg")
                    cv2.imwrite(captured_image_path, img_rd)
                    print(f"Captured image saved at: {captured_image_path}")

                    # Stop processing after recognizing and saving one face
                    stream.release()
                    cv2.destroyAllWindows()
                    return recognized_name

                else:
                    print("Unknown person detected.")
                    cv2.putText(img_rd, "Unknown", (d.left(), d.top() - 10), self.font, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.rectangle(img_rd, (d.left(), d.top()), (d.right(), d.bottom()), (0, 0, 255), 2)

            # Show the frame
            cv2.imshow("camera", img_rd)

            # Press 'q' to exit manually
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the video stream and close windows
        stream.release()
        cv2.destroyAllWindows()
        return None

    def run(self):
        cap = cv2.VideoCapture(0)  # Use this for live camera feed
        recognized_name = self.process(cap)
        if recognized_name:
            print(f"Face recognized: {recognized_name}")
        else:
            print("No face recognized.")


def main():
    logging.basicConfig(level=logging.INFO)
    face_recognizer = FaceRecognizer()
    face_recognizer.run()


if __name__ == '__main__':
    main()
