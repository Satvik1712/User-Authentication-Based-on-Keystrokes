# Extract features from images and save into "features_all.csv"

import os
import dlib
import csv
import numpy as np
import logging
import cv2

#  Path of cropped faces
path_images_from_camera = "data/"

#  Use frontal face detector of Dlib
detector = dlib.get_frontal_face_detector()

#  Get face landmarks
predictor = dlib.shape_predictor('dlib/shape_predictor_68_face_landmarks.dat')

#  Use Dlib resnet50 model to get 128D face descriptor
face_reco_model = dlib.face_recognition_model_v1("dlib/dlib_face_recognition_resnet_model_v1.dat")


#  Return 128D features for single image

def return_128d_features(path_img):
    img_rd = cv2.imread(path_img)
    
    # Check if the image was loaded successfully
    if img_rd is None:
        logging.error("Image not loaded correctly: %s", path_img)
        return 0
    
    faces = detector(img_rd, 1)
    logging.info("%-40s %-20s", "Image with faces detected:", path_img)

    # For photos of faces saved, we need to make sure that we can detect faces from the cropped images
    if len(faces) != 0:
        shape = predictor(img_rd, faces[0])
        face_descriptor = face_reco_model.compute_face_descriptor(img_rd, shape)
    else:
        face_descriptor = 0
        logging.warning("No face detected in %s", path_img)
    
    return face_descriptor



#   Return the mean value of 128D face descriptor for person X

def return_features_mean_personX(path_face_personX):
    features_list_personX = []
    photos_list = os.listdir(path_face_personX)
    if photos_list:
        for i in range(len(photos_list)):
            #  return_128d_features()  128D  / Get 128D features for single image of personX
            logging.info("%-40s %-20s", " / Reading image:", path_face_personX + "/" + photos_list[i])
            features_128d = return_128d_features(path_face_personX + "/" + photos_list[i])
            #  Jump if no face detected from image
            if features_128d == 0:
                i += 1
            else:
                features_list_personX.append(features_128d)
    else:
        logging.warning(" Warning: No images in%s/", path_face_personX)

   
    if features_list_personX:
        features_mean_personX = np.array(features_list_personX, dtype=object).mean(axis=0)
    else:
        features_mean_personX = np.zeros(128, dtype=object, order='C')
    return features_mean_personX


def main():
    logging.basicConfig(level=logging.INFO)
    # Get the list of image files in the data directory
    image_files = [f for f in os.listdir("data/") if f.endswith('.jpg')]

    with open("features_all.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for image_file in image_files:
            image_path = os.path.join(path_images_from_camera, image_file)
            logging.info("Processing %s", image_path)
            
            # Extract 128D features for the image
            features_128d = return_128d_features(image_path)
            
            # Skip if no face was detected
            if features_128d == 0:
                logging.warning("No face detected in %s", image_file)
                continue
            
            # Convert features_128d to a list and prepend the filename
            features_128d = [image_file] + list(features_128d)
            writer.writerow(features_128d)
            logging.info("Saved features for %s", image_file)

    logging.info("Saved all the features of faces into: features_all.csv")



if __name__ == '__main__':
    main()