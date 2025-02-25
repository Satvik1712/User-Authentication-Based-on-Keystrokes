Title: Multi-Modal User Authentication: Integrating Keystroke Dynamics and Facial Recognition.


This project has 2 parts authentication based on key strokes and face detection.
The two parts are kept in 2 separate folders.

Keystrokes Folder:

1) collection.py => to collect the keystrokes and create user_credential.csv and key_pattern.csv files.
2) user_credential.csv => contains the credentials.
3) key_pattern.csv => contains keystroke data of each user typing "CS@prjt21"
4) DSL-StrongPasswordData.csv => Benchmark dataset of CMU school of computer science.
5) Model_Own_dataset.ipynb => code for models using own dataset.
6) Model_Online_dataset.ipynb => code for models using Bench mark dataset.
7) rf_model.joblib => model with highest accuracy is saved for testing
8) testing_keystrokes.py => code to test only keystrokes.
9) Results.xlsv => Results are written in this xl sheet


Face Detection Folder:

1) data => Image of the people used for training.
2) dlib => pretrained model to find the face shape and discriminators.
3) test_data => Image of the people used for testing.
4) features_extraction_to_csv.py => code to extract the features.
5) features_all.csv => Extracted features.
6) savemodel.py => Training and saving model.
7) model.pkl => Saved model for testing.
8) evaluation.ipynb => code for the evaluation of model using the test data.
9) testing_facerecognization.py => code to test only face recognition.


testing_final.py => code to test the final integrated model.

Order to run code:

1) Key stroke Folder: collection.py -> Model_Own_dataset.ipynb -> testing_keystrokes.py

2) Face Detection Folder:features_extraction_to_csv.py -> savemodel.py -> evaluation.ipynb -> testing_facerecognization.py

3) testing_final.py


https://ieeexplore.ieee.org/document/10859646

T. S. Gupta, K. P. Karthik, S. S. Suhas Sanisetty and S. Basavaraju, "An Ensemble Model for User Authentication Leveraging Keystroke Dynamics and Facial Recognition," 2024 9th International Conference on Communication and Electronics Systems (ICCES), Coimbatore, India, 2024, pp. 921-926, doi: 10.1109/ICCES63552.2024.10859646.