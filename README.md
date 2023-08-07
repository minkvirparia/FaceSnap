# FaceSnap - Face Recoginiton System
FaceSnap is a face recognition system that consists of three main components: dataset generation, training using LBPHFaceRecognizer, and face recognition. The system utilizes the haarcascade_frontalface_default for face detection and recognition.

# Installation
**pip install opencv-python numpy Pillow**

This will install the following libraries:

opencv-python: OpenCV for computer vision tasks, including face detection and recognition.
numpy: NumPy for numerical operations and data handling.
Pillow: Python Imaging Library for working with images.

# Usage

## Dataset Generation
Run the dataset_generator.py script to generate a dataset of face images. This involves capturing multiple images of each individual's face to create a diverse set of training data.

**python dataset_generator.py**

## Training
Execute the trainer.py script to train the model using the LBPHFaceRecognizer algorithm. This step processes the dataset generated in the previous step and creates a trained model for face recognition.

**python trainer.py**

## Face Recognition
Run the face_recognizer.py script to perform real-time face recognition using the trained model. The system captures video frames from the camera and attempts to recognize faces present in the frame.

**python face_recognizer.py**

# Note
1. Change the path in each files according to your system path.
2. Put all these files in same folder with haarcascade_frontalface_default.xml.








