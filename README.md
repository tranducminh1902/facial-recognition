# facial-recognition
Using YOLOv3 for detection and custom 3 classes trained mobilenetv2 model for prediction.
Build an application that can detect the faces of members in my team using OpenCV and YOLO

Locate and recognize human face using YOLO and OpenCV

• Download YOLO model and weight at: https://drive.google.com/drive/folders/1QO9ydq_cUHlfpKK78DSUymHii5aix2jf?usp=sharing

• Store the files in folder name yolo in the main directory.

• To run detection run:

python face_recog_app.py
• To collect data images by capturing from webcam, create a folder name data in the model directory and run:

python data_creator.py
Captured images will be saved in the data folder with the target name as its directory.

• To train and export classifier, use the model_1.h5
