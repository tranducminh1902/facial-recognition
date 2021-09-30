# Facial Recognition
Using YOLOv3 for detection and custom 3 classes trained mobilenetv2 model for prediction.

OpenCV and streamlit was used to deploy using webcam.

## The dataset for training:
The dataset includes images of team members, captured by webcam using [data_creator.py](https://github.com/tranducminh1902/facial-recognition/blob/main/data_creator.py).

Dataset has been put in structural folders with each folder name as label (name of team member):
- long: 597 images
- minh: 669 images
- tung: 507 images

Link for the dataset: [images_260921.zip](https://drive.google.com/file/d/1ESfbGrmAJxO5kLWmIENmFPW44ZukXJeC/view?usp=sharing)

## Training the model:
MobilenetV2 model was used as based model and customized classification dense layers to predict the team members.

The model was trained using [Train_model.ipynb](https://github.com/tranducminh1902/facial-recognition/blob/main/Train_model.ipynb)

The best model was exported to .h5 file for deployment: [model_1.h5](https://github.com/tranducminh1902/facial-recognition/blob/main/prediction_model/model_1.h5)

## Deployment:
Streamlit was used to deploy the file: [face_recog_app.py](https://github.com/tranducminh1902/facial-recognition/blob/main/face_recog_app.py)
- Download YOLOv3 model and weight at: https://drive.google.com/drive/folders/1QO9ydq_cUHlfpKK78DSUymHii5aix2jf?usp=sharing
- Load YOLOv3 for face detection using weights and config files
- Load trained CNN model for face prediction
- Create bounding box around each face detected and show name with confidence level (in probability)

## Contributions and References:
Contributions:
- Minh Tran: https://github.com/tranducminh1902
- Long Nguyen: https://github.com/longnguyentruong0607
- Vu Thanh Tung: https://github.com/tung151078

References:
- Creating dataset: https://github.com/murtazahassan/OpenCV-Python-Tutorials-and-Projects/blob/master/Intermediate/Custom%20Object%20Detection/createData.py
