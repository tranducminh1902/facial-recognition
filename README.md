# Facial Recognition
Using YOLOv3 for detection and custom 3 classes trained mobilenetv2 model from tensorflow for prediction.

OpenCV and streamlit was used to deploy using webcam.

## The dataset for training:
The dataset includes images of team members, captured by webcam using [data_creator.py](https://github.com/tranducminh1902/facial-recognition/blob/main/data_creator.py).

Each image using YOLO to crop to take only the face area.

![3_1354_1632590101 9486406](https://user-images.githubusercontent.com/86507088/135449194-01263d60-cb74-4fa2-a741-432f935b7d68.png)
![12_1292_1632590124 0402558](https://user-images.githubusercontent.com/86507088/135449214-5c0c1ee0-8655-485f-8186-95b120bee9e0.png)
![3_179_1632475036 7965965](https://user-images.githubusercontent.com/86507088/135449731-b8cd61bf-1464-4d5a-b938-3d7c05ff359b.png)
![14_221_1632476223 0819051](https://user-images.githubusercontent.com/86507088/135449745-4945a282-b575-4c65-b319-9118b44d9198.png)
![9_220_1632631786 335681](https://user-images.githubusercontent.com/86507088/135449570-2242dba3-3cf7-4c07-bcdd-e02e2d09e7b6.png)
![35_171_1632476931 8139071](https://user-images.githubusercontent.com/86507088/135449623-084ec616-0206-4c34-8371-21af85381568.png)

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

![SmartSelect_20211001-120313_Gallery](https://user-images.githubusercontent.com/86507088/135598198-c13c34f0-4f59-4e69-8463-885df7e48384.gif)

## Contributions and References:
Contributions:
- Minh Tran: https://github.com/tranducminh1902
- Long Nguyen: https://github.com/longnguyentruong0607
- Vu Thanh Tung: https://github.com/tung151078

References:
- Creating dataset: https://github.com/murtazahassan/OpenCV-Python-Tutorials-and-Projects/blob/master/Intermediate/Custom%20Object%20Detection/createData.py
