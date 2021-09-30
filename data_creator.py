import cv2
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt




#####################################################

myPath = 'images' # Rasbperry Pi:  '/home/pi/Desktop/data/images'
cameraNo = 0
cameraBrightness = 40
moduleVal = 2  # SAVE EVERY ITH FRAME TO AVOID REPETITION
minBlur = 150  # SMALLER VALUE MEANS MORE BLURRINESS PRESENT
grayImage = False # IMAGES SAVED COLORED OR GRAY
saveData = True   # SAVE DATA FLAG
showImage = True  # IMAGE DISPLAY FLAG
imgWidth = 300
imgHeight = 300


#####################################################

global countFolder
# cap = cv2.VideoCapture(cameraNo)



count = 0
countSave =0

def saveDataFunc():
    global countFolder
    countFolder = 0
    while os.path.exists( myPath+ str(countFolder)):
        countFolder += 1
    os.makedirs(myPath + str(countFolder))

if saveData:saveDataFunc()



MODEL = 'yolov3-face.cfg'
WEIGHT = 'yolov3-wider_16000.weights'

net = cv2.dnn.readNetFromDarknet(MODEL, WEIGHT)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

print('ok')

cap = cv2.VideoCapture(0)

cap.set(3, 640)
cap.set(4, 480)
cap.set(10,cameraBrightness)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")



while True:
    ret, frame = cap.read()
################################################

    IMG_WIDTH, IMG_HEIGHT = 416, 416

    # Making blob object from original image
    blob = cv2.dnn.blobFromImage(frame, 
                                1./255, (IMG_WIDTH, IMG_HEIGHT),
                                [0, 0, 0], 1, crop=False)

    # Set model input
    net.setInput(blob)

    # Define the layers that we want to get the outputs from
    output_layers = net.getUnconnectedOutLayersNames()

    # Run 'prediction'
    outs = net.forward(output_layers)
    print('outs shape =',len(outs))
    print(len(outs[0]))
    print(outs[0])


    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only
    # the ones with high confidence scores. Assign the box's class label as the
    # class with the highest score.

    confidences = []
    boxes = []

    # Each frame produces 3 outs corresponding to 3 output layers
    for out in outs:
    # One out has multiple predictions for multiple captured objects.
        for detection in out:
            confidence = detection[-1]
        # Extract position data of face area (only area with high confidence)
            if  confidence > 0.5:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)

                            # Find the top left point of the bounding box 
                topleft_x = int(center_x - width/2)
                topleft_y = int(center_y - height/2)
                confidences.append(float(confidence))
                boxes.append([topleft_x, topleft_y, width, height])

    # Perform non-maximum suppression to eliminate 
    # redundant overlapping boxes with lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)


    result = frame.copy()
    final_boxes = []

    for i in indices:
        i = i[0]
        box = boxes[i]
        final_boxes.append(box)

        # Extract position data
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]

        # Draw bouding box with the above measurements
        ### YOUR CODE HERE
        top_left = (left,top)
        bottom_right = (left + width, top + height)
        # result = cv2.rectangle(frame,top_left,bottom_right,(255,0,0),1)
        imcrop = frame[top:top+height,left:left+width]

        
        if grayImage:imcrop = cv2.cvtColor(imcrop,cv2.COLOR_BGR2GRAY)
        if saveData:
            blur = cv2.Laplacian(imcrop, cv2.CV_64F).var()
            if count % moduleVal ==0 and blur > minBlur:
                nowTime = time.time()
                cv2.imwrite(myPath + str(countFolder) +
                        '/' + str(countSave)+"_"+ str(int(blur))+"_"+str(nowTime)+".png", imcrop)
                countSave+=1
            count += 1

    if showImage:
        cv2.imshow("Image", imcrop)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()