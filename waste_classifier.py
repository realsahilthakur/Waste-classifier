import os
import cvzone
from cvzone.ClassificationModule import Classifier
import cv2
import numpy as np

# Initialize video capture from webcam
cap = cv2.VideoCapture(0)

# Initialize the classifier with model and labels
classifier = Classifier('Resources/Model/keras_model.h5', 'Resources/Model/labels.txt')

# Load the arrow image
imgArrow = cv2.imread('Resources/arrow.png', cv2.IMREAD_UNCHANGED)

# Initialize the classIDBin variable
classIDBin = 0

# Load all waste images
imgWasteList = []
pathFolderWaste = "Resources/Waste"
pathList = os.listdir(pathFolderWaste)
for path in pathList:
    img = cv2.imread(os.path.join(pathFolderWaste, path), cv2.IMREAD_UNCHANGED)
    if img is not None:
        imgWasteList.append(img)

# Load all bin images
imgBinsList = []
pathFolderBins = "Resources/Bins"
pathList = os.listdir(pathFolderBins)
for path in pathList:
    img = cv2.imread(os.path.join(pathFolderBins, path), cv2.IMREAD_UNCHANGED)
    if img is not None:
        imgBinsList.append(img)

# Dictionary for class to bin mapping
classDic = {
    0: None,
    1: 0,
    2: 0,
    3: 3,
    4: 3,
    5: 1,
    6: 1,
    7: 2,
    8: 2,
}

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break
    
    # Check if the camera is covered by analyzing the brightness
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    
    # Load background image
    imgBackground = cv2.imread('Resources/background.png')
    
    if avg_brightness < 30:  # Threshold for detecting if the camera is covered
        cv2.putText(imgBackground, "Camera is covered", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        imgResize = cv2.resize(img, (454, 340))

        # Get prediction from classifier
        prediction = classifier.getPrediction(img)
        classID = prediction[1]
        print(classID)

        if classID != 0:
            # Overlay waste image and arrow
            imgBackground = cvzone.overlayPNG(imgBackground, imgWasteList[classID - 1], (909, 127))
            imgBackground = cvzone.overlayPNG(imgBackground, imgArrow, (978, 320))

            # Get corresponding bin
            classIDBin = classDic[classID]

        # Overlay bin image
        imgBackground = cvzone.overlayPNG(imgBackground, imgBinsList[classIDBin], (895, 374))

        # Overlay resized webcam image
        imgBackground[148:148 + 340, 159:159 + 454] = imgResize

    # Display the final output
    cv2.imshow("Output", imgBackground)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()