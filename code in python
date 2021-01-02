import imutils
import cv2
import numpy as np


def person_detection():
    protopath =  "D:\open cv\MobileNetSSD_deploy.prototxt.txt"
    modelpath = "D:\open cv\MobileNetSSD_deploy.caffemodel"
    detector = cv2.dnn.readNetFromCaffe(protopath, modelpath)  #detector is used to detect the person.
    
    
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                 "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                 "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                 "sofa", "train","tvmonitor"]
        
    cap  = cv2.VideoCapture(r'C:\Users\Priya Mittal\Downloads\People - 6387.mp4')
    
    total_frames = 0
    
    while True:
        _, frame = cap.read()
        frame = imutils.resize(frame, width=600)
        total_frames = total_frames 

        (H, W) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)

        detector.setInput(blob)
        person_detections = detector.forward()
        for i in np.arange(0, person_detections.shape[2]):
            confidence = person_detections[0, 0, i, 2]
            if confidence > 0.5:
                idx = int(person_detections[0, 0, i, 1])

                if CLASSES[idx] != "person":
                    continue

                person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = person_box.astype("int")
                
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                
                
        cv2.imshow("person detetction", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break

    cv2.destroyAllWindows()
person_detection()
