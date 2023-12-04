# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 07:48:17 2023

@author: dangh
"""

import cv2
import matplotlib.pyplot as plt

net = cv2.dnn.readNetFromTensorflow('graph_opt.pb') #pre_-train model form tensorflow

inWidth = 358
inHeight = 358
thresh = 0.175

BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["RShoulder", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["LShoulder", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"], ["RHip", "LHip"] ]

def pose__estimation(frame):
    frame_Width = frame.shape[1]
    frame_Height = frame.shape[0]
    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB = True, crop = False))
    out = net.forward()
    out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements
    
    assert(len(BODY_PARTS) == out.shape[1])
    
    points = []
    for i in range(len(BODY_PARTS)):
        # Slice heatmap of corresponging body's part.
        heatMap = out[0, i, :, :]

        # Originally, we try to find all the local maximums. To simplify a sample
        # we just find a global one. However only a single pose at the same time
        # could be detected this way.
        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = (frame_Width * point[0]) / out.shape[3]
        y = (frame_Height * point[1]) / out.shape[2]
        # Add a point if it's confidence is higher than threshold.
        points.append((int(x), int(y)) if conf > thresh else None)

    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert(partFrom in BODY_PARTS)
        assert(partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv2.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            cv2.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)

    #t, _ = net.getPerfProfile()
    #freq = cv2.getTickFrequency() / 1000
    #cv2.putText(frame, '%.2fms' % (t / freq), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    return frame


img = cv2.imread("1se.jpg")
result_img = pose__estimation(img)
plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
plt.show()


'_____________________________Real time________________________________'

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FPS, 10)
cap.set(3,800)
cap.set(4,800)

if not cap.isOpened():
    cap = cv2.VideoCapture(0)

while cv2.waitKey(1) < 0:
    hasFrame, frame = cap.read()
    if not hasFrame:
        cv2.waitKey()
        break
    
    frame_Width = frame.shape[1]
    frame_Height = frame.shape[0]
    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB = True, crop = False))
    out = net.forward()
    out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements

    #if frame_Width > inWidth or frame_Height > inHeight:
     #   frame = cv2.resize(frame, (inWidth, inHeight))

    # Chuyển đổi ảnh sang ảnh đen trắng
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Tính giá trị ngưỡng tự động bằng phương pháp Otsu
    #_, thresh = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY)
    
    assert(len(BODY_PARTS) == out.shape[1])
    
    points = []
    for i in range(len(BODY_PARTS)):
        # Slice heatmap of corresponging body's part.
        heatMap = out[0, i, :, :]

        # Originally, we try to find all the local maximums. To simplify a sample
        # we just find a global one. However only a single pose at the same time
        # could be detected this way.
        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = (frame_Width * point[0]) / out.shape[3]
        y = (frame_Height * point[1]) / out.shape[2]
        # Add a point if it's confidence is higher than threshold.
        points.append((int(x), int(y)) if conf > thresh else None)

    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert(partFrom in BODY_PARTS)
        assert(partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv2.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            cv2.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)

    cv2.imshow("Pose Estimation", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Nhấn 'Esc' để thoát
        break
    
cap.release()
cv2.destroyAllWindows()
