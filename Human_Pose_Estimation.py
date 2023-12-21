# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 22:09:54 2023

@author: dangh
"""


"""
Nhận diện và phân loại dáng người ( yoga pose )
"""
import os
import random as r
import math
import cv2
import numpy as np
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle as pic
from keras.models import Sequential
from keras.layers import LSTM, Dense, InputLayer, Dropout
from keras.utils import to_categorical
from keras.models import load_model

#Tạo class pose
mp_pose = mp.solutions.pose

## Tạo một đối tượng Pose với các chức năng
# static_image_mode=True: Xử lý chỉ một ảnh (False: video)
# min_detection_confidence=0.3: Ngưỡng nhận diện tối thiểu, chỉ nhận diện đối tượng khi độ tự tin (confidence) là 0.3 trở lên
# min_tracking_confidence=0.5: Ngưỡng theo dõi tối thiểu, chỉ theo dõi vị trí của các điểm chính khi độ tự tin là 0.5 trở lên.
# model_complexity=2: Độ phức tạp của mô hình (2 cho độ phức tạp trung bình)
# smooth_landmarks=True: Áp dụng làm mịn cho vị trí của các điểm chính, chỉ hoạt động khi static_image_mode=False
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, min_tracking_confidence=0.5, model_complexity=2, smooth_landmarks=True)

#Tạo class drawing_utils
mp_drawing = mp.solutions.drawing_utils

#Đọc ảnh
sample_img = cv2.imread('image.jpg')

#Tạo 1 figure size 10*10 để show ảnh và chuyển ảnh từ BGR to RGB
plt.figure(figsize = [10, 10])
plt.title("Image");plt.axis('off');plt.imshow(sample_img[:,:,::-1]);plt.show()

#Trả về 1 list gồm 33 điểm landmarks trong ảnh, các giá trị tọa độ được chuẩn hóa trong khoảng (0.0, 1.0)
results = pose.process(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB))

#In giá trị 33 điểm landmarks
if results.pose_landmarks:
    for i in range(33):
        print(f'{mp_pose.PoseLandmark(i).name}:\n{results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value]}')

#Quy đổi giá trị 33 điểm landmarks về kích thước ảnh gốc
image_height, image_width, _ = sample_img.shape

if results.pose_landmarks:
    for i in range(33):
        print(f'{mp_pose.PoseLandmark(i).name}:')
        print(f'x: {results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].x * image_width}')
        print(f'y: {results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].y * image_height}')
        print(f'z: {results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].z * image_width}')
        print(f'visibility: {results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].visibility}\n')

#Tạo 1 ảnh copy để vẽ các landmarks 
img_copy = sample_img.copy()

#Hiển thị pose trên ảnh copy
if results.pose_landmarks:
    mp_drawing.draw_landmarks(image=img_copy, landmark_list=results.pose_landmarks, connections=mp_pose.POSE_CONNECTIONS)
    fig = plt.figure(figsize = [10, 10])
    plt.title("Output");plt.axis('off');plt.imshow(img_copy[:,:,::-1]);plt.show()
    
# Hiển thị pose trên không gian 3D
mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)


def detectPose(image, pose, display=True):
    """
    Hàm này thực hiện việc nhận diện pose trên một hình ảnh.
    
    Đối số:
        image: Hình ảnh đầu vào với một người nổi bật cần phải nhận diện các landmarks của pose.
        pose: Hàm thiết lập cấu hình cho việc thực hiện nhận diện pose.
        display: Một giá trị boolean, nếu được đặt là true, hàm sẽ hiển thị hình ảnh gốc, hình ảnh kết quả và các landmarks trên đồ thị 3D và không trả về gì.

    Trả về:
        output_image: Hình ảnh đầu vào với các điểm landmarks của pose đã được vẽ lên.
        landmarks: Một danh sách các điểm landmarks đã nhận diện được chuyển đổi về kích thước gốc.
    """
    output_image = image.copy()
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(imageRGB)
    height, width, _ = image.shape

    landmarks = []
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS)
        for landmark in results.pose_landmarks.landmark:
            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                                  (landmark.z * width)))
    if display:

        plt.figure(figsize=[22,22])
        plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axis('off');
        plt.subplot(122);plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');

        mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
    else:
        return output_image, landmarks


'_________________________________Pose Classification________________________________'

def calculateAngle(landmark1, landmark2, landmark3):
    """
    Hàm này để tính toán góc độ dựa trên tọa độ x,y của 3 landmarks
    Giá trị của góc có thể nằm trong khoảng từ -180 đến 180 độ.
    Ví dụ, nếu angle ban đầu là -45 độ, khi thêm 360, nó sẽ trở thành 315 độ. 
    Do đó, bất kỳ giá trị âm nào đều được chuyển đổi thành giá trị dương tương đương trong khoảng từ 0 đến 360 độ.
    """

    # Get the required landmarks coordinates.
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    # Calculate the angle between the three points
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    
    # Kiểm tra xem góc có nhỏ hơn 0 không, nếu có thì thêm 360 độ vào góc để đảm bảo góc là dương
    if angle < 0:
        angle += 360
    
    # Return the calculated angle.
    return angle

def calculateDistance(landmark1, landmark2):
    """
    Hàm này để tính khoảng cách dựa trên tọa độ x,y của 2 landmarks
    """
    #Get the required landmarks coordinates
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    
    #Calculate the distance between the two points
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    #Return the calculate distance
    return distance

'''
# Calculate the angle between the three landmarks.
angle = calculateAngle((558, 326, 0), (642, 333, 0), (718, 321, 0))
print(f'The calculated angle is {angle}')
'''

def angles_finder(landmarks):
    # Calculate the required angles.
    #----------------------------------------------------------------------------------------------------------------
    
    # Get the angle between the left shoulder, elbow and wrist points. 
    left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
    
    # Get the angle between the right shoulder, elbow and wrist points. 
    right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])   
    
    # Get the angle between the left elbow, shoulder and hip points. 
    left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])

    # Get the angle between the right hip, shoulder and elbow points. 
    right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])

    # Get the angle between the left hip, knee and ankle points. 
    left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

    # Get the angle between the right hip, knee and ankle points 
    right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    
    #----------------------------------------------------------------------------------------------------------------
    
    return [left_elbow_angle,right_elbow_angle,left_shoulder_angle,right_shoulder_angle,left_knee_angle,right_knee_angle]

def distances_finder(landmarks):
    # Calculate the required distances
    
    #Get the distance between the left wrist, right wrist
    wrist_distance = calculateDistance(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])
    
    #Get the distance between the left ankle, right ankle
    ankle_distance = calculateDistance(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    
    #Get the distance between the left wrist, left ankle
    left_wrist_ankle_distance = calculateDistance(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value],
                                                  landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
    
    #Get the distance between the right wrist, right ankle
    right_wrist_ankle_distance = calculateDistance(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value],
                                                   landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    
    
    return [wrist_distance, ankle_distance, left_wrist_ankle_distance, right_wrist_ankle_distance]

#Đọc dataset
files = []
for dirname, _, filenames in os.walk('C:/Users/dangh/OneDrive/Documents/Subjects/Computer Vision/ProjectCK/3d_pose/DATASET/TRAIN'):
    for filename in filenames:
        file_path = os.path.join(dirname, filename)
        normalized_path = os.path.normpath(file_path)
        normalized_path = normalized_path.replace("\\", "/")
        files.append(normalized_path)
        #files.append(os.path.join(dirname, filename))
r.shuffle(files)
print(len(files))
#print('C:/Users/dangh/OneDrive/Documents/Subjects/Computer Vision/ProjectCK/3d_pose/DATASET/TRAIN'.split('/')[10])
#print(files[10])
#print('C:/Users/dangh/OneDrive/Documents/Subjects/Computer Vision/ProjectCK/3d_pose/DATASET/TRAIN/tree'.split('/')[11])

'''
'Seperate filename and Label Name'
def grab_data(file):
    image = cv2.imread(file)
    label = file.split('/')[11]
    output_image, landmarks = detectPose(image, pose, display=False)
    if landmarks:
        print(label)
        print(angles_finder(landmarks))
'''

'Create data frame angles landmarks'
df = pd.DataFrame(columns = ['Label','left_elbow_angle','right_elbow_angle','left_shoulder_angle','right_shoulder_angle','left_knee_angle','right_knee_angle'])
print(df)


'Create data frame distance landmarks'
df2 = pd.DataFrame(columns = ['Label', 'wrist_distance', 'ankle_distance', 'left_wrist_ankle_distance', 'right_wrist_ankle_distance'])
print(df2)

#Trích xuất đường dẫn ảnh và lưu các giá trị angles và distances vào df và df2
for i in range(len(files)-1):
    image = cv2.imread(files[i])
    label = files[i].split('/')[11]
    output_image, landmarks = detectPose(image, pose, display=False)
    if landmarks:
        #Angles
        r = angles_finder(landmarks)
        df = pd.concat([df,pd.DataFrame.from_records([{'Label':label,'left_elbow_angle':r[0],'right_elbow_angle':r[1],'left_shoulder_angle':r[2],'right_shoulder_angle':r[3],'left_knee_angle':r[4],'right_knee_angle':r[5]}])])
        
        #Distance
        r = distances_finder(landmarks)
        df2 = pd.concat([df2, pd.DataFrame.from_records([{'Label':label, 'wrist_distance':r[0], 'ankle_distance':r[1], 'left_wrist_ankle_distance':r[2], 'right_wrist_ankle_distance':r[3]}])])
        
#Mã hóa cột Label trong df và df2
le = preprocessing.LabelEncoder()
labels =[]
cols = ["Label"]
for col in cols:
    df[col] = le.fit_transform(df[col])
    df2[col] = le.fit_transform(df2[col])
    labels = le.classes_
print(labels)

#y = df["Label"]
#X = df.drop("Label",axis=1)

# Gộp DataFrame df và df2
combined_df = pd.merge(df, df2, on='Label')

# Tạo X từ combined_df bằng cách loại bỏ cột 'Label'
X_combined = combined_df.drop('Label', axis=1)

# Tạo y từ cột 'Label'
y_combined = combined_df['Label']

X_train, X_val, y_train, y_val = train_test_split(X_combined,y_combined,test_size=0.2, random_state=42)

# Chuyển đổi nhãn thành mã one-hot
y_train_one_hot = to_categorical(y_train, num_classes=5)

'___________________________________________Create a model__________________________________________'

model2 = Sequential([
    InputLayer(input_shape=(10,)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(5, activation='softmax')
])

model2.summary()
model2.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=['categorical_accuracy'])
print(model2.summary())

model2.fit(X_train, y_train_one_hot, epochs=25)

model2.save('tripleDense_500steps.h5')


'_____________________________Load model___________________________'

#clf = RandomForestClassifier(random_state = 42,
 #                            n_jobs = 1,
  #                           n_estimators = 1000).fit(X_train, y_train)


#pic.dump(clf, open('model.clf', 'wb'))

model_nn = load_model('tripleDense_500steps.h5')

def Pred_fun(filename):
    #Hàm dùng để đưa ra dự đoán pose
    image = cv2.imread(filename)
    label = filename.split('/')[11]
    output_image, landmarks = detectPose(image, pose, display=False)
    if landmarks:
        result_angles = angles_finder(landmarks)
        print('Angles:', result_angles)
        result_distance = distances_finder(landmarks)
        print('Distances:', result_distance)
        #prediction = clf.predict(np.array(result_angles).reshape(1,-1))
        #prediction = model2.predict(np.array(result_angles + result_distance).reshape(1, -1))
        #prediction = preprocess_input_nn(np.array(result_angles + result_distance).reshape(1, -1))
        prediction = model_nn.predict(np.array(result_angles + result_distance).reshape(1, -1))
        print("Actual Label:",label," Predicted Label:", labels[np.argmax(prediction)])
        cv2.putText(output_image, 'Actual: {}'.format(label), (10, 30),cv2.FONT_HERSHEY_PLAIN, 1,(0, 0, 255), 2)
        cv2.putText(output_image, 'Pred: {}'.format(str(labels[np.argmax(prediction)])), (10, 50),cv2.FONT_HERSHEY_PLAIN, 1,(0, 255, 0),2)
        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
Pred_fun(files[159])

#prediction = clf.predict(X_val)
prediction = model_nn.predict(X_val)
# print accuracy metrics
print("Accuracy:",accuracy_score(y_val,np.argmax(prediction, axis = 1)))

img = cv2.imread("C:/Users/dangh/OneDrive/Documents/Subjects/Computer Vision/ProjectCK/3d_pose/treepose1.jpg")
output_image, landmarks = detectPose(img, pose, display=False)
if landmarks:
    result_angles = angles_finder(landmarks)
    print('Angles:', result_angles)
    result_distance = distances_finder(landmarks)
    print('Distances:', result_distance)
    #prediction = clf.predict(np.array(result_angles).reshape(1,-1))
    #prediction = model2.predict(np.array(result_angles + result_distance).reshape(1, -1))
    #prediction = preprocess_input_nn(np.array(result_angles + result_distance).reshape(1, -1))
    prediction = model_nn.predict(np.array(result_angles + result_distance).reshape(1, -1))
    print(" Predicted Label:", labels[np.argmax(prediction)])
    cv2.putText(output_image, str(labels[np.argmax(prediction)]), (10, 50),cv2.FONT_HERSHEY_PLAIN, 1,(0, 255, 0),2)
    plt.figure(figsize=[10,10])
    plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');



'_____________________________________Real time____________________________________'

video = cv2.VideoCapture(0)
cv2.namedWindow('Pose Detection', cv2.WINDOW_NORMAL)

pTime = 0

while True:
    ok, frame = video.read()
    
    if not ok:
        break

    #frame = cv2.flip(frame, 1)
    #frame_height, frame_width, _ =  frame.shape
    #frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))
    frame, _ = detectPose(frame, pose, display=False)
    
    output_frame, landmarks = detectPose(frame, pose, display=False)
    if landmarks:
        result_angles = angles_finder(landmarks)
        print('Angles:', result_angles)
        result_distance = distances_finder(landmarks)
        print('Distances:', result_distance)
        #prediction = clf.predict(np.array(result_angles).reshape(1,-1))
        #prediction = model2.predict(np.array(result_angles + result_distance).reshape(1, -1))
        #prediction = preprocess_input_nn(np.array(result_angles + result_distance).reshape(1, -1))
        prediction = model_nn.predict(np.array(result_angles + result_distance).reshape(1, -1))
        print(" Predicted Label:", labels[np.argmax(prediction)])
        cv2.putText(frame, 'Pred: {}'.format(str(labels[np.argmax(prediction)])), (10, 50),cv2.FONT_HERSHEY_PLAIN, 1,(0, 255, 0),2)

        
    cTime = time()
    if (cTime - pTime) > 0:
        frames_per_second = 1.0 / (cTime - pTime)

        cv2.putText(frame, 'FPS: {}'.format(int(frames_per_second)), (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)

    pTime = cTime

    cv2.imshow('Pose Detection', frame)

    k = cv2.waitKey(1) & 0xFF
    if(k == 27):
        break

video.release()
cv2.destroyAllWindows()
