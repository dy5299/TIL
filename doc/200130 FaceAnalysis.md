# Face Analysis

## Shape Keypoints 추출

```python
predictor = dlib.shape_predictor("../../../../shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()
```

```python
frame = cv2.imread("img/face.jpg")
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
rects = detector(gray,0) #Number of faces detected
```

```python
#얼굴의 keypoints
for k, d in enumerate(rects) :
    print("Detection {}: Left: {} Top: {} Right: {} Bottom {}".format(
    k, d.left(), d.top(), d.right(), d.bottom()))
    
    shape = predictor(gray, d)  #(영상, 좌표)
    shape = face_utils.shape_to_np(shape)  #객체의 좌표 정보들만 반환
    for (x,y) in shape:
        cv2.circle(frame, (x,y),2,(0,255,0),-1)
```

`enumerate` 사용하면 index, 내용 둘 다 쓸 수 있다.

```python
rects[0]					#[(413, 85) (485, 157)]
shape = predictor(gray, rects[0])	#<dlib.dlib.full_object_detection object at 0x0000027A092F9180> 라는 객체에 저장
face_utils.shape_to_np(shape)	#객체의 좌표 정보들
len(face_utils.shape_to_np(predictor(gray, rects[0]))) #68
```

<img src="C:/Users/student/Dropbox/MultiCampus/TIL/doc/images/200131_facial_landmarks_68markup.jpg" alt="200131_facial_landmarks_68markup" style="zoom:50%;" />

## Face Recognition

굳이 color image 사용 안 하고 gray image 사용한다.

```python
img = cv2.imread("img/face2.jpg")

face_locations = face_recognition.face_locations(img) #HOG algorithm
#face_locations = face_recognition.face_locations(img, model="cnn") #CNN algorithm, slower but more delicate
print("I found {} face(s) in this photograph.".format(len(face_locations)))

for face_location in face_locations:
    top, right, bottom, left = face_location
    cv2.rectangle(img, (left,top), (right,bottom),(0,0,255),3)
imshow("",img)
```



## Classification

```python
img = face_recognition.load_image_file("img/face1.jpg")
face_encoding = face_recognition.face_encodings(img)
```

`face_encoding[0]` : 한 사람의 특징. 이미지를 128차원으로 축소.(1차원 벡터) -> 추후 DNN으로 학습
classification은 거리 가까운 순으로.
mnist로 DNN했던 거랑 똑같아. db구성만 좀 달라지지. mnist는 28x28=784

### 같은 얼굴 찾기 - distance 비교

```python
files = os.listdir("img/faces")

known_face_encodings = []
known_face_names = []


for filename in files:
    name, ext = os.path.splitext(filename)
    if ext == '.jpg' :
        known_face_names.append(name)
        pathname = os.path.join("img/faces", filename)
        img = face_recognition.load_image_file(pathname)
        face_encoding = face_recognition.face_encodings(img)[0]
        known_face_encodings.append(face_encoding)

```

```python
test = face_recognition.load_image_file("img/faces_test.jpg")

face_locations = face_recognition.face_locations(test)
face_encodings = face_recognition.face_encodings(test, face_locations)

face_names = []
for face_encoding in face_encodings :
    distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    print(distances)
    min_value = min(distances)
    
    name = "unknown"
    if min_value < 0.6:
        index = np.argmin(distances)
        name = known_face_names[index]

print(name)
​```
[0.49864701 0.45618233 0.59453172 0.56340606 0.42147897 0.49540881
 0.50613125 0.30585624 0.50250804 0.32485049 0.46050412 0.53886023
 0.5256142  0.30441893 0.50097711 0.34303809]
download15
​```
```

### Face Classification

```python
imagePaths = list(paths.list_images("img/faces"))
 
knownEncodings = []
knownNames = []

for (i, imagePath) in enumerate(imagePaths):    
    name = imagePath.split(os.path.sep)[-2]  #뒤쪽2자리빼기
    image = cv2.imread(imagePath)
    boxes = face_recognition.face_locations(image)
    encodings = face_recognition.face_encodings(image, boxes) 

    for encoding in encodings:        
        knownEncodings.append(encoding)
        knownNames.append(name)
        
import pickle
data = {"encodings": knownEncodings, "names": knownNames} #둘 다 리스트
f = open("known.bin", "wb")  #binary 통채로 저장,불러들이는 방식
f.write(pickle.dumps(data))
f.close()
```

```python
data = pickle.loads(open("known.bin", "rb").read())

image = cv2.imread("img/faces_test2.jpg")

boxes = face_recognition.face_locations(image)
encodings = face_recognition.face_encodings(image, boxes)

names = []
for encoding in encodings:
    matches = face_recognition.compare_faces(data["encodings"], encoding)
    name = "Unknown"
    if True in matches:  #리스트 안에서 하나라도 특정 값(true)이 있으면
        matchedIdxs = [i for (i, b) in enumerate(matches) if b]
        counts = {}
        for i in matchedIdxs:
            name = data["names"][i]
            counts[name] = counts.get(name, 0) + 1  #name에 해당하는 element 불러오고 없으면 0 반환
        name = max(counts, key=counts.get)
    names.append(name)
for ((top, right, bottom, left), name) in zip(boxes, names): #zip:paring
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    y = top - 15 if top - 15 > 15 else top + 15
    cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,  0.75, (0, 255, 0), 2)

imshow("Image", image)
```

`np.array(data["encodings"])` : trained data: num of samples * dimension 을 DNN 에 넣으면 된다.

## Age, Gender

성별은 99% confidence, 나이는 10대/20대/.. 정도만 가능하다.

```python
import cv2 as cv2
import math
import time

def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, bboxes


faceProto = "../../../../agegender/opencv_face_detector.pbtxt"
faceModel = "../../../../agegender/opencv_face_detector_uint8.pb"

ageProto = "../../../../agegender/age_deploy.prototxt"
ageModel = "../../../../agegender/age_net.caffemodel"

genderProto = "../../../../agegender/gender_deploy.prototxt"
genderModel = "../../../../agegender/gender_net.caffemodel"

#MODEL_MEAN_VALUES = 
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Load network
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)
faceNet = cv2.dnn.readNet(faceModel, faceProto)

frame = cv2.imread("img/face5.jpg")
frameFace, bboxes = getFaceBox(faceNet, frame)
padding = 10
for bbox in bboxes:
        face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),(78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        print("Gender : {}, conf = {:.3f}".format(gender, genderPreds[0].max()))

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        print("Age Output : {}".format(agePreds))
        print("Age : {}, conf = {:.3f}".format(age, agePreds[0].max()))

        label = "{},{}".format(gender, age)
        cv2.putText(frameFace, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)        
        cv2.imwrite("img/face5_out.jpg",frameFace)

​```
Gender : Female, conf = 0.624
Age Output : [[0.01280975 0.00564699 0.0084409  0.01936056 0.79849434 0.12481617
  0.02942146 0.00100985]]
Age : (25-32), conf = 0.798
​```
```

## Emotion Expression

```python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
%matplotlib inline
from matplotlib.colors import ListedColormap
import seaborn as sns
import cv2
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image
import sys
from keras.models import load_model
from keras.layers import Activation, Convolution2D, Dropout, Conv2D
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential
from keras.layers import Flatten
from keras.models import Model
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import SeparableConv2D
from keras import layers



def imshow(tit, image) :
    plt.title(tit)    
    if len(image.shape) == 3 :
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else :
        plt.imshow(image, cmap="gray")
    plt.show()
```

```python
df = pd.read_csv("../../../../fer2013.csv")
#  0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral**
df.shape	#(35887, 3)
```

```python
image_size=(48,48)

pixels = df['pixels'].tolist() # Converting the relevant column element into a list for each row
width, height = 48, 48
faces = []
for pixel_sequence in pixels:
    
    face = [int(pixel) for pixel in pixel_sequence.split(' ')] # Splitting the string by space character as a list
    face = np.asarray(face).reshape(width, height) #converting the list to numpy array in size of 48*48
    face = cv2.resize(face.astype('uint8'),image_size) #resize the image to have 48 cols (width) and 48 rows (height)
    faces.append(face.astype('float32')) #makes the list of each images of 48*48 and their pixels in numpyarray form

faces = np.asarray(faces) #converting the list into numpy array
faces = np.expand_dims(faces, -1) #Expand the shape of an array -1=last dimension
```

```python
emotions = pd.get_dummies(df['emotion']).to_numpy()

print(faces.shape)		#(35887, 48, 48, 1)
print(emotions.shape)	#(35887, 7)
img = faces[0,:,:,0]
imshow("img", img)
```

```python
x = faces.astype('float32')
x = x / 255.0 #Dividing the pixels by 255 for normalization

# Scaling the pixels value in range(-1,1)
x = x - 0.5
x = x * 2.0
```

```python
num_samples = x.shape[0]
print(num_samples)			#35887
```

```python
num_samples, num_classes = emotions.shape
num_train_samples = int((1 - 0.2)*num_samples)

# Traning data
train_x = x[:num_train_samples]
train_y = emotions[:num_train_samples]

# Validation data
val_x = x[num_train_samples:]
val_y = emotions[num_train_samples:]

train_data = (train_x, train_y)
val_data = (val_x, val_y)
```

```python
print('Training Pixels',train_x.shape)		#(28709, 48, 48, 1)
print('Training labels',train_y.shape)		#(28709, 7)

print('Validation Pixels',val_x.shape)		#(7178, 48, 48, 1)
print('Validation labels',val_y.shape)		#(7178, 7)
```

```python
input_shape=(48, 48, 1)
num_classes = 7
```

```python
model = Sequential()
model.add(Convolution2D(filters=16, kernel_size=(7, 7), padding='same',
                            name='image_array', input_shape=input_shape))
model.add(BatchNormalization())
model.add(Convolution2D(filters=16, kernel_size=(7, 7), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(.5))

model.add(Convolution2D(filters=32, kernel_size=(5, 5), padding='same'))
model.add(BatchNormalization())
model.add(Convolution2D(filters=32, kernel_size=(5, 5), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(.5))

model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(.5))

model.add(Convolution2D(filters=128, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Convolution2D(filters=128, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(.5))

model.add(Convolution2D(filters=256, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Convolution2D(filters=num_classes, kernel_size=(3, 3), padding='same'))
model.add(GlobalAveragePooling2D())
model.add(Activation('softmax',name='predictions'))
```

- average pooling
  max pooling: 2x2에서 최댓값(필터에 가장 강하게 반응한 값)을 사용

  avg pooling: 평균값(smooth value) 사용하겠다

- sampling = pooling
  기본적인 풀링은 각 4개 중에서 선택하는건데,
  global pooling은 전체 pool의 avg. 필터마다 1개의 평균값 -> 1차원

```python
batch_size = 32
num_epochs = 200
verbose = 1
num_classes = 7 
```

```python
data_generator = ImageDataGenerator(
                        featurewise_center=False,
                        featurewise_std_normalization=False,
                        rotation_range=10,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=.1,
                        horizontal_flip=True)
```

- 이미지는 항상 data augmentation 해야한다.

얼굴은 padding, align을 얼마나 정교하게 했느냐에 따라 결과가 달라진다.

shift(10%정도-경험적), rotation, scale, flip 등을 넣는다.

```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

```python
train_faces, train_emotions = train_data
history=model.fit_generator(data_generator.flow(train_faces, train_emotions,
         batch_size), epochs=1, verbose=1,  validation_data=val_data)
```

```python
model.save("emotion_temp.h5")
score = model.evaluate(val_x, val_y, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1]*100)
```



```python
def load_detection_model(model_path):
    detection_model = cv2.CascadeClassifier(model_path)
    return detection_model

def detect_faces(detection_model, gray_image_array):
    return detection_model.detectMultiScale(gray_image_array, 1.3, 5)

def draw_bounding_box(face_coordinates, image_array, color):
    x, y, w, h = face_coordinates
    cv2.rectangle(image_array, (x, y), (x + w, y + h), color, 2)

def apply_offsets(face_coordinates, offsets):
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)

def draw_text(coordinates, image_array, text, color, x_offset=0, y_offset=0,
                                                font_scale=0.5, thickness=2):
    x, y = coordinates[:2]
    cv2.putText(image_array, text, (x + x_offset, y + y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)
    
def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x
```

```python
image_path = "tes.jpg"
detection_model_path = 'haarcascade_frontalface_alt2.xml'
emotion_model_path = 'fer2013_big_XCEPTION.54-0.66.hdf5'
emotion_model_path = "emotion_temp.h5"
emotion_labels = {0:'angry',1:'disgust',2:'fear',3:'happy',4:'sad',5:'surprise',6:'neutral'}
font = cv2.FONT_HERSHEY_SIMPLEX

emotion_offsets = (0, 0)

# loading models
face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
emotion_target_size = emotion_classifier.input_shape[1:3]
```





