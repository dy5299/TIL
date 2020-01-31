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