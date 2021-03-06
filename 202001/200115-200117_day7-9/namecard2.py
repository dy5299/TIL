from flask import Flask, escape, request
import pickle
import urllib.request
import cv2, numpy as np
import time
import matplotlib.pylab as plt
import pytesseract
from pandas import DataFrame as df
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
import json
from collections import OrderedDict



app = Flask(__name__)

def imgPross(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    img_blur = cv2.GaussianBlur(gray, (3,3) , 0)
    _, binary = cv2.threshold(img_blur, 0, 255,
            cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=5)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=10)


    testing = closing.copy()
    contours, _ = cv2.findContours(testing, cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)


    peri_list=[]
    maxArea = 0
    for i in range(len(contours)) :        
        con = contours[i]
        peri = cv2.arcLength(con, True)
        peri_list.append(peri)
        area = cv2.contourArea(con)  
        approx = cv2.approxPolyDP(con, 0.02 * peri, True)    
        if area > maxArea  :
            maxArea = area
            maxContour = approx
        
    draw = cv2.drawContours(testing, [maxContour], -1, (0,0,255), 20)
    if len(maxContour) != 4:
        return '다시 찍어주세요^^'

    import math
    def length(p1,p2) :
        result = math.sqrt(  (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2  )
        return result


    height = img.shape[0]
    width = img.shape[1]
    centerpt = [
        (maxContour[0,0,0]+maxContour[1,0,0]+maxContour[2,0,0]+maxContour[3,0,0])/4, 
        (maxContour[0,0,1]+maxContour[1,0,1]+maxContour[2,0,1]+maxContour[3,0,1])/4]

    idx = [  0,1,2,3  ]


    for i in range(4) :
        if maxContour[i,0,0] > centerpt[0] and maxContour[i,0,1] > centerpt[1] :
            idx[3] = i
        elif maxContour[i,0,0] > centerpt[0] and maxContour[i,0,1] < centerpt[1] :
            idx[1] = i
        elif maxContour[i,0,0] < centerpt[0] and maxContour[i,0,1] > centerpt[1] :
            idx[2] = i
        elif maxContour[i,0,0] < centerpt[0] and maxContour[i,0,1] < centerpt[1] :
            idx[0] = i

    pts1 = np.array(approx[idx, 0, :])
    pts1 = np.float32(pts1)
        


    width = length(pts1[0],pts1[1])
    height = length(pts1[0],pts1[2])



    pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])


    M = cv2.getPerspectiveTransform(pts1, pts2)  # (시작지점, 목적지점)으로 자동으로 행렬 생성
    img_result = cv2.warpPerspective(img, M, (int(width),int(height)))


    str = pytesseract.image_to_string(img_result, 'eng+kor')
    return str


@app.route('/namecard', methods=['POST'])  #명함인식
def namecard():
    db = df(columns=["ImgURL","INFO"])  #empty df

    body = request.json
    print("payload >> ")
    print(body)
    image_url = body['userRequest']['params']['media']['url']
    print(image_url)
    # print(body['userRequest']['utterance'])
    if 'http://dn-m.talk.kakao.com' in image_url:
        urllib.request.urlretrieve(image_url, 'card.png')
        img = cv2.imread("card.png")
        info = imgPross(img)
        info.rstrip('\n')

        templist = [image_url, info]
        db.append(df(np.array([templist]), columns=["ImgURL","INFO"]), ignore_index=True)

        if len(info) < 1: info = "인식을 실패했습니다. 다시 찍어주세요.^^"
        return{
            "version":"2.0",
            "template":{
                "outputs":[
                    {
                        "carousel":{
                            "type" : "basicCard",
                            "items": [
                                {
                                    "title":"이번에 인식한 명함입니다",
                                    "description":info,
                                    "thumbnail":{
                                        "imageUrl":image_url
                                    },
                                },
                            ]
                        }
                    }
                ]
            }
        }
    else:
        return{
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": "명함을 올려주세요"
                    }
                }
            ]
        }
    }

    # print('image_url>'+image_url)
    #f=request.files['file']
    #f.save(srcure_filename(f.filename))



@app.route('/list', methods=['POST'])
def cardlist():
    global idx
    body = request.json
    print(body)
    body['idx'] = int(idx)
    db[str(idx)]=body['idx']
    idx+=1
    db[ImgURL].append(image_url)
    db[INFO].append(info)


    db_json = OrderedDict()
    
    db_json["items"] = 


    
    return {
            "version":"2.0",
            "template":{
                "outputs":[
                    {
                        "carousel":{
                            "type" : "basicCard",
                            "items": [
                                for i in range(len(db)) :
                                    temp_list = 
                                    {
                                        "title":"명함 n번째",
                                        "description":info,
                                        "thumbnail":{
                                            "imageUrl":image_url
                                        },
                                    },
                            ]
                        }
                    }
                ]
            }
        }



@app.route('/abc', methods=['POST'])
def abc():
    return {
        "version": "2.0",
            "template": {
            "outputs": [
                {
                    'simpleText': {
                        'text': 'abc'
                    }
                }
            ]
        }
    }