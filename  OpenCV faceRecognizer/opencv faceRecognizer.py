#!/usr/bin/env python
# coding: utf-8

# In[61]:


import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[62]:


cv2.__version__


# In[63]:


#初始化
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") # 載入人臉追蹤模型
recog = cv2.face.LBPHFaceRecognizer_create()      # 啟用訓練人臉模型方法
faces = []   # 儲存人臉位置大小的串列
ids = []     # 記錄該人臉 id 的串列


# In[64]:


#讀取HAO圖片

import os 
imagelist=os.listdir('./haoTest/')
for i in imagelist:
    print(i)
    
for i in imagelist:
    face=()
    file_name = "./haoTest/"+i
    print(file_name)
    img = cv2.imread(file_name)       # 依序開啟每一張Tony_Blair的照片
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_np = np.array(gray,'uint8')           # 轉換成指定編碼的 numpy 陣列
#     print(img_np)
    face = detector.detectMultiScale(gray,1.5,4)# 擷取人臉區域
    print(len(face))
    if(len(face)==1):
        for(x,y,w,h) in face:
            faces.append(img_np[y:y+h,x:x+w])         # 記錄蔡英文人臉的位置和大小內像素的數值
            ids.append(1)                            # 記錄蔡英文人臉對應的 id，只能是整數，都是 1 表示蔡英文的 id 為 1
print("訓練張數:",len(faces))
print("標記id:",len(ids))


# In[13]:


#讀取yisinVideoOutput圖片

import os 
imagelist=os.listdir('./yisinOutput/')
for i in imagelist:
    print(i)
    
for i in imagelist:
    face=()
    file_name = "./yisinOutput/"+i
    print(file_name)
    img = cv2.imread(file_name)       # 依序開啟每一張Tony_Blair的照片
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_np = np.array(gray,'uint8')           # 轉換成指定編碼的 numpy 陣列
#     print(img_np)
    face = detector.detectMultiScale(gray,1.5,2)# 擷取人臉區域
    print(len(face))
        
    if(len(face)==1):
        for(x,y,w,h) in face:
            faces.append(img_np[y:y+h,x:x+w])         # 記錄蔡英文人臉的位置和大小內像素的數值
            ids.append(2)                        # 記錄蔡英文人臉對應的 id，只能是整數，都是 1 表示蔡英文的 id 為 1
#             cv2.rectangle(img_np,(x,y),(x+w,y+h),(0,255,0),2)
#             cv2.namedWindow('My Image',0)
#             cv2.resizeWindow('My Image',1000,1000)
#             cv2.imshow('My Image', img_np)
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()

# cv2.waitKey(0)
# cv2.destroyAllWindows()
    
    
print("訓練張數:",len(faces))
print("標記id:",len(ids))


# In[65]:


#讀取KennyVideoOutput圖片

import os 
imagelist=os.listdir('./kenny2Output/')
for i in imagelist:
    print(i)
    
for i in imagelist:
    face=()
    file_name = "./kenny2Output/"+i
    print(file_name)
    img = cv2.imread(file_name)       # 依序開啟每一張Tony_Blair的照片
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_np = np.array(gray,'uint8')           # 轉換成指定編碼的 numpy 陣列
#     print(img_np)
    face = detector.detectMultiScale(gray,1.2,5)# 擷取人臉區域
    print(len(face))
        
    if(len(face)==1):
        for(x,y,w,h) in face:
            faces.append(img_np[y:y+h,x:x+w])         # 記錄蔡英文人臉的位置和大小內像素的數值
            ids.append(3)                        # 記錄蔡英文人臉對應的 id，只能是整數，都是 1 表示蔡英文的 id 為 1
#             cv2.rectangle(img_np,(x,y),(x+w,y+h),(0,255,0),2)
#             cv2.namedWindow('My Image',0)
#             cv2.resizeWindow('My Image',1000,1000)
#             cv2.imshow('My Image', img_np)
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()

# cv2.waitKey(0)
# cv2.destroyAllWindows()
    
    
print("訓練張數:",len(faces))
print("標記id:",len(ids))


# In[66]:


#讀取NiVideoOutput圖片

import os 
imagelist=os.listdir('./niTest/')
for i in imagelist:
    print(i)
    
for i in imagelist:
    face=()
    file_name = "./niTest/"+i
    print(file_name)
    img = cv2.imread(file_name)       # 依序開啟每一張Tony_Blair的照片
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_np = np.array(gray,'uint8')           # 轉換成指定編碼的 numpy 陣列
#     print(img_np)
    face = detector.detectMultiScale(gray,1.2,5)# 擷取人臉區域
    print(len(face))
        
    if(len(face)==1):
        for(x,y,w,h) in face:
            faces.append(img_np[y:y+h,x:x+w])         # 記錄蔡英文人臉的位置和大小內像素的數值
            ids.append(4)                        # 記錄蔡英文人臉對應的 id，只能是整數，都是 1 表示蔡英文的 id 為 1
#             cv2.rectangle(img_np,(x,y),(x+w,y+h),(0,255,0),2)
#             cv2.namedWindow('My Image',0)
#             cv2.resizeWindow('My Image',1000,1000)
#             cv2.imshow('My Image', img_np)
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()

# cv2.waitKey(0)
# cv2.destroyAllWindows()
    
    
print("訓練張數:",len(faces))
print("標記id:",len(ids))


# In[67]:




print(len(faces))
print(ids)


# In[68]:


print('training...')                              # 提示開始訓練
recog.train(faces,np.array(ids))                  # 開始訓練
recog.save('face.yml')                            # 訓練完成儲存為 face.yml
print('ok!')


# In[32]:


#開始辨識 利用 face.yml


# In[1]:


name = {
      '1':'Hao',
      '2':'YISIN',
      '3':'KENNY',
      '4':'NISIN'
  }


# In[2]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
recognizer = cv2.face.LBPHFaceRecognizer_create()         # 啟用訓練人臉模型方法
recognizer.read('face.yml')                               # 讀取人臉模型檔
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # 載入人臉追蹤模型


# In[62]:


#單張測試
img = cv2.imread('./HaoTestOld/(2).jpg')
# img = cv2.imread('./picture/Tony_Blair/Tony_Blair_00.jpg')
# img = cv2.resize(img,(1620,900))              # 縮小尺寸，加快辨識效率
gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)  # 轉換成黑白
faces = face_cascade.detectMultiScale(gray,1.2)  # 追蹤人臉 ( 目的在於標記出外框 )
print(faces)
# plt.imshow(gray) 
for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)            # 標記人臉外框
        idnum,confidence = recognizer.predict(gray[y:y+h,x:x+w]) # 取出 id 號碼以及信心指數 confidence
        print(confidence)
        if confidence <60 :
            text = name[str(idnum)]                             # 如果信心指數小於 60，取得對應的名字
        else:
            text = '???'
        print(text)
        cv2.putText(img, text, (x,y-5),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
# plt.imshow(img)
cv2.namedWindow('My Image',0)
cv2.resizeWindow('My Image',1000,1000)
cv2.imshow('My Image', img)

cv2.waitKey(0)
cv2.destroyAllWindows()


# In[28]:


import os 
imagelist=os.listdir('./HaoTestOld/')
dataNumber=0
correctnumber=0
face1Number=0
for i in imagelist:
    print(i)
    dataNumber+=1
print(dataNumber)
print("---------------------------------------------------------------")
for i in imagelist:
    face=()
    faces=()
    file_name = "./HaoTestOld/"+i
    img = cv2.imread(file_name)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  # 轉換成黑白
    faces = face_cascade.detectMultiScale(gray,1.2)  # 追蹤人臉 ( 目的在於標記出外框 )
    if(len(faces)==1):
            face1Number+=1
#     print(i)
#     print(len(faces))
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)            # 標記人臉外框
        idnum,confidence = recognizer.predict(gray[y:y+h,x:x+w]) # 取出 id 號碼以及信心指數 confidence
#         print(confidence)
        
        if confidence < 60:
            text = name[str(idnum)]                               # 如果信心指數小於 60，取得對應的名字
            print(i)
            print(idnum)
            print(confidence)
            correctnumber+=1
        else:
            text = '???'
        cv2.putText(img, text, (x,y-5),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
        cv2.imwrite('./result/'+i, img)  
print(correctnumber)
print('------------------------------------------')
print('face1Number=',face1Number)


# In[4]:


#讀取影片測試
import cv2
cap = cv2.VideoCapture('./videoSource/kenny.mp4')
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    ret, frame = cap.read()             # 讀取影片的每一幀
    if not ret:
        print("Cannot receive frame")   # 如果讀取錯誤，印出訊息
        break
    cv2.namedWindow('My Image',0)
    cv2.resizeWindow('My Image',1000,1000)
    cv2.imshow('My Image', frame)
    if cv2.waitKey(1) == ord('q'):      # 每一毫秒更新一次，直到按下 q 結束
        break
cap.release()                           # 所有作業都完成後，釋放資源
cv2.destroyAllWindows()    


# In[3]:


dataNumber=0
correctnumber=0
face1Number=0
i=0


# In[10]:


# 影片輸出結果
import cv2
cap = cv2.VideoCapture('./videoSource/allPeople2.mp4')
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    ret, frame = cap.read()             # 讀取影片的每一幀
    if not ret:
        print("Cannot receive frame")   # 如果讀取錯誤，印出訊息
        break
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)  # 轉換成黑白
    faces = face_cascade.detectMultiScale(gray,1.2,5)  # 追蹤人臉 ( 目的在於標記出外框 )
    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)            # 標記人臉外框
        idnum,confidence = recognizer.predict(gray[y:y+h,x:x+w]) # 取出 id 號碼以及信心指數 confidence
#       print(confidence)
        if confidence < 60:
            text = name[str(idnum)]                               # 如果信心指數小於 60，取得對應的名字
#             print(i)
#             print(idnum)
#             print(confidence)
            correctnumber+=1
        else:
            text = '???'
        text=str(idnum)+text+' confidence='+str(confidence)
        cv2.putText(frame, text, (x,y-5),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
    cv2.namedWindow('My Image',0)
    cv2.resizeWindow('My Image',1000,1000)
    cv2.imshow('My Image', frame)
    if cv2.waitKey(1) == ord('q'):      # 每一毫秒更新一次，直到按下 q 結束
        break
cap.release()                           # 所有作業都完成後，釋放資源
cv2.destroyAllWindows() 
#     cv2.imwrite('./VideoSourcePicture/'+str(i), frame)
#     i=i+1


# In[ ]:




