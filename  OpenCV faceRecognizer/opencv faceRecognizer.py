#!/usr/bin/env python
# coding: utf-8

# In[13]:


import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[14]:


cv2.__version__


# In[17]:


#初始化
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") # 載入人臉追蹤模型
recog = cv2.face.LBPHFaceRecognizer_create()      # 啟用訓練人臉模型方法
faces = []   # 儲存人臉位置大小的串列
ids = []     # 記錄該人臉 id 的串列


# In[18]:


#讀取VideoOutput圖片

detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") # 載入人臉追蹤模型
recog = cv2.face.LBPHFaceRecognizer_create()      # 啟用訓練人臉模型方法
faces = []   # 儲存人臉位置大小的串列
ids = []     # 記錄該人臉 id 的串列

import os 
imagelist=os.listdir('./output/')
for i in imagelist:
    print(i)
    
for i in imagelist:
    face=()
    file_name = "./output/"+i
    print(file_name)
    img = cv2.imread(file_name)       # 依序開啟每一張Tony_Blair的照片
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_np = np.array(gray,'uint8')           # 轉換成指定編碼的 numpy 陣列
#     print(img_np)
    face = detector.detectMultiScale(gray,1.2)# 擷取人臉區域
    print(len(face))
    if(len(face)==1):
        for(x,y,w,h) in face:
            faces.append(img_np[y:y+h,x:x+w])         # 記錄蔡英文人臉的位置和大小內像素的數值
            ids.append(1)                            # 記錄蔡英文人臉對應的 id，只能是整數，都是 1 表示蔡英文的 id 為 1
print("訓練張數:",len(faces))
print("標記id:",len(ids))


# In[4]:


#讀取Tony_Blair圖片

detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") # 載入人臉追蹤模型
recog = cv2.face.LBPHFaceRecognizer_create()      # 啟用訓練人臉模型方法
faces = []   # 儲存人臉位置大小的串列
ids = []     # 記錄該人臉 id 的串列

import os 
imagelist=os.listdir('./picture/pins_Anthony Mackie/')
for i in imagelist:
    print(i)
    
for i in imagelist:
    face=()
    file_name = "./picture/pins_Anthony Mackie/"+i
    print(file_name)
    img = cv2.imread(file_name)       # 依序開啟每一張Tony_Blair的照片
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_np = np.array(gray,'uint8')           # 轉換成指定編碼的 numpy 陣列
#     print(img_np)
    face = detector.detectMultiScale(gray)# 擷取人臉區域
    print(len(face))
    if(len(face)==1):
        for(x,y,w,h) in face:
            faces.append(img_np[y:y+h,x:x+w])         # 記錄蔡英文人臉的位置和大小內像素的數值
            ids.append(1)                            # 記錄蔡英文人臉對應的 id，只能是整數，都是 1 表示蔡英文的 id 為 1
print("訓練張數:",len(faces))
print("標記id:",len(ids))


# In[13]:


#讀取在陰文圖片
import os 
imagelist=os.listdir('./picture/tisa/')
for i in imagelist:
    print(i)
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") # 載入人臉追蹤模型
recog = cv2.face.LBPHFaceRecognizer_create()      # 啟用訓練人臉模型方法
faces = []   # 儲存人臉位置大小的串列
ids = []     # 記錄該人臉 id 的串列

for i in imagelist:
    face=()
    file_name = "./picture/tisa/"+i
    print(file_name)
    img = cv2.imread(file_name)       # 依序開啟每一張蔡英文的照片
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # 色彩轉換成黑白
    img_np = np.array(gray,'uint8')           # 轉換成指定編碼的 numpy 陣列
#     print(img_np)
    face = detector.detectMultiScale(gray,1.2,5)# 擷取人臉區域
    print(len(face))
    if(len(face)==1):
        for(x,y,w,h) in face:
            faces.append(img_np[y:y+h,x:x+w])         # 記錄蔡英文人臉的位置和大小內像素的數值
            ids.append(1)                            # 記錄蔡英文人臉對應的 id，只能是整數，都是 1 表示蔡英文的 id 為 1
print("訓練張數:",len(faces))


# In[170]:


#讀取賴清德圖片
import os 
imagelist=os.listdir('./picture/lip/')
for i in imagelist:
    print(i)

for i in imagelist:
    face=()
    file_name ="./picture/lip/"+i
    print(file_name)
    img = cv2.imread(file_name)       # 依序開啟每一張賴的照片
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # 色彩轉換成黑白
    img_np = np.array(gray,'uint8')           # 轉換成指定編碼的 numpy 陣列
    #print(img_np)
    face = detector.detectMultiScale(gray,1.2,5) # 擷取人臉區域
    print(len(face))
    if(len(face)==1):
        for(x,y,w,h) in face:
            faces.append(img_np[y:y+h,x:x+w])         # 記錄蔡英文人臉的位置和大小內像素的數值
            ids.append(2)                             # 記錄蔡英文人臉對應的 id，只能是整數，都是 1 表示蔡英文的 id 為 1
print("訓練張數:",len(faces))


# In[19]:


print('training...')                              # 提示開始訓練
recog.train(faces,np.array(ids))                  # 開始訓練
recog.save('face.yml')                            # 訓練完成儲存為 face.yml
print('ok!')


# In[229]:


#開始辨識 利用 face.yml


# In[63]:


name = {
      '1':'kennychuang',
      '2':'Li',
      '3':'oxxostudio'
  }


# In[64]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
recognizer = cv2.face.LBPHFaceRecognizer_create()         # 啟用訓練人臉模型方法
recognizer.read('face.yml')                               # 讀取人臉模型檔
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # 載入人臉追蹤模型


# In[65]:


img = cv2.imread('./outputTest/(5).jpg')
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
        if confidence < 60:
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


# In[67]:


import os 
imagelist=os.listdir('./outputTest/')
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
    file_name = "./outputTest/"+i
    img = cv2.imread(file_name)
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)  # 轉換成黑白
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
            print(confidence)
            correctnumber+=1
        else:
            text = '???'
        cv2.putText(img, text, (x,y-5),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
        cv2.imwrite('./result/'+i, img)  
print(correctnumber)
print('------------------------------------------')
print('face1Number=',face1Number)


# In[ ]:




