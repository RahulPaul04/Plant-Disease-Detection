import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import tensorflow as tf
import xml.etree.ElementTree as ET
from google.colab.patches import cv2_imshow

path = "/content/drive/My Drive/PlantDoc/Images_2"
annot = "/content/drive/My Drive/PlantDoc/Annotations_2"

cv2.setUseOptimized(True);
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

train_images=[]
train_labels=[]

def get_iou(bb1, bb2):
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


tb = 0
tm = 0
tl = 0
tc = 0

for k,i in enumerate(os.listdir(annot)):

  co = 0
  xmlfile = os.path.join(annot,"pp1244-05.jpeg.xml")
  for l in i:
    if l == '.':
      co = co + 1

   

  
 
  if k<4000:
          filename = ""
          for num in range(co):
             filename = filename + i.split(".")[num]+"."

          filename = filename + "jpeg"
          print(path)
          print(os.path.join(path,filename))
          image = cv2.imread(os.path.join(path,filename))


          
          root = ET.parse(os.path.join(annot,i)).getroot()
          for var in root.findall('object'):
            nn = var.find('name').text

        
         
          print(tb)
          print(tm)
          print(tl)
          gtvalues=[]
          if nn in ('Tomato leaf bacterial spot','Tomato leaf mosaic virus','Tomato leaf yellow virus'):
            print(i)
            tc = tc + 1
            print("Total count ",k,"  Image count  ",tc)
            for var in root.findall('object'):
                xs = var.find('bndbox')
                x1 =int(xs.find('xmin').text)
                x2 = int(xs.find('xmax').text)
                y1 = int(xs.find('ymin').text)
                y2 = int(xs.find('ymax').text)
                nn = var.find('name').text

                gtvalues.append({"x1":x1,"x2":x2,"y1":y1,"y2":y2})
                print(gtvalues)
            #cv2.rectangle(image,(x1,y1),(x2,y2),(255,0,0), 2)
          
          
            ss.setBaseImage(image)
            ss.switchToSelectiveSearchFast()
            ssresults = ss.process()
            imout = image.copy()
            counter = 0
            falsecounter = 0
            flag = 0
            fflag = 0
            bflag = 0
            for e,result in enumerate(ssresults):
                if e < 200 and flag == 0:
                    for gtval in gtvalues:
                        x,y,w,h = result
                        iou = get_iou(gtval,{"x1":x,"x2":x+w,"y1":y,"y2":y+h})
                        if counter < 30:
                            if iou > 0.70:
                                timage = imout[y:y+h,x:x+w]
                                resized = cv2.resize(timage, (224,224), interpolation = cv2.INTER_AREA)
                                train_images.append(resized)
                                if nn == 'Tomato leaf bacterial spot':
                                  train_labels.append(1)
                                  tb = tb + 1
                                 
                                if nn == 'Tomato leaf mosaic virus':
                                  train_labels.append(2)
                                  tm = tm + 1
                                  
                                if nn == 'Tomato leaf yellow virus':
                                  train_labels.append(3)
                                  tl = tl +1
                                      
                                counter += 1
                        else :
                            fflag =1
                        if falsecounter <30:
                            if iou < 0.3:
                                timage = imout[y:y+h,x:x+w]
                                resized = cv2.resize(timage, (224,224), interpolation = cv2.INTER_AREA)
                                train_images.append(resized)
                                train_labels.append(0)
                                falsecounter += 1
                        else :
                            bflag = 1
                    if fflag == 1 and bflag == 1:
                        print("inside")
                        flag = 1



X_new = np.array(train_images)
y_new = np.array(train_labels)


data = open('/content/drive/My Drive/trainfile_4.pickle',"wb")
label = open('/content/drive/My Drive/labelfile_4.pickle',"wb")
pickle.dump(X_new, data)
pickle.dump(y_new, label)
data.close()
          
