z=0
model_final.load_weights("/content/rcnn.h5")

print("test")

z += 1
img = cv2.imread("/content/drive/My Drive/Dataset_2/Tomato leaf bacterial spot/Bacterial_spots2276.jpg")
ss.setBaseImage(img)
ss.switchToSelectiveSearchFast()
ssresults = ss.process()
imout = img.copy()
print("test")
for e,result in enumerate(ssresults):
    if e < 2000:
        x,y,w,h = result
        timage = imout[y:y+h,x:x+w]
        resized = cv2.resize(timage, (224,224), interpolation = cv2.INTER_AREA)
        img = np.expand_dims(resized, axis=0)

        out= model_final.predict(img)
        p = np.argmax(out, axis=1)
        
        
        if out[0][p] > 0.6 and p != 0:
          print(p,out[0][p])
         
          if p == 1:
            boxclr = (255,255,0) #yellow
          elif p == 2:
            boxclr = (255,0,0) #red
          elif p == 3:
            boxclr = (255,0,255) #magenta

          cv2.rectangle(imout, (x, y), (x+w, y+h), boxclr, 3, cv2.LINE_AA)
plt.figure()
plt.imshow(imout)


