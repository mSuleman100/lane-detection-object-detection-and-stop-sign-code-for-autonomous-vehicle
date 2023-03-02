import cv2
import numpy as np
from gpiozero import LED


classifier = cv2.CascadeClassifier('stopsign_classifier.xml')

cap = cv2.VideoCapture(0)
cap.set(3,320)
cap.set(4,240)
ret,frame = cap.read()
roi1 = (78,65,201,136)



def getObstacle(frame,roi):
    font = cv2.FONT_HERSHEY_COMPLEX
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Crop selected roi from raw image
    roi_cropped=gray[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
    (thresh, im_bw) = cv2.threshold(roi_cropped, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    close = 255 - cv2.morphologyEx(im_bw, cv2.MORPH_CLOSE, kernel, iterations=2)            
    contours,hierarchy = cv2.findContours(close,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    if(len(contours)>0):
        for c in contours:
            x,y,w,h = cv2.boundingRect(c)
            if w>50 and w < 100:
                cv2.drawContours(frame, c, -1, (0, 0, 255), 3)
                print('obstacle found')
                cv2.putText(frame,'Obstacle',(0,100),font,2,(255,255,255),3)
            
    return frame



kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
while(1):
    ret,frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)        
        frame = getObstacle(frame,roi1)
        
   
        stop_signs = classifier.detectMultiScale(gray, 1.02, 10)
        if len(stop_signs)>0:
            for (x,y,w,h) in stop_signs:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
                print('stop sign estimated')
            
        else:        
            th, dst = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY);
            close = 255 - cv2.morphologyEx(dst, cv2.MORPH_CLOSE, kernel, iterations=2)            
            contours,hierarchy = cv2.findContours(close,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
            lines = cv2.HoughLinesP(close, 1, np.pi/180, 60, np.array([]), 50, 5)
            if(len(contours)>0):
                for c in contours:
                    x,y,w,h = cv2.boundingRect(c)
                    if w<150 and w>20:
                        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                        cv2.drawContours(frame, c, -1, (0, 255, 0), 3)
              
        #cv2.imshow('res',dst)
        cv2.imshow('frame',frame)
        #cv2.imshow('erode',c2)
        #cv2.imshow('m2',mask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break
        
cap.release()
cv2.destroyAllWindows()
