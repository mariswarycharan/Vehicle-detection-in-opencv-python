
import cv2
from cv2 import VideoCapture

cars = 0
trained_dataset = cv2.CascadeClassifier(r"C:\Users\USER\projects\Real-Time_Vehicle_Detection-as-Simple\cars.xml")

video = cv2.VideoCapture(r"C:\Users\USER\projects\Real-Time_Vehicle_Detection-as-Simple\video.avi")

    
while True:
    a ,frame  = video.read()
    
    
    gray1 = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        
    eyes = trained_dataset.detectMultiScale(frame)
    for x,y,w,h in eyes:
        
        cv2.rectangle(frame, (x,y),(x+w,y+h),(255,0,0),2)
        cars += 1
        roi_gray = gray1[y:y+h,x:x+w]
        roi_color = frame[y:y+h,x:x+w]
    
    
    
    print(cars)
        
    cv2.imshow("video",frame)    

    
    if cv2.waitKey(1) == 81:
        break
    
video.release()
cv2.destroyAllWindows()       
    
        
        