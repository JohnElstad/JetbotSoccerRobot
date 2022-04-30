import cv2
import numpy as np
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    hsvframe = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    lower = np.array([ 90, 100,  93])
    upper = np.array([151, 269, 193])
    mask = cv2.bitwise_not(cv2.inRange(hsvframe, lower, upper))
    cv2.imshow("MASK",mask)    
    c = cv2.waitKey(10)
    if c == 'q':
        break

    edges = cv2.Canny(mask,50,150,apertureSize = 3)

    lines = cv2.HoughLines(edges,1,np.pi/180,200)
    if lines is not None:
        for rho,theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))

            cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),2)
            cv2.imshow('Frame', frame)


cap.release()
cv2.destroyAllWindows()