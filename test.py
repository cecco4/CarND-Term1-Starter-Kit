import numpy as np
import cv2
from skvideo.io import VideoCapture
#from cv2 import VideoCapture

def compute_lines(image):
	return image


cap = VideoCapture("test.avi")

while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = compute_lines(frame)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
