from numba import jit
import numpy as np
import cv2

@jit(nopython=True)
def frame2ascii(frame, images, box_height=12, box_width=16):
    height,width = frame.shape
    for i in range(0, height,box_height):
        for j in range(0, width,box_width):
            roi = frame[i:i + box_height, j:j + box_width]
            best_match = np.inf
            best_match_index = 0
            for k in range(1,images.shape[0]):
                total_sum = np.sum(np.absolute(np.subtract(roi,images[k])))
                if total_sum < best_match:
                    best_match = total_sum
                    best_match_index = k
                    
            roi[:,:] = images[best_match_index]
    return frame
def gen_ascii():
    images = []
    #letters = "# $%&\\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_`abcdefghijklmnopqrstuvwxyz{|}~"
    letters = " \\ '(),-./:;[]_`{|}~"
    for letter in letters:
        img = np.zeros((12, 16), np.uint8)
        img = cv2.putText(img, letter, (0, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
        images.append(img)
    return np.stack(images)

def track(x):
    print(x)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
images = gen_ascii()
cv2.namedWindow('ascii art')
cv2.namedWindow('Webcam')
cv2.createTrackbar('lower tresh','ascii art',0,100,track)
cv2.createTrackbar('upper tresh','ascii art',0,100,track) 
while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    webcam_frame = cv2.resize(frame,None,None,0.5,0.5)
    gb = cv2.GaussianBlur(frame, (5, 5), 0)
    treshh1 = cv2.getTrackbarPos('lower tresh','ascii art')
    treshh2 = cv2.getTrackbarPos('upper tresh','ascii art')
    can = cv2.Canny(gb, treshh1, treshh2)
    ascii_art = frame2ascii(can,images)
    cv2.imshow('ascii art',ascii_art)
    cv2.imshow("Webcam", webcam_frame)
    
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()