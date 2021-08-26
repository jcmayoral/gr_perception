from crop_detector import CropDetector
import cv2

def on_change(val):
    print (val)

if __name__ == "__main__":
    cv2.createTrackbar("process", "OK", 0,100,on_change)#self.params[i].on_change)

    CropDetector()
