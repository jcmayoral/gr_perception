#!/usr/bin/python3
import cv2
import numpy as np
from itertools import product

# K = Intrinsic camera matrix [ fx, 0, cx; 0 fy, cy ; 0  0 1]
# R = rectification matrix
# P = projection matrix [fx', 0, cx', Tx; 0 fy' cy Ty; 0 0 1 0]
frame_id = "camera_color_optical_frame"
height =  480
width = 640
distortion_model = "plumb_bob"
D = [0.0, 0.0, 0.0, 0.0, 0.0]
K = np.array(((617.4837646484375, 0.0, 321.1391906738281),(0.0, 617.0948486328125, 226.2030792236328), (0.0, 0.0, 1.0)))
R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
#P = [617.4837646484375, 0.0, 321.1391906738281, 0.0, 0.0, 617.0948486328125, 226.2030792236328, 0.0, 0.0, 0.0, 1.0, 0.0]

P = np.array(((617.4837646484375, 0.0, 321.1391906738281, 0.0),(0.0, 617.0948486328125, 226.2030792236328, 0.0), (0.0, 0.0, 1.0, 0.0)))
binning_x = 0
binning_y = 0
roi = {
  'x_offset': 0,
  'y_offset': 0,
  'height': 0,
  'width': 0,
  'do_rectify': False}

 # extinsic base_link to Camera
tx = 0.0
ty = 0.0
tz = 0.8
y = -  np.pi*0.75
p = 0
r= - np.pi/2

co = np.cos
si = np.sin

cr = co(r)
sr = si(r)
cp = co(p)
sp = si(p)
cy = co(y)
sy = si(y)
Rx = np.array(((1,0,0),(0, cr,-sr),(0,sr,cr)))
Ry = np.array(((cp,0,sp),(0, 1,0),(-sp,0,cp)))
Rz = np.array(((cy,-sy,0),(sy, cy,0),(0,0,1)))

R = np.matmul(np.matmul(Rz,Ry),Rx)

H = np.zeros((4,4))
H[:3,:3] = R
H[3,3] = 1.0
H[:,3] = [tx,ty,tz,1]

X = 0.0
Y = 0
Z = 0.0

point = [X,Y,Z,1]

#project point from base to camera location
print ("point on robot frame ", point)
print ("extrinsic matrix ", H)
#print ("projection matrix ", P)
camerapoint = H.dot(point)
print("point on camera frame: " , camerapoint)
#projection in camera frame
#print ("pixels, coordinate" , P.dot(camerapoint))
pixels = P.dot(camerapoint)
pixels[0] = np.round(pixels[0]/pixels[2])
pixels[1] = np.round(pixels[1]/pixels[2])
pixels[2] = pixels[2]/pixels[2] # =1

pixels = np.asarray(pixels, dtype = np.uint8)
print ("image coordinates", pixels)

img = cv2.imread('thorvald.jpg',1)
#print(img[pixels[0], pixels[1]])

for i,j in product(np.arange(100),np.arange(100)):
    img[pixels[0]+i, pixels[1]+j]= [255,0,0]
cv2.namedWindow('test', cv2.WINDOW_NORMAL)
cv2.imshow('test',img)

while(cv2.waitKey(0)!= 27):
    pass


cv2.destroyAllWindows()
