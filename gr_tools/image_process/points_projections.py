#!/usr/bin/python3
import cv2
import numpy as np
from itertools import product
from skimage import io

# K = Intrinsic camera matrix [ fx, 0, cx; 0 fy, cy ; 0  0 1]
# R = rectification matrix
# P = projection matrix [fx', 0, cx', Tx; 0 fy' cy Ty; 0 0 1 0]
rgb_data = {
    'frame_id' : "camera_color_optical_frame",
    'height' : 480,
    'width' : 640,
    'distortion_model' : "plumb_bob",
    'D' : [0.0, 0.0, 0.0, 0.0, 0.0],
    'K' : np.array(((617.4837646484375, 0.0, 321.1391906738281),(0.0, 617.0948486328125, 226.2030792236328), (0.0, 0.0, 1.0))),
    'R' : [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    #P = [617.4837646484375, 0.0, 321.1391906738281, 0.0, 0.0, 617.0948486328125, 226.2030792236328, 0.0, 0.0, 0.0, 1.0, 0.0]
    'P' : np.array(((617.4837646484375, 0.0, 321.1391906738281, 0.0),(0.0, 617.0948486328125, 226.2030792236328, 0.0), (0.0, 0.0, 1.0, 0.0))),
    'binning_x' : 0,
    'binning_y' : 0,
    'roi' : {
      'x_offset': 0,
      'y_offset': 0,
      'height': 0,
      'width': 0,
      'do_rectify': False}

}

depth_data = {
    'frame_id': "camera_depth_optical_frame",
    'height': 480,
    'width': 640,
    'distortion_model': "plumb_bob",
    'D': [0.0, 0.0, 0.0, 0.0, 0.0],
    'K': np.array(((595.2252807617188, 0.0, 314.1288757324219), (0.0, 595.2252807617188, 230.69776916503906), (0.0, 0.0, 1.0))),
    'R': [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    'P': np.array(((595.2252807617188, 0.0, 314.1288757324219, 0.0), (0.0, 595.2252807617188, 230.69776916503906, 0.0), (0.0, 0.0, 1.0, 0.0))),
    'binning_x': 0,
    'binning_y': 0,
    'roi': {
        'x_offset': 0,
        'y_offset': 0,
        'height': 0,
        'width': 0,
        'do_rectify': False}
}

depth_to_color = {
    'frame_id': "depth_to_color_extrinsics",
    'rotation' : np.array(((0.9999988079071045, 0.00032068302971310914, -0.0015222439542412758),
                            (-0.00032303109765052795, 0.9999987483024597, -0.0015425152378156781),
                            (0.001521747326478362, 0.0015430051134899259, 0.9999976754188538))),
    'translation': [0.01496141403913498, -1.9867740775225684e-05, -0.000291701901005581],
    'H' : (((0.9999988079071045, 0.00032068302971310914, -0.0015222439542412758,0.01496141403913498 ),
                            (-0.00032303109765052795, 0.9999987483024597, -0.0015425152378156781,-1.9867740775225684e-05),
                            (0.001521747326478362, 0.0015430051134899259, 0.9999976754188538,-0.000291701901005581),
                            (0,0,0,1)))
}



def project(point, data):
    #project point from base to camera location
    #print ("point on robot frame ", point)
    #print ("projection matrix ", P)
    #camerapoint = data['P'].dot(point)
    camerapoint = point
    print("point on camera frame: " , camerapoint)
    #projection in camera frame
    #print ("pixels, coordinate" , P.dot(camerapoint))
    pixels = data['K'].dot(camerapoint)
    pixels[0] = pixels[0]/pixels[2]
    pixels[1] = pixels[1]/pixels[2]
    pixels[2] = pixels[2]/pixels[2] # =1

    pixels = np.asarray(pixels).astype(np.int64)#, dtype = np.uint8)
    print ("image coordinates", pixels)
    pixels[0] = int(pixels[0])
    pixels[1] = int(pixels[1])
    print ("image coordinates", type(pixels[0]))

    return tuple(pixels[:2])

img = cv2.imread('example.png',1)
#img = depth_img
#print(img[pixels[0], pixels[1]])
startpoint = [0.3,0,1.0,1]
endpoint = [0.3,0,1.0,1]
print ("startpoint ", startpoint, sep ='\n')
print ("endpoint ", endpoint, sep ='\n')

startpixels = project(startpoint, rgb_data)

endpixels = project(endpoint, rgb_data)

#for i,j in product(np.arange(100),np.arange(100)):
#    img[pixels[0]+i, pixels[1]+j]= [255,0,0]
color = (0, 255, 0)
thickness = 9
#startpixels  = tuple(reversed(startpixels))
#endpixels  = tuple(reversed(endpixels))

#img = cv2.line(img, startpixels,endpixels, color, thickness)
#startpixels  = tuple(reversed(startpixels))
#endpixels  = tuple(reversed(endpixels))


for i,j in product(np.arange(3),np.arange(3)):
    print(startpixels)
    img[startpixels[0]+i, startpixels[1]+j]= [255,0,0]

for i,j in product(np.arange(3),np.arange(3)):
    img[endpixels[0]+i, endpixels[1]+j]= [0,0,255]


cv2.namedWindow('test', cv2.WINDOW_NORMAL)
cv2.imshow('test',img)

while(cv2.waitKey(0)!= 27):
    pass


cv2.destroyAllWindows()


"""
depth_path= "/media/WIP/depth/155845253177486610475.npz"
depth_img=io.imread(depth_path)

(rows,cols) = depth_img.shape
M = cv2.getRotationMatrix2D((rows/2, cols/2), 180, 1)
depth_image = cv2.warpAffine(depth_img, M, (rows, cols))
cv2.imshow("depth", depth_image)
while(cv2.waitKey(0)!= 27):
    pass


# extinsic base_link to Camera
tx = 0.0
ty = 0.0
tz = 0.0
y = 0#np.pi/2
p = -np.pi/2
r= 0#-np.pi/2

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

Rzy = np.matmul(Rz,Ry)
R = np.matmul(Rzy,Rx)

H = np.zeros((4,4))
H[:3,:3] = R
H[3,3] = 1.0
H[:,3] = [tx,ty,tz,1]
"""
