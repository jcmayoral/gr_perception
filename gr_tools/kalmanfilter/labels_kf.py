from filterpy.kalman import KalmanFilter
import numpy as np

f = KalmanFilter(dim_x = 2, dim_z=2)
f.x = np.array([[2.], #position
                [0.]]) #velocity
#State transition matrix
f.F = np.array([[1.,1.],
                [0, 1.]])
#measurement function
f.H = np.array([[1.,0.]])
#covariance matrix
#P pe contains np.eye(dim_x)
f.P *=1000
#or
#f.P = np.array([[1000.,0],
#               [   0., 1000.] ])

#measurement noise / state certainty
f.R = 4
#f.R = np.array([[5.]])


#process noise
from filterpy.common import Q_discrete_white_noise
f.Q = Q_discrete_white_noise(dim=2, dt=0.1, var=0.13)

#standard predic/update loop pseudocode
"""
z = get_sensor_reading()
f.predict()
f.update(z)
do_something_with_estimate(f.x)
"""


f.x = np.array([[2.],[0.]])     # initial state (location and velocity)

f.F = np.array([[1.,1.],
                [0.,1.]]).T    # state transition matrix

f.H = np.array([[1.,0.]])    # Measurement function
f.P *= 1000.                 # covariance matrix
f.R = 5                      # state uncertainty
dt = 0.1
f.Q = Q_discrete_white_noise(2, dt, .1) # process uncertainty

z = np.squeeze(np.array( [[0],[1]]))

while True:
    print f.x.ndim, z.ndim, f.K.ndim, f.y.ndim
    print "A"
    f.predict()
    print "B"
    print f.y, f.x, f.K
    f.update(z.T)
    #do_something_with_estimate(f.x)
    print ("OUTPUT", f.x)
