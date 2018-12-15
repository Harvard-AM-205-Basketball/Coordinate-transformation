import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as misc
from math import *
from numpy import vectorize
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
warnings.filterwarnings("ignore",category=FutureWarning)
import imageio

# Imports frame as matrix of (pixel y-coordinate, pixel x-coordinate, color value)
# shape=(1080, 1920, 3)
im = imageio.imread('Camera3_Frame00020.png')
print(im.shape)

# Write as a linear system and solve for intrinsic parameters
#-----------------------------------------------------------------------------#
# PARAMETERS:
# Focal length (0.024147 ft is from manufacturer), changes with zoom
f=0.024147

# Position of camera (adjusted for height of tripod)
campos=np.array([22,-1.5,5.2])

# Parameters for rotation matrix to get from real-world to camera
# Camera z-axis in world coords
# Camera 3 is ROUGHLY pointed towards a point 2 feet below the front of the rim
z=np.array([-22,42.5,2.8]) # 2 worked well

# Camera x-axis in world coords
x=np.array([42.5,22,0])


#-----------------------------------------------------------------------------#
# REAL-WORLD COORDINATES:
# Real-world coordinates of foul line run from (-6, 28, 0) to (6, 28, 0)
foul_line=np.zeros((13,3))
foul_line[:,1]=28
foul_line[:,0]=np.arange(-6,7,1)

# Real-world coordinates for right side of free throw box run from (6,47,0) to (6,28,0)
foul_linebox=np.zeros((20,3))
foul_linebox[:,0]=6
foul_linebox[:,1]=np.arange(28,48,1)
# Left side of free throw box
foul_linebox2=np.zeros((20,3))
foul_linebox2[:,0]=-6
foul_linebox2[:,1]=np.arange(28,48,1)
# Baseline part of free throw box
foul_linebox3=np.zeros((13,3))
foul_linebox3[:,1]=47
foul_linebox3[:,0]=np.arange(-6,7,1)

square=np.array([[-1,43,10],[-1,43,11.5],[1,43,10],[1,43,11.5]])

# Concatenate all coordinates
coords=np.vstack((foul_line,foul_linebox,foul_linebox2,foul_linebox3,square))

#-----------------------------------------------------------------------------#
# APPLY TRANSFORMATIONS:
# Normalize each vector and make a rotation matrix R
x=x/np.linalg.norm(x)
z=z/np.linalg.norm(z)

# Get y-axis via cross product
y=np.cross(x,z)

R=np.vstack((x,y,z))
Rt=np.transpose(R)

# Subtract position of camera on tripod
coords=coords-campos

# Apply rotation matrix (easier to use R^T here)
coords=np.matmul(coords, Rt)


# Convert to 2D by using focal length
coords[:,0]=f*coords[:,0]/coords[:,2] # Divide X by Z and multiply by f
coords[:,1]=f*coords[:,1]/coords[:,2] # Divide Y by Z and multiply by f


# Delete z-coordinate (not needed since pixel plane is 2D)
coords = np.delete(coords, 2, 1)


#-----------------------------------------------------------------------------#
# SET UP LEAST SQUARES PROBLEM TO SOLVE FOR PIXEL SIZE
# Coefficient matrix for least squares problem
A=np.zeros((2*coords.shape[0],2)) 
for i in range(coords.shape[0]):
    A[2*i,0]=coords[i,0]
    A[2*i+1,1]=-coords[i,1]

# Pixel coordinates (eventual RHS)
xvals1=np.linspace(290,915,13)
xvals2=np.linspace(915,1330,20)
xvals3=np.linspace(290,855,20)
xvals4=np.linspace(855,1330,13)
xvals5=np.array([958,956,1031,1029])

yvals1=np.linspace(1000,1060,13)
yvals2=np.linspace(1060,915,20)
yvals3=np.linspace(1000,888,20)
yvals4=np.linspace(888,915,13)
yvals5=np.array([456,399,452,395])

# Adjust by 960 or 540 so that we have a linear system
b=np.zeros(2*coords.shape[0])
for i in range(13):
    b[2*i]=xvals1[i]-960
    b[2*i+1]=yvals1[i]-540
for i in range(13,33):
    b[2*i]=xvals2[i-13]-960
    b[2*i+1]=yvals2[i-13]-540
for i in range(33,53):
    b[2*i]=xvals3[i-33]-960
    b[2*i+1]=yvals3[i-33]-540
for i in range(53,66):
    b[2*i]=xvals4[i-53]-960
    b[2*i+1]=yvals4[i-53]-540
for i in range(66,70):
    b[2*i]=xvals5[i-66]-960
    b[2*i+1]=yvals5[i-66]-540
    
# Solve for least-squres pixel size
s=np.linalg.lstsq(A,b)[0]
    
# Set pixel width and height based on least squares
pixelheight=1/s[1]
pixelwidth=1/s[0]

#-----------------------------------------------------------------------------#
# Pixel coordinates (v, u) of points (with top left pixel as (0,0))
coords[:,0]=coords[:,0]/pixelwidth
coords[:,1]=coords[:,1]/pixelheight

coords[:,0]+=960
coords[:,1]=-coords[:,1]+540


# Round to nearest pixel, change type to int
coords=np.rint(coords)
coords=coords.astype(int)


# Switch columns because image is imported with y-pixel first. Coordinates become (u, v)
temp = np.copy(coords[:, 0])
coords[:, 0] = coords[:, 1]
coords[:, 1] = temp


# Paint foul line with red pixels
for i in range(coords.shape[0]):
    im[coords[i][0], coords[i][1], :]=np.array([0,0,100])
    

imageio.imwrite('Frame20_withLines.jpg', im)


# Testing purposes
add=np.zeros(len(b))
for i in range(len(add)):
    if i%2==0:
        add[i]=960
    else:
        add[i]=540

print((b+add)[26:66])
print((b+add)[66:106])