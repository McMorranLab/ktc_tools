import numpy as np
import scipy

#This function takes in the size of the modeling space
#and the number of desired modeling points
#and generates 2 nx by ny arrays that encode the x coordinates and the y coordinates
#for many gratings this is the fundamental object that is manipulated to create the desired shapes
def generateCoordinates(xLength,yLength,nx,ny):
    xArray, yArray = np.meshgrid(np.linspace(0,xLength,nx),np.linspace(0,yLength,ny))
    return xArray, yArray

#this takes in a single coordinate array i.e. x or y
#a desired depth and desired period and applys a square wave 
#across the given array
def oneDimensionBinary(coordArray,depth,period):
    output = depth/2 * (scipy.signal.square(2 * np.pi  * coordArray / period) + 1)
    return output

def oneDimensionSin(coordArray,depth,period):
    output = depth/2 * ( np.sin(2 * np.pi * coordArray / period) + 1)
    return output

def oneDimensionBlazed(nx,periodPix,thickness,depth,period,fracPos):
    """
    following the convention defined in CWJ thesis 
    
    nx = number of pixels total in 1d
    periodPix = number of pixels per a period
    xarr : xaxis, defining the hologram
    thickness : thickness of the silicon nitride grating
    depth : maximum depth of the grating to be milled
    period : in microns
    fracPos : fraction of the grating with a positive slope
    """
    #set our length to mesh nicely with the pre defined values above
    length = (nx -1)/ periodPix * period 
    print("blazed array length: ",length)
    #define a coordinate array for the x-axis
    xarr,yarr = generateCoordinates(length,length,nx,nx)
    #since the below formula is only valid for one period, redefine the x-axis
    xPeriod = np.mod(xarr,period)
    #apply the blazed grating formula to a single period of x values
    term1 = np.heaviside(period * fracPos - xPeriod,.5) * xPeriod / (period * fracPos)
    term2 = np.heaviside(xPeriod - period*fracPos,.5) * (xPeriod - period) / (period * (1 - fracPos))
    gratingArr = thickness - depth*(term1 - term2)
    
    return xarr, gratingArr

def oneDimArbitrary(function,nx,periodPix,thickness,depth,period):
    
    pass

#The mismatch of period, pixel number, and length, leads to beating in the modulus 
#fixed in the new blazed grating function
# #alternatively you could use fourier coefficients to define these grating profiles
# #this is how AET did it in her thesis, and there may be merit in this approach
# def oneDimensionBlazed(coordArray,depth,period) :
#     output = depth  * np.mod(coordArray / period,1)
#     return output

