import numpy as np
import scipy

#This function takes in the size of the modeling space
#and the number of desired modeling points
#and generates 2 nx by ny arrays that encode the x coordinates and the y coordinates
#for many gratings this is the fundamental object that is manipulated to create the desired shapes
def generateCoordinates(xLength,yLength,nx,ny):
    xArray, yArray = np.meshgrid(np.linspace(0,xLength,nx),np.linspace(0,yLength,ny),indexing='xy')
    return xArray, yArray

#this takes in a single coordinate array i.e. x or y
#a desired depth and desired period and applys a square wave 
#across the given array
def oneDimensionBinary(coordArray,depth,period,duty = 1):
    """
    coordArray: 2d array from generateCoordinates representing the x or y axis
    depth: the value you want the peaks of the square wave to have
    period: the period you want the square wave to have
    duty: the fraction of the square wave peak that will remain
    """
    
    periodicX = np.mod(coordArray,period)
    output = np.where(periodicX < period*duty/2,depth,0)

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

#takes in a one dimensional function, and generates a 2d grating with that function as the 
#depth profile
def oneDimensionArbitrary(xFunction,nx,periodPix,thickness,depth,period,otherParams = []):
    """
    xFunction: Function which we only need to input the x axis into
    nx = number of pixels total in 1d
    periodPix = number of pixels per a period
    thickness : thickness of the silicon nitride grating
    depth : maximum depth of the grating to be milleda
    period : microns
    otherParams: optional list of other parameters to pass to xFunction
    """
    
    #set our length to mesh nicely with the pre defined values above
    length = (nx -1)/ periodPix * period 
    print("array length in microns: ",length)
    #define a coordinate array for the x-axis
    xarr,yarr = generateCoordinates(length,length,nx,nx)

    #apply the function to the calculated x array
    paramList = [thickness,depth,period]
    gratingArr = xFunction(xarr,paramList)
    
    return xarr, gratingArr

def optTrip(x,paramList):
    thickness = paramList[0]
    depth = paramList[1]
    period = paramList[2]
    
    a = 2.65718
    height = 1.211 #this depends on a and was found using desmos
    return thickness - depth * ((np.arctan(a * np.sin(x * np.pi * 2 / period)) + height) / (height*2))

#The mismatch of period, pixel number, and length, leads to beating in the modulus 
#fixed in the new blazed grating function
# #alternatively you could use fourier coefficients to define these grating profiles
# #this is how AET did it in her thesis, and there may be merit in this approach
# def oneDimensionBlazed(coordArray,depth,period) :
#     output = depth  * np.mod(coordArray / period,1)
#     return output

#given a grating array, generate a grid of evenly spaced copies of that
#grating. I can imagine a future functionality where this function can take in a 
#list of gratings, and arranges them into a grid pattern, but thats for the future
def genGratingGrid(gratingArray,rowNum,colNum,spacing,\
                   horzSize,vertSize,spacerValue = 0):
    """
    gratingArray: Input grating we wish to make grid out of
    horzSize: real space width of gratingArray
    vertSize: real space height of gratingArray
    spacing: real space spacing between gratings in the grid
    rowNum: how many rows you want in the final grating grid
    colNum: how many columns you want in the final grating grid
    """

    ny = gratingArray.shape[0]
    nx = gratingArray.shape[1]
    #convert real space distances to pixels 
    spacingHorzPixels = round(spacing / (horzSize / nx))
    spacingVertPixels = round(spacing / (vertSize / ny))
    #create the generic spacing arrays we will append between grating arrays
    spacingArrayHorz = np.full((spacingVertPixels, nx),spacerValue)
    spacingArrayVert = np.full((ny + spacingVertPixels,spacingHorzPixels),spacerValue)

    #create a row of evenly spaced grating arrays with a 
    #spacer on the bottom
    rowArr = np.append(spacingArrayHorz,gratingArray,axis = 0)
    for i in range(colNum-1):
        newArr1 = np.append(rowArr,spacingArrayVert ,axis = 1)
        newArr2 = np.append(spacingArrayHorz, gratingArray,axis = 0)
        rowArr = np.append(newArr1,newArr2,axis = 1)

    #append each row of arrays into a total array
    oldRowArr = rowArr
    for j in range(rowNum-1):
        oldRowArr = np.append(oldRowArr,rowArr,axis = 0)

    #remove top spacer for subindexing to be easier at a later point
    testArr = oldRowArr[spacingHorzPixels:,:]

    #give back the array 
    return testArr
