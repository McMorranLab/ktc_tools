import numpy as np
from matplotlib import pyplot as plt

#function for generating array of beam locations and on/off status of the beam at each location
#I should add a part that controls the dwell time too
def binaryStreamGen(hologram,xstride,ystride,xStreamfilePix,yStreamfilePix):

    xStartind = 0
    yStartind = 0
    
    xmax = hologram.shape[0]
    ymax = hologram.shape[1]
    
    yArray, xArray = np.meshgrid(np.linspace(0,xmax,xmax),np.linspace(0,ymax,ymax))


    #define this array so it occurs in the center of the screen 
    #these values come from the FIB manual
    xdirStreamPixels = 65536
    ydirStreamPixels = 56576
    midpointX = xdirStreamPixels / 2
    midpointY = ydirStreamPixels / 2
    xmaxStream = midpointX + round(xStreamfilePix/2)
    xminStream = midpointX - round(xStreamfilePix/2)
    ymaxStream = midpointY + round(yStreamfilePix/2)
    yminStream = midpointY - round(yStreamfilePix/2)

    #arrays with values of the actual streamfile coordinates to use
    yStreamFileArray, xStreamFileArray = np.meshgrid(np.linspace(xminStream,xmaxStream,xmax)\
                                                   ,np.linspace(yminStream,ymaxStream,ymax))
    
    streamlist = []
    ylast = 0
    
    #more debugging
    print(xmax,ymax)

    #loop through hologram skipping points according to how far apart you want your beam locations
    for j in range(yStartind,ymax,ystride):
        for i in range(xStartind,xmax,xstride):

            #check if beam location is on hologram
            if hologram[i,j] >= 1:
                
                #determine whether to keep the beam on or not
                #if the beam is moving over a large region without points turn off
                ydiff = np.abs(ylast - yArray[j,i])

                #if the current y val is too far from the last y val turn off the beam 
                #for the last streampoint
                if ydiff > 4*ystride and streamlist != []:
                    streamlist[-1][2] = 0
                    
                ylast = yArray[j,i]
                
                #add streampoint to the list
                xStreamPoint = xStreamFileArray[j,i]
                yStreamPoint = yStreamFileArray[j,i]
                streamlist.append([xStreamPoint,yStreamPoint,1])
                
    #repeat the last point
    if streamlist: #make sure we didn't pass a blank array or something
        streamlist.append(streamlist[-1])
        #change the beam to be blanked to avoid milling outside the desired areas between passes
        streamlist[-1][2] = 0
    
    #turn list into array
    streamArr = np.array(streamlist)
                
    return streamArr


#function that takes a stream array and plots it with an option for zooming
def plotStreams(streamArray,zoom = False):
    
    #inaccurate quick and dirty maxes for the zoom
    xmax = streamArray[-1][0]
    ymax = streamArray[-1][1]
    
    plt.scatter(streamArray[:,0],streamArray[:,1], s=0.1)

    #here we isolate only the points that the beam is turned off for
    #and plot those with red
    Boolarr = streamArray[:,2] == 0
    beamarr = streamArray[Boolarr]
    
    plt.scatter(beamarr[:,0], beamarr[:,1],s = 1, c='red')
    
    if zoom:
        plt.gca().set_xlim(xmax * .8, xmax)  # Adjust the range according to your data
        plt.gca().set_ylim(ymax * .8, ymax) 

    plt.show()
    

#take the stream array and the necessary parameters and generate a stream file format
#the dwell time is in units of .1 microseconds
"""may want to add a functionality to use a base name and incorporate
num_Passes and dwellTime into the filename when it is saved going to wait for now and
just build the simple version
"""
def generateStreamFile(streamArray,numPasses,dwellTime,fileLoc):
    
    pointNum = streamArray.shape[0]
    
    f = open(fileLoc,"w+")
    f.write("s16\n")
    f.write(str(int(numPasses)) + "\n")
    f.write(str(int(pointNum)) + "\n")
    
    for i in range(0,pointNum):
        
        xcoord = int(streamArray[i][0])
        ycoord = int(streamArray[i][1])
        beamCond = int(streamArray[i][2])
        
        line = str(dwellTime) + " "  + str(xcoord) + " " + str(ycoord) + " "  + str(beamCond) + "\n"
        
        f.write(line)
    
    return None

## STREAM FILE FORMATTING
# beam_con 1 for beam on, 0 for beam off, nothing for don't change beam_con
"""
1st line: s16e
2nd line: number of passes
3rd line: number of points
4th line and rest:
Dwell_time1 x_coord1 y_coord1 beam_con
Dwell_time2 x_coord2 y_coord2 
Dwell_time3 x_coord3 y_coord3 beam_con
"""

# #this function takes a list of streamfile names all within the same folder
# #and it writes an autoscript file to run all the streamfiles in order
#useless since we can't use autoscripts on fib2 without some expensive software
# def basicAutoscript(streamNameList,fileLoc):
    
#     f = open(fileLoc,"w+")

#     for streamName in streamNameList:

#         #this tells the FIB to load a streamfile into memory
#         streamline = "patternfile " + streamName + "\n"
#         f.write(streamline)
#         #this line tells the FIB to mill the currently loaded streamfile
#         f.write("mill\n")
        
#     print("all done!")
    
    
def streamConversions(hfw,millDens,gratingLength,calcRes):
    """
    hfw: Half Field Width in microns
    millDens: milling points per micron
    gratingLength: length of grating in microns
    calcRes: number of points to use for calculation of streamfiles
    
    Returns
    millSpacing: number of grating array pixels that seperate mill points
    lengthPix: width of grating in streamfile pixels
    """
    #find the pixel resolution of the fib at whatever magnification we are at
    xpointspacing =  65536 / hfw #in units of pixels / micron
    #use thaat pixel resolution and the mill point density to find the pixel spacing between mill points
    pointMillSpacing = (1 / millDens) * (xpointspacing) #now has units of pixels / mill point
    #determine how many streamfile pixels across our grating will be
    lengthPix = gratingLength*xpointspacing
    #take the array resolution and the test area size in pixels to find the pixel resolution
    #of the grating arrays
    xGratPix  = lengthPix / calcRes
    yGratPix  = lengthPix / calcRes
    #finally calculate how many array pixels each mill spacing corresponds to 
    millSpacing = int(np.round(pointMillSpacing/xGratPix))
    
    print("each returned array pixel represents")
    print(xGratPix,"by",yGratPix, "streamfile pixels\n")
    print("mill point spacing values in streamfile pixels")
    print(pointMillSpacing,"\n")
    print("the mill points will be seperated by ")
    print(millSpacing, " pixels in the returned arrays\n")
     
    return millSpacing, lengthPix

def sliceCalculations(depth,layerNum,nmPerPass):
    """
    depth: depth of grating in nanometers
    layerNum: number of layersDesired
    nmPerPass: experimental value gathered through depthTesting
    """
    
    dh = depth / layerNum
    print("deltaH (nm): ",dh)
    passPerLayer = round(dh / nmPerPass)
    print("Passes / Layer: ", passPerLayer)
    
    return dh, passPerLayer


#this function is for generating a folder full of streamfiles representing horizontal 
#slices of some depth profile
def sliceStream(folder,baseName,\
                numLayers,depth,nmPerPass,\
                gratingArr,gratSpacing,lengthPix,dwellTime):
    
    dh, passPerLayer = sliceCalculations(depth,numLayers,nmPerPass)
    
    #usual thickness of our membranes
    thickness = 50 #nm
    
    for i in range(1,numLayers):
    
        layerArr = np.copy(gratingArr)
        layerArr[gratingArr+i*dh > thickness] = 0
        layerArr[gratingArr+i*dh <= thickness] = 1

        #generate the stream array
        layerStream = binaryStreamGen(layerArr,gratSpacing,gratSpacing,lengthPix,lengthPix)

        if not layerStream.any():
            print("skipped empty layer")
            continue

        plotStreams(layerStream)
        #define streamfile name
        streamName = baseName + '{:02d}'.format(i) + ".str"
        #create streamfile 
        generateStreamFile(layerStream,passPerLayer,dwellTime,folder + streamName)