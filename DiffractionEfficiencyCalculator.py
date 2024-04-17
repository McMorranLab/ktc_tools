#%% #to allow notebook style development in VSCode
from ktc_tools import grating_generation as ggen
from ktc_tools import fourier_diffraction as fd
from matplotlib import pyplot as plt
import seaborn as sn
import numpy as np


"""The code for a binary grating
"""
# xlength = 10 #microns
# n = 500

# period = .2 #100 nanometers
# depth = 20.75
# padlength = int(n * .2)

# #create a grating of arbitrary depth to diffract the electron beam and create
# # multiple probes
# xarr, yarr = ggen.generateCoordinates(xlength,xlength,n,n)

# gratingArr = ggen.oneDimensionBinary(xarr,depth,period)

"""The code for a blazed grating
"""
n = 1000
period = .2 #microns
depth = 20.88 #in nanometers
padlength = int(n * .2)

##parameters for new style of grating generation
thickness = 50
pixPerPeriod = n * .02
fracPos = .13

#define a blazed grating and an array of x-coordinates to match it
xarr , gratingArr = ggen.oneDimensionBlazed(n, pixPerPeriod, thickness,\
                                             depth, period, fracPos)

#pull the xlength implicitly from the xarray defined by ggen
xlength = xarr[-1,-1]

fig,ax = plt.subplots()
ax.set_title("Grating Array")
# plt.imshow(gratingArr[0:200,0:200]) #zoomed in version
plt.imshow(gratingArr)
plt.colorbar()
plt.show()

#this should be the wavefunction of the electron directly after the grating
psi1 = fd.postHoloWaveFunc(gratingArr, 80*10**3)
#apply a circular aperature after the wavefunction is calculated
#to create the region of zero amplitude
psi1 = fd.circleAperture(psi1,.4)

#propogate that wavefunction down to the detector plane
inten2 = fd.fourierPropogateDumb(psi1)
fig2,ax2 = plt.subplots()

ax2.set_title("Intensity on Detector")
fd.intensPlot(inten2, ax = ax2)

#calculate the diffraction efficiencies and plot
xax, efficiencies = fd.calcDiffractionEfficiencies(inten2, xlength)
plt.plot(xax,efficiencies)
plt.show()

# %%
