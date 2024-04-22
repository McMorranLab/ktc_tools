
import numpy as np
import scipy
import seaborn as sb
from matplotlib import pyplot as plt

hbar = 6.582 * 10**(-16) # eV s (electronvolt seconds)
c = 2.998*10**8 # m/s  (meters per second)
me = 0.5109989461 * 10**6 # eV / c^2

KE = 300 * 10**3 # eV # Kinetic Energy of the Electron
krel = (1 / (hbar*c)) * np.sqrt(KE**2 + 2*KE*me) # 1 / meters #wave number of the electron
#make this coefficient a % < 100 to account for momentum in transverse direction
kz = 1 *krel #wave number in the z direction
aperatureRad = 5 * 10**-6 # try a 1 micron aperature

#from Cameron Johnson's Thesis these values may need to be updated
alphaDecay = 0.008 * 10**9 # in units of 1 / meters
meanInnPot = 15 #Volts

#grating parameters
membraneThickness = 50*10**-9

#calculating the sigma parameter which stands for the interaction 
#parameter in CWJ thesis
def calcSigmaUmip(kineticEnergy = 80*10**3):
    #this mass matches online calculators
    mrel = me + kineticEnergy
    h = 2 * np.pi * hbar
    wavelengthDenom =  np.sqrt(kineticEnergy**2 + 2 * kineticEnergy * me )
    #this wavelength matches online calculators 
    wavelength = h * c / wavelengthDenom
    #combining umip and electron charge since their combined units are just eV
    eChargeUmip = 15 #eV
    sigmaUmip = eChargeUmip * (mrel/ c**2) * wavelength / (2 * np.pi * hbar**2)
        
    return sigmaUmip #in units of rad / m 

#determine what depth of SiNx is needed to achieve the desired phase shift
def radiansToDepth(radians,KE):
    """
    radians: the desired phase shift in radians
    KE: the kinetic energy in eV to do the calculation for
    """
    sigUmip = fd.calcSigmaUmip(KE) *1E-9 #now in rad/nm
    #returns depth of SiNx in nanometers
    return radians / sigUmip

#with all the wisdom and knowledge ive gained in the past year
#writing a new version of fourierPropogate which uses values 
#more close to unity and not include any physical constants
def fourierPropogateDumb(grating, wavefunc = False, pad = .7):
    """
    grating: array which represents in nanometers the thickness of the grating

    wavefunc: boolean that if true makes this function return a complex
    wavefunction instead of an intensity array

    pad: fraction that determines how much bigger the array becomes after padding
    set to 0 for no padding
    """
    #padding because that is what the old gods requested
    padlength = int(grating.shape[0] * pad)
    padGrating = np.pad(grating, [padlength, padlength], mode='constant', constant_values=0)
    #find fourier transform of transmission function, now (0,0) is at center of grid in frequency space
    fourierTrans = np.fft.fftshift(np.fft.fft2(padGrating))
    #do some transformations to get back into real space
    intensity = np.abs(fourierTrans)**2
    
    if wavefunc:
        output = fourierTrans
    else:
        output = intensity
        
    return output

#using experimental values from CWJ thesis modelling the 
#electron wavefunction immediately after a phase grating
#units in this function are in nanometers so use nanometers in definition of
#the grating
def postHoloWaveFunc(grating, accVoltage  = 80 * 10**3):
    sigUmip = calcSigmaUmip(accVoltage) * 10**-9 #now in units of rad / nm
    alpha = .008 #1/nm
    phiTwiddle = sigUmip + 1j * alpha
    psi = np.exp(1j * phiTwiddle * grating)
    return psi

#define a function that computes the coefficients c_sub-n from AET thesis
def gratingCoefficients(xarr,gratingArr,period,accVoltage,nmax):
    
    k1 = 2 * np.pi / period
    
    #this will ensure we only integrate a single period
    periodMask = (xarr < period)
    
    coefList = []
    
    for n in range(-nmax,nmax+1):
        
        gratingTerm = gratingArr
        fourierTerm = np.exp(-1j * n * k1 * xarr)
        integrand = gratingTerm * fourierTerm
        cn = (1 / period )* scipy.integrate.trapz(integrand[periodMask],xarr[periodMask])
        
        coefList.append(cn)
        
    return coefList

def twoBeamAperture(efficiencyArr,waveFuncArr):
    #if your grating has been optimized for two beam diffraction, then the two 
    #beams should have efficiencies of at least .2, and be the only beams with 
    #values greater than .2
    locArr,heightArr = scipy.signal.find_peaks(efficiencyArr,height = .1)
    locArr = np.sort(locArr)
    beamSeparation = locArr[1] - locArr[0]
    #define the edges of our aperature based on beam location and separation
    rightEdge = round(locArr[1] + beamSeparation/2)
    leftEdge = round(locArr[0] - beamSeparation/2)
    waveFuncArr[:,rightEdge:] = 0
    waveFuncArr[:,:leftEdge] = 0
    
    return waveFuncArr

def oneBeamAperture(efficiencyArr,waveFuncArr):
    #if your grating has been optimized for two beam diffraction, then the two 
    #beams should have efficiencies of at least .2, and be the only beams with 
    #values greater than .2
    locArr,heightArr = scipy.signal.find_peaks(efficiencyArr,height = .1)
    locArr = np.sort(locArr)
    beamSeparation = locArr[1] - locArr[0]
    #define the edges of our aperature based on beam location and separation
    rightEdge = round(locArr[1] + beamSeparation/2)
    leftEdge = round(locArr[1] - beamSeparation/2)
    waveFuncArr[:,rightEdge:] = 0
    waveFuncArr[:,:leftEdge] = 0
    
    return waveFuncArr

#applies a circular aperture to the center of the inputArr, rfrac defines the 
#radius of the circle as a fraction of the arrays width
def circleAperture(inputArr,rfrac = .2):
    
    width = inputArr.shape[0]
    r = rfrac*width
    cx = width / 2
    cy = cx
    
    x = np.linspace(0,width,width)
    y = np.linspace(0,width,width)

    mask = ((x[np.newaxis,:] - cx)**2 + (y[:,np.newaxis] - cy)**2) > r**2
    inputArr[mask] = 0
    
    return inputArr
    

def logintensSurf(xmesh,ymesh,intensity,ax = None,cbar = True):

    #set resolution
    # plt.figure(dpi=200)
    flattened_arr = intensity.flatten()
    array_2d = np.log(intensity)
    
    # Create the surface plot with color
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_axis_off()
    surface = ax.plot_surface(xmesh, ymesh, array_2d, cmap='inferno')

    # Set labels for x, y, and z axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Add a colorbar
    fig.colorbar(surface, ax=ax)

    # Enable click and drag rotation
    ax.mouse_init()
    
def intensPlot(intensity, percentileCutoff = 99.99, ax = None, cbar = True):
    
    #set resolution
    plt.figure(dpi=200)
    flattened_arr = intensity.flatten()
    
    #the lower the percentile chosen for max the more fine structure becomes visible
    maxval = np.percentile(flattened_arr,percentileCutoff)
    # maxval = np.amax(intensity)
    minval = np.amin(intensity)
    
    sb.heatmap(intensity,vmin = minval,vmax = maxval,ax = ax,cbar = cbar)
    
def logintensHeat(intensity, ax = None, cbar = True):
    
    logint = np.log(intensity)
    #set resolution
    plt.figure(dpi=200)
    flattened_arr = logint.flatten()

    sb.heatmap(logint,ax = ax,cbar = cbar)
    
def valueDistribution(intensity):
    fig1, ax1 = plt.subplots()
    plt.yscale("log")
    plt.title("Distribution of Intensity Values")
    ax1.hist(intensity.ravel(),bins = 100)
    
#takes the 2d intensity array and calculates the intensity in each order
def calcDiffractionEfficiencies(intensity, xlength):
   
    arrayLength = intensity.shape[0]

    # #integrate the intensity over the direction perpendicular to the 
    integratedIntensity = np.trapz(intensity,axis = 0)
    totalIntensity = np.trapz(integratedIntensity)
    #normalize so that the efficiencies are in % of the total
    efficiencies = integratedIntensity / totalIntensity
    
    #take the array shape and xlength to make an array to represent the xaxis
    xaxis = np.linspace(0,xlength,arrayLength)
    
    return xaxis, efficiencies

def normalizeWavefunc(wavefunc):
    magnitude = np.abs(wavefunc)**2
    normFactor = np.trapz(np.trapz(magnitude,axis = 0))
    normalizedWavefunc = wavefunc / normFactor
    
    return normalizedWavefunc

def rescaleArray(targetSizeArray,transformArray):
    targetSize = targetSizeArray.shape[0]
    transformSize = transformArray.shape[0]
    zoomFactor = targetSize / transformSize
    rescaledArray = scipy.ndimage.zoom(transformArray,zoomFactor)
    return rescaledArray


# #I don't trust this function any more
# #dont use until it is heavily inspected
# def fourierPropogate(grating,zval):
#     #Defining the transmission function
#     cConst = 2*np.pi * (KE + me) / ((wavelength * accVoltage)*(KE + 2*me))
#     ktwiddle = cConst * meanInnPot + 1.0j * alphaDecay

#     transFunc = np.exp(grating * 1.0j * ktwiddle)
#     #padding because that is what the old gods requested
#     padlength = int(transFunc.shape[0] * .1)
#     transFunc = np.pad(transFunc, [padlength, padlength], mode='constant', constant_values=0)

#     #find fourier transform of transmission function, now (0,0) is at center of grid in frequency space
#     fourierTrans = np.fft.fftshift(np.fft.fft2(transFunc))
    
#     #do some transformations to get back into real space
    # wavefunc = (2*np.pi)**2 * np.exp(1.0j * krel * zval) * fourierTrans / (1.0j * wavelength * zval)
    # intensity = np.abs(wavefunc)**2
    
    # return intensity