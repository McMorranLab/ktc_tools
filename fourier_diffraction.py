
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

#simple function for calculating the electron wavelength
def calcWavelength(acceleratingVoltage):
    
    krel = (1 / (hbar*c)) * np.sqrt(acceleratingVoltage**2 + 2*acceleratingVoltage*me) # 1 / meters #wave number of the electron
    return 2*np.pi / krel

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
    sigUmip = calcSigmaUmip(KE) *1E-9 #now in rad/nm
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
    fourierTrans = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padGrating)))
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


#this function takes in a 1d periodic curve defining a grating, and calculates the 
#fourier coefficients of the curve in each diffraction order
def gratingCoefficientsFFT(function1d,pixPerPeriod):
    """"
    function1d: 1d periodic array of y values
    pixPerPeriod: user defined when defining function 1d

    orderCoefficients: the fourier coefficients of said periodic curve where 
    orderCoefficient[m] is the mth fourier coefficient of function1d
    orderLabels: an array of the same length as orderCoefficients for labeling plots
    """

    #extract n and use it to define the frequency domain
    n = function1d.shape[0]
    frequencies = np.fft.fftfreq(n)
    
    #rescale the frequency domain so that each diffraction order 
    #is represented by an integer
    orderArray = frequencies*pixPerPeriod
    #extract indices where the diffraction orders are located 
    orderIndices = np.where(orderArray % 1 == 0)[0]
    #extract order labels
    orderLabels = orderArray[orderIndices]

    fourierTransform = np.fft.fft(function1d) #fft
    normFourierFunc = normalizeWavefunc(fourierTransform) #normalize

    #extract the efficienies for each diffraction order
    orderCoefficients = normFourierFunc[orderIndices]

    return orderCoefficients, orderLabels

def normalizeWavefunc(wavefunc):
    magnitude = np.abs(wavefunc)**2
    normFactor = np.sqrt(np.sum(magnitude))
    normalizedWavefunc = wavefunc / normFactor
    
    return normalizedWavefunc

def rescaleArray(targetSizeArray,transformArray):
    targetSize = targetSizeArray.shape[0]
    transformSize = transformArray.shape[0]
    zoomFactor = targetSize / transformSize
    rescaledArray = scipy.ndimage.zoom(transformArray,zoomFactor)
    return rescaledArray

#Function to predict the intensity from a single grating diffraction experiment
def singleGratingDiffraction(grating,acceleratingVoltage):
    """
    grating: 2d numpy array describing the grating
    acceleratingVoltage: float for the accelerating voltage in eV
    """

    #compute the effects of the grating
    g1psi1 = postHoloWaveFunc(grating,acceleratingVoltage)
    #add a circular aperature
    g1psi1Ap = circleAperture(g1psi1,rfrac = .4)
    #normalize cuz why not
    g1psi1Norm = normalizeWavefunc(g1psi1Ap)
    #use fourier transforms to propagate the wavefunction down the column
    diffraction = fourierPropogateDumb(g1psi1Norm,wavefunc = True,pad = 0)
    #give the resulting diffraction pattern back
    return diffraction

#function to simulate a two grating interferometry setup
def twoGratingInterferometry(grating1,grating2,gratingLength,acceleratingVoltage, sampleArray = 1):
    """
    grating1: 2d numpy array representing the first grating in the interferometer
    grating2: 2d numpy array representing the second grating in the interferometer
    gratingLength: float representing the length of the grating in microns
    acceleratingVoltage: float representing the accelerating voltage to be used in the experiment
    sampleArray: 2d numpy array representing the effect of a sample upon the first gratings probe beams
                0 represents blocking 1 represents perfect transmittance and e^(i*phase) represents a phase
                shift introduced to the beams this can be shaped to be applied to one or both probe beams
    """

    #do the whole IFM model
    #this should be the wavefunction of the electron directly after the first grating
    psi1 = postHoloWaveFunc(grating1, acceleratingVoltage)

    #apply the circular aperature
    psi1  = circleAperture(psi1,rfrac = .4)
    #normalize to save from rounding errors
    psi1Norm = normalizeWavefunc(psi1)
    #propogate that wavefunction down to  L1
    L1 = fourierPropogateDumb(psi1Norm,wavefunc = True, pad = 0)

    #find and plot the beam efficiencies at the L1 plane
    xax, L1beams = calcDiffractionEfficiencies(np.abs(L1)**2, gratingLength)
    #apply an aperature that isolates the two beams
    L1aperture = twoBeamAperture(L1beams, L1)

    #apply the effect of the sample
    L1sample = L1aperture * sampleArray

    #propogate the aperture 
    beamFactor = fourierPropogateDumb(L1sample,wavefunc = True,pad = 0)
    beamTrans = 0
    transBeamFac = np.roll(beamFactor,beamTrans,axis = 1)
    beamFactor = transBeamFac
    #apply the second grating
    holoFactor = postHoloWaveFunc(grating2,acceleratingVoltage)

    L2 = holoFactor * beamFactor
    #propogate one more time to get to the detector plane
    L3 = fourierPropogateDumb(L2,wavefunc = True,pad = 0)
    #normalize the wavefunction
    L3norm = normalizeWavefunc(L3)
    # L3norm = np.fft.fftshift(L3norm)
    intensity = np.abs(L3norm)**2

    #return the intensity as observed on the detector
    return intensity
