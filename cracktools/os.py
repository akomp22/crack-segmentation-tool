import numpy as np
import scipy
import cmath
import math
import sys
import skimage
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import scipy.signal
from skimage.filters import threshold_otsu
# from skimage import filters
# from skimage import measure



def ErfSet(size,No,periodicity):
    """ErfSet retuns a set of 2 D error functions.This function is used to \
        cut the wavelets in two (in the spatial domain)"""
    out = np.zeros((No,size,size))
    for i in range(1,No+1):
        xx = -1
        for x in np.arange( -((size - 1) / 2) - ((size - 1) / 2)%1, (size - 1) / 2 - ((size - 1)/2)%1 +1):
            xx += 1
            yy = -1
            for y in np.arange( -((size - 1) / 2) - ((size - 1) / 2)%1, (size - 1) / 2 - ((size - 1)/2)%1 +1):
                yy += 1
                out[i-1,xx,yy] = 1/2 * (1 + scipy.special.erf(x * np.cos( (periodicity*(-1+i))/ No)\
                                                              + y * np.sin((periodicity*(-1+i))/No)))
    return out

def WindowGauss(size,sigma_s):
    """WindowGauss retuns the spatial Gauss envelope"""
    out = np.zeros((size,size))
    i = -1
    j = -1
    for x in np.arange( (- ((size - 1) / 2) - (size - 1) / 2 % 1), (size - 1) / 2 - (size-1) / 2 % 1 + 1,1):
        i = i+1
        j = -1
        for y in np.arange( (- ((size - 1) / 2) - (size - 1) / 2 % 1), (size - 1) / 2 - (size-1) / 2 % 1 + 1,1):
            j = j+1
            out[i,j] = math.e**( -( x**2 / (2 * sigma_s ** 2 )) -  y**2 / (2 * sigma_s ** 2 ))
            
    return out

def PolarCoordinateGridAngular(size):
    """
    PolarCoordinateGridRadial returns a matrix in which each element \
    gives the corresponding radial coordinate (with the origin in the \
    center of the matrix
    """
    m = np.zeros((size,size))
    centerx = np.ceil((size-1)/2)
    centery = centerx
    for i in range(size):
        for j in range(size):
            dx = i-centerx
            dy = j-centery
            m[i,j] = cmath.phase(complex(dx,dy))
    return m

def PolarCoordinateGridRadial(size):
    """
    PolarCoordinateGridRadial returns a matrix in which each element \
    gives the corresponding radial coordinate (with the origin in the \
    center of the matrix
    """
    m = np.zeros((size,size))
    centerx = np.ceil((size-1)/2)
    centery = centerx
    for i in range(size):
        for j in range(size):
            dx = centerx-i
            dy = centery-j
            m[i,j] = (np.sqrt(dx**2 + dy**2) + sys.float_info.epsilon) / ((size - 1) / 2)
    return m

def MnWindow(size,n,inflectionPoint):
    """MnWindow gives the radial windowing matrix for sampling the fourier \
        domain"""
    
    eps = sys.float_info.epsilon
    po_matrix = eps + 1 / (np.sqrt(2 * inflectionPoint**2 / (1 + 2*n))) * PolarCoordinateGridRadial(size)
    s = np.zeros_like(po_matrix)
    for k in range(n+1):
        s = s + math.e**(-po_matrix**2) * po_matrix**(2*k) / np.math.factorial(k)
    return s

def BSplineMatrixFunc(n,x):
    eps = sys.float_info.epsilon
    r = 0
    for i in np.arange((1 - n - 1) / 2, (n - 1 + 1) / 2+1):
        s = 0
        for k in range(n+2):
            binom_cof = scipy.special.binom(n+1, k)
            sign = np.sign(i + (n + 1) / 2 - k)
            s += binom_cof * (x + (n + 1) / 2 - k) ** (n + 1 - 1) * (-1)**k * sign

        f = 1/(2 * np.math.factorial(n+1-1)) * s
        if i < (n+1-1)/2:
            ic = np.heaviside((x - (i - 1/2 + eps)), 1) * np.heaviside(-(x - (i + 1/2)), 1)
        else :
            ic = np.heaviside((x - (i - 1/2 + eps)), 1) * np.heaviside(-(x - (i + 1/2 - eps)), 1)
        
        r += f*np.round(ic)
    return r

def CakeWaveletStackFourier(size, sPhi, splineOrder, overlapFactor, inflectionPoint, mnOrder, dcStdDev,
                            noSymmetry):
    """CakeWaveletStackFourier constructs the cake wavelets in the Fourier \
        domain (note that windowing in the spatial domain is still required \
        after this"""
    dcWindow = np.ones((size,size)) - WindowGauss(size,dcStdDev)
    mnWindow = MnWindow(size, mnOrder, inflectionPoint)
    angleGrid = PolarCoordinateGridAngular(size)
    sPhiOverlapped = sPhi/overlapFactor
    if noSymmetry == True:
        s = 2*np.pi
    else :
        s = np.pi
    
    out = np.array([], dtype=np.int64).reshape(0,size,size)
    for theta in np.arange(0, s, sPhiOverlapped):
        x = mod_offset(angleGrid - theta - np.pi / 2, 2*np.pi, -np.pi) / sPhi 
        f = dcWindow*mnWindow * BSplineMatrixFunc(splineOrder,x) / overlapFactor
        f = np.expand_dims(f,axis = 0)
        out = np.vstack([out,f])
    
    filters = np.vstack([out,np.expand_dims((1-dcWindow),axis = 0)])
    return filters

def CakeWaveletStack(size, nOrientations, design, inflectionPoint, mnOrder, splineOrder, overlapFactor,
                    dcStdDev, directional):
    
    noSymmetry = nOrientations%2 == 1
    dcSigma = (1/dcStdDev)*(size/(2*np.pi))
    filters = CakeWaveletStackFourier(size, 2*np.pi / nOrientations, splineOrder, overlapFactor,
                                      inflectionPoint, mnOrder, dcSigma, noSymmetry)
#     print(filters.shape)
    cakeF = filters[:-1,:,:]
#     print(cakeF.shape)
    dcFilter = filters[-1,:,:]
    if design == "M":
        cakeF = np.sqrt(cakeF)
        dcFilter = np.sqrt(dcFilter)

    cake = np.zeros_like(cakeF,dtype=np.complex_)
    for i in range(cakeF.shape[0]):
        cakeIF = RotateLeft(cakeF[i,:,:],np.floor(np.array([size,size])/2).astype(int))
       
        ##### ifftn gives result not similar to wolfram (gives conjucate)########
        cakeIF = np.conj(np.fft.ifftn(cakeIF))
        
        cakeIF = RotateRight(cakeIF,np.floor(np.array([size,size])/2).astype(int))
        cake[i,:,:] = cakeIF
        
    if directional:
        if not noSymmetry:
            cake = np.vstack([cake,np.conj(cake)])
        cake = cake*ErfSet(size, (overlapFactor*nOrientations), 2*np.pi)
    else :
        if not noSymmetry:
            cake = np.vstack([cake,np.conj(cake)])
    
    return cake

def mod_offset(arr,divv,offset):
    return arr-(arr-offset)//divv*divv

def RotateLeft(arr,k):
    if type(k) == int or type(k) == float:
        arr1 = arr[:k]
        arr2 = arr[k:]
        arr = np.concatenate((arr2,arr1),axis = 0)
        return arr
    if len(k) == 2 and len(arr.shape) == 2:
        arr1 = arr[:,:k[1]]
        arr2 = arr[:,k[1]:]
        arr = np.concatenate((arr2,arr1),axis = 1)
        arr1 = arr[:k[0],:]
        arr2 = arr[k[0]:,:]
        arr = np.concatenate((arr2,arr1),axis = 0)
        return arr
            
def RotateRight(arr,k):
    if type(k) == int or type(k) == float:
        arr1 = arr[:-k]
        arr2 = arr[-k:]
        arr = np.concatenate((arr2,arr1),axis = 0)
        return arr
    if len(k) == 2 and len(arr.shape) == 2:
        arr1 = arr[:,:-k[1]]
        arr2 = arr[:,-k[1]:]
        arr = np.concatenate((arr2,arr1),axis = 1)
        arr1 = arr[:-k[0],:]
        arr2 = arr[-k[0]:,:]
        arr = np.concatenate((arr2,arr1),axis = 0)
        return arr

def CheckWavelet(window_size = 70,size = 75, nOrientations = 32, design = "N", 
                inflectionPoint = 0.9, mnOrder = 8, splineOrder = 3,
                overlapFactor = 1, dcStdDev = 8, directional = False,display_orientations = 0,mode='real'):
    a = np.zeros((window_size, window_size))
    a[np.int(window_size/2),np.int(window_size/2)] = 1
    os_check = OrientationScoreTransform(a, size = size, nOrientations = nOrientations, design = design, inflectionPoint = inflectionPoint, mnOrder = mnOrder, splineOrder = splineOrder,
                              overlapFactor = overlapFactor, dcStdDev = dcStdDev, directional = directional)
    # for i in display_orientations:
    #     if mode=='real':
    #         plt.imshow(os_check[i,:,:].real)
    #         plt.show()
    #     elif mode=='imag':
    #         plt.imshow(os_check[i,:,:].imag)
    #         plt.show()
    return os_check[display_orientations,:,:].real
            
def OrientationScoreTransform(im, size, nOrientations, design = "N", inflectionPoint = 0.8, mnOrder = 8, splineOrder = 3,
                              overlapFactor = 1, dcStdDev = 8, directional = False):
    
    """
    directional     - Determines whenever the filter goes in both directions;
    design          - Indicates which design is used N = Subscript[N, \[Psi]] or M = Subscript[M, \[Psi]]
    inflectionPoint - Is the location of the inflection point as a factor in (positive) radial direction
    splineOrder     - Order of the B - Spline that is used to construct the wavelet
    mnOrder         - The order of the (Taylor expansion) gaussian decay used to construct the wavelet
    dcStdDev        - The standard deviation of the gaussian window (in the Spatial domain) \
                      that removes the center of the pie, to avoid long tails in the spatial domain
    overlapFactor   - How much the cakepieces overlaps in \[Phi] - direction, this can be \
                      seen as subsampling the angular direction
    """
    im = np.pad(im, pad_width=((size, size), (size, size)), mode='symmetric')
    cws = CakeWaveletStack(size, nOrientations, design, inflectionPoint, mnOrder, splineOrder, overlapFactor,
                    dcStdDev, directional)
    
#     print(cws.shape)
    cwsP = np.pad(cws, ([0,0],[np.floor((im.shape[0]-cws.shape[1])/2).astype(int),
                           np.ceil((im.shape[0]-cws.shape[1])/2).astype(int)],
                    [np.floor((im.shape[1]-cws.shape[2])/2).astype(int),
                     np.ceil((im.shape[1]-cws.shape[2])/2).astype(int)]), mode='constant')
    os = WaveletTransform2D(im, cwsP.real)
    os = os[:,size:-size,size:-size]
    return os

def plot_orientation_score(os,display_orientations=[0]):
    for i in display_orientations:
        plt.imshow(os[i,:,:].real,cmap = 'gray')
        plt.show()

def WaveletTransform2D(im, kernels):
    os = np.zeros((kernels.shape[0],im.shape[0],im.shape[1]),dtype=np.complex_)
    imf = np.fft.fftn(im)
    for i in range(kernels.shape[0]):
        v = kernels[i,:,:]
        v = np.fft.fftn(v)
        v = np.fft.ifftn(v * imf)
        v = RotateRight(v, np.ceil( 0.1 + np.array(im.shape) / 2).astype(int))
        os[i,:,:] = v
    return os

def Rescale(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def IntegerDigits(num):
    x = [int(a) for a in str(num)]
    return x

class ObjPositionOrientationData:
    def __init__(self,Data,Symmetry,Wavelets = None,InputData = None,
                 DcFilterImage = 0):
        self.Data = Data
        self.Symmetry = Symmetry
        self.Wavelets = Wavelets
        self.InputData = InputData
        self.DcFilterImage = DcFilterImage
        self.AngularResolution = Symmetry/Data.shape[0]
        self.FullOrientationList = np.arange(0,Symmetry,self.AngularResolution)

def LeftInvariantDerivative(osObj,sigmaSpatial,sigmaOD,order,symmetry,anglesMatrix):
        n = np.sum(np.array(order) == 1) + np.sum(np.array(order) == 2)+1
        scaledSigmaSpatial = 1/np.sqrt(n)*sigmaSpatial
        angularOrder = np.sum(np.array(order) == 3)
        der = OrientationDerivative(osObj.Data, scaledSigmaSpatial, sigmaOD, osObj.AngularResolution,
                                    symmetry, angularOrder)
        for dirr in order:
            if dirr == 3:
                continue
            if symmetry == np.pi:
                symmetry1 = -np.pi
            elif symmetry == -np.pi:
                symmetry1 = np.pi
            elif symmetry == 2*np.pi:
                symmetry1 = 2*np.pi
            
            der = SpatialDerivative(der, scaledSigmaSpatial, 0, dirr,anglesMatrix, symmetry1)
            
            
        return der

def OrientationScoreTensor3(osObj,sigmaSpatial,sigmaOrientation,method):
    if method == "LIF":
        order_list = [11,22]
    else :
        order_list = [11,21,31,12,22,32,13,23,33]
    sigmaOD = sigmaOrientation / osObj.AngularResolution
    tensor = np.zeros((*osObj.Data.shape,3,3))
    
    angles = np.arange(0,abs(osObj.Symmetry),osObj.AngularResolution)
    anglesMatrix = np.zeros_like(osObj.Data.real)
    for i in range(0,osObj.Data.real.shape[0]):
        anglesMatrix[i,:,:] = angles[i]
        
    symmetry = osObj.Symmetry
    der = []
    for order in order_list:
        if order!= 31 and order!= 32:
            order = IntegerDigits(order)
            der = LeftInvariantDerivative(osObj,sigmaSpatial,sigmaOD,order[::-1],symmetry,anglesMatrix)
            
        elif order== 31:
            order = IntegerDigits(order)
            der2 = LeftInvariantDerivative(osObj,sigmaSpatial,sigmaOD,[2],symmetry,anglesMatrix)
            der13 = LeftInvariantDerivative(osObj,sigmaSpatial,sigmaOD,[3,1],symmetry,anglesMatrix)
            der = der2+der13
        elif order == 32:
            order = IntegerDigits(order)
            der23 = LeftInvariantDerivative(osObj,sigmaSpatial,sigmaOD,[3,2],symmetry,anglesMatrix)
            der1 = LeftInvariantDerivative(osObj,sigmaSpatial,sigmaOD,[1],symmetry,anglesMatrix)
            der = der23-der1
            
        
        tensor[:,:,:,order[1]-1,order[0]-1] = der        
    return tensor

def CreatePeriodicOrientationAxes(os,symmetry):
    if symmetry == np.pi or symmetry == 2*np.pi:
        return os
    elif symmetry == -np.pi:
        return np.concatenate([os,np.conj(os)],axis = 0)

def OrientationDerivative(derivativesIn, scaledSigmaSpatial, sigmaOriantation, angularResolution, symmetry, order):
    periodicOS = CreatePeriodicOrientationAxes(derivativesIn, symmetry)
    sigma = scaledSigmaSpatial
    trunc = (4*scaledSigmaSpatial + 1)/sigma
    spatialBlurredOs = skimage.filters.gaussian(periodicOS,sigma = [0,sigma,sigma],truncate = trunc)
    derivative = scipy.ndimage.gaussian_filter(spatialBlurredOs, [sigmaOriantation,0.125,0.125], order=[order,0,0], mode="wrap")
    if symmetry == np.pi or symmetry == -np.pi:
        derivative = derivative[0:derivativesIn.shape[0],:,:]
    derivative = derivative / (angularResolution**order)
    
    return derivative

def SpatialDerivative(derivativesIn, scaledSigmaSpatial, scaledSigmaOrientation, dirr,angles, symmetry):
    periodicOS = CreatePeriodicOrientationAxes(derivativesIn, symmetry)
    orientationBlurredOS = periodicOS
    trunc = (4*scaledSigmaSpatial + 1)/scaledSigmaSpatial
    if symmetry == np.pi or symmetry == -np.pi:
        orientationBlurredOS = orientationBlurredOS[0:derivativesIn.shape[0],:,:]
    dx = norm_gaussian_filter(orientationBlurredOS, [0,scaledSigmaSpatial,scaledSigmaSpatial],
                                       order=[0,1,0],truncate=trunc, mode="nearest")
    dy = norm_gaussian_filter(orientationBlurredOS, [0,scaledSigmaSpatial,scaledSigmaSpatial],
                                       order=[0,0,1],truncate=trunc, mode="nearest")
    if dirr == 1:
        derivative = dx*np.cos(angles) + dy*np.sin(angles)
    elif dirr == 2:
        derivative = -dx*np.sin(angles) + dy*np.cos(angles)
    return derivative

def norm_gaussian_filter(data,sigma,order,truncate = 4,mode = "wrap"):
#     print(sigma[1])
#     if sigma[1]<0.5:
    zeros = np.zeros_like(data)
    ind = [int(i/2) for i in data.shape]
    zeros[tuple(ind)] = 1
    weights = scipy.ndimage.gaussian_filter(zeros, sigma,
                                        order=order,truncate=truncate, mode=mode)

    r = scipy.ndimage.gaussian_filter(data, sigma,
                                        order=order,truncate=truncate, mode=mode)
    r = r/np.sum(abs(weights))
#     else :
#         r = scipy.ndimage.gaussian_filter(data, sigma,
#                                     order=order,truncate=truncate, mode=mode)
    return r

def CostFunctionVesselnessFiltering(U,ksi,zeta,sigma_s, method,sigmas_ext = 0, sigmaa_ext = 0):
    Nx = U.shape[1]
    Ny = U.shape[2]
    No = U.shape[0]
    betha = 0.75/sigma_s
    sigma1 = 0.5
    obj = ObjPositionOrientationData(U,2*np.pi,Wavelets = None,InputData = None,DcFilterImage = 0)
    H = OrientationScoreTensor3(obj,0.5*sigma_s**2, 0.5*(2*betha*sigma_s)**2,method)

    M = np.diag([1/ksi,zeta/ksi,1])

    if sigmas_ext!=0 or sigmaa_ext !=  0:
        H = ExternalRegularization(H,obj.FullOrientationList,sigmas_ext,sigmaa_ext)
    a = np.ones((3,3))
    b = np.dot(M,np.dot(a,M))
    Hess = np.zeros((No,Nx,Ny,3,3))
    for i in range(No):
        for j in range(Nx):
            for z in range(Ny):
                Hess[i,j,z,:,:] = b

    Hess = Hess*H

    if method == "LIF":
        lambda1 = Hess[:,:,:,0,0]
        c = Hess[:,:,:,1,1]
        Q = c
    else :
        pass

    S = lambda1**2 + c**2
    R = lambda1/c
    sigma2 = 0.2*np.max(abs(S))
    cost = np.exp(-R**2/(2*sigma1**2)) * (1 - np.exp(-S/(2*sigma2)))
    Qgreater0 = 1-np.heaviside(-Q,0)

    cost = cost*Qgreater0
    
    return cost

def CostFunction(oc,lambdaa, p):
    cost = 1/(1 + lambdaa*(oc)**p)
    return cost

def MultiScaleVesselness(U,ksi,zeta,sigmas_s,method,sigmas_ext = 0, sigmaa_ext = 0):
    """Ërosion gives not the same results!!!"""
    vesselnessfilter = []
    for sigma in sigmas_s:
        print(sigma)
        vesselness = CostFunctionVesselnessFiltering(U,ksi,zeta,sigma, method,sigmaa_ext = sigmaa_ext)
        pad = 5
        vesselness_pad = np.pad(vesselness,pad,mode = 'wrap')

        vesselnessErosion = scipy.ndimage.morphology.grey_erosion(vesselness_pad,size=(3,0,0))
        vesselnessErosion =  vesselnessErosion[pad:-pad,pad:-pad,pad:-pad]
        vesselnessErosion[:,:5,:] = 0
        vesselnessfilter.append(vesselnessErosion)
        
    return (vesselnessfilter)

def MultiScaleVesselnessFilter(vesselnessfilters):
    """in nootebook it returns sum1. Why ???"""
    sum1 = np.sum(vesselnessfilters,axis = 0)
    mu = np.max(sum1)
    cost = sum1/mu
    return cost

def LeftInvariantFrame(theta):
    return np.array([[np.cos(theta),np.sin(theta),0],[-np.sin(theta),np.cos(theta),0],[0,0,1]])

def FromLeftInvariantFrame(orientationList,tensor):
    out = np.zeros_like(tensor)
    leftInvariantFrame = []
    for o in range(tensor.shape[0]):
        rot = LeftInvariantFrame(orientationList[o])
        for i in range(tensor.shape[1]):
            for j in range(tensor.shape[2]):
                ten = tensor[o,i,j,:,:]
                out[o,i,j,:,:] = np.dot(rot.T,np.dot(ten,rot))
    return out

def ToLeftInvariantFrame(orientationList,tensor):
    out = np.zeros_like(tensor)
    leftInvariantFrame = []
    for o in range(tensor.shape[0]):
        rot = LeftInvariantFrame(orientationList[o])
        for i in range(tensor.shape[1]):
            for j in range(tensor.shape[2]):
                ten = tensor[o,i,j,:,:]
                out[o,i,j,:,:] = np.dot(rot,np.dot(ten,rot.T))
    return out

def ExternalRegularization(tensor,orientations,sigmaSpatialExternal,sigmaAngularExternal):
    oriantations1 = np.sort(orientations)
    oriantations1 = oriantations1[1:] - oriantations1[:-1]
    sigmaAngularExternal = sigmaAngularExternal/np.mean(oriantations1)

    tensor = FromLeftInvariantFrame(orientations, tensor)
    tensor = norm_gaussian_filter(tensor,sigma = [sigmaAngularExternal,sigmaSpatialExternal,sigmaSpatialExternal,0,0],
                                  order = 0,mode='nearest')
    tensor = ToLeftInvariantFrame(orientations, tensor)
    return tensor