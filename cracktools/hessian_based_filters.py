import skimage
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import frangi, hessian
import sys
import cv2
import scipy
from numpy import arccos, array
from numpy.linalg import norm

def frangi_modified(image, sigmas=range(1, 10, 2), alpha=0.5, beta=0.5, gamma=15,
           black_ridges=True, mode='reflect', cval=0, sigma_mode = 'max'):
    """
    Modified from:
    https://github.com/scikit-image/scikit-image/blob/main/skimage/filters/ridges.py
    added possibility to take sum,max,mean and min of scale to give final result

    Filter an image with the Frangi vesselness filter.

    This filter can be used to detect continuous ridges, e.g. vessels,
    wrinkles, rivers. It can be used to calculate the fraction of the
    whole image containing such objects.

    Defined only for 2-D and 3-D images. Calculates the eigenvectors of the
    Hessian to compute the similarity of an image region to vessels, according
    to the method described in [1]_.

    Parameters
    ----------
    image : (N, M[, P]) ndarray
        Array with input image data.
    sigmas : iterable of floats, optional
        Sigmas used as scales of filter, i.e.,
        np.arange(scale_range[0], scale_range[1], scale_step)
    scale_range : 2-tuple of floats, optional
        The range of sigmas used.
    scale_step : float, optional
        Step size between sigmas.
    alpha : float, optional
        Frangi correction constant that adjusts the filter's
        sensitivity to deviation from a plate-like structure.
    beta : float, optional
        Frangi correction constant that adjusts the filter's
        sensitivity to deviation from a blob-like structure.
    gamma : float, optional
        Frangi correction constant that adjusts the filter's
        sensitivity to areas of high variance/texture/structure.
    black_ridges : boolean, optional
        When True (the default), the filter detects black ridges; when
        False, it detects white ridges.
    mode : {'constant', 'reflect', 'wrap', 'nearest', 'mirror'}, optional
        How to handle values outside the image borders.
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.
    sigma_mode : {'min','max','mean','sum'}, optional
        How to combine data from different sigma scale to obtain final result

    Returns
    -------
    out : (N, M[, P]) ndarray
        Filtered image (maximum of pixels across all scales).

    Notes
    -----
    Written by Marc Schrijver, November 2001
    Re-Written by D. J. Kroon, University of Twente, May 2009, [2]_
    Adoption of 3D version from D. G. Ellis, Januar 20017, [3]_

    See also
    --------
    meijering
    sato
    hessian

    References
    ----------
    .. [1] Frangi, A. F., Niessen, W. J., Vincken, K. L., & Viergever, M. A.
        (1998,). Multiscale vessel enhancement filtering. In International
        Conference on Medical Image Computing and Computer-Assisted
        Intervention (pp. 130-137). Springer Berlin Heidelberg.
        :DOI:`10.1007/BFb0056195`
    .. [2] Kroon, D. J.: Hessian based Frangi vesselness filter.
    .. [3] Ellis, D. G.: https://github.com/ellisdg/frangi3d/tree/master/frangi
    """


    # Check image dimensions
    skimage._shared.utils.check_nD(image, [2, 3])

    # Check (sigma) scales
    sigmas = skimage.filters.ridges._check_sigmas(sigmas)

    # Rescale filter parameters
    alpha_sq = 2 * alpha ** 2
    beta_sq = 2 * beta ** 2
    gamma_sq = 2 * gamma ** 2

    # Get image dimensions
    ndim = image.ndim

    # Invert image to detect dark ridges on light background
    if black_ridges:
        image = skimage.util.invert(image)

    # Generate empty (n+1)D arrays for storing auxiliary images filtered
    # at different (sigma) scales
    filtered_array = np.zeros(sigmas.shape + image.shape)
    lambdas_array = np.zeros_like(filtered_array)

    # Filtering for all (sigma) scales
    for i, sigma in enumerate(sigmas):

        # Calculate (abs sorted) eigenvalues
        lambda1, *lambdas = skimage.filters.ridges.compute_hessian_eigenvalues(image, sigma,
                                                        sorting='abs',
                                                        mode=mode, cval=cval)
        
        # Compute sensitivity to deviation from a plate-like
        # structure see equations (11) and (15) in reference [1]_
        r_a = np.inf

        # Compute sensitivity to deviation from a blob-like structure,
        # see equations (10) and (15) in reference [1]_,
        # np.abs(lambda2) in 2D, np.sqrt(np.abs(lambda2 * lambda3)) in 3D
        filtered_raw = np.abs(np.multiply.reduce(lambdas)) ** (1/len(lambdas))
        r_b = skimage.filters.ridges._divide_nonzero(lambda1, filtered_raw) ** 2

        # Compute sensitivity to areas of high variance/texture/structure,
        # see equation (12)in reference [1]_
        r_g = sum([lambda1 ** 2] + [lambdai ** 2 for lambdai in lambdas])

        # Compute output image for given (sigma) scale and store results in
        # (n+1)D matrices, see equations (13) and (15) in reference [1]_
        filtered_array[i] = ((1 - np.exp(-r_a / alpha_sq))
                             * np.exp(-r_b / beta_sq)
                             * (1 - np.exp(-r_g / gamma_sq)))

        lambdas_array[i] = np.max(lambdas, axis=0)

    # Remove background
    filtered_array[lambdas_array > 0] = 0

    # Return for every pixel the fused value over all (sigma) scales
    if sigma_mode == 'max':
        return np.max(filtered_array, axis=0)
    # Return for every pixel the mean value over all (sigma) scales
    if sigma_mode == 'mean':
        return np.mean(filtered_array, axis=0)
    if sigma_mode == 'sum':
        return np.sum(filtered_array, axis=0)
    if sigma_mode == 'min':
        return np.min(filtered_array, axis=0)
    
def my_frangi_visualize(image, sigmas=range(1, 10, 2), alpha=0.5, beta=0.5, gamma=15,
           black_ridges=True, mode='reflect', cval=0,sigma_mode = 'max',comput_skip = 1,arrow_skip = 30,
                                         arrow_mult = 10, arrow_thickness = 1, ellipse_skip = 30,
                                         ellipse_mult = 10, ellipse_thickness = 1, ellipse_division = 2):
    """
    Filter an image with the Frangi vesselness filter.

    This filter can be used to detect continuous ridges, e.g. vessels,
    wrinkles, rivers. It can be used to calculate the fraction of the
    whole image containing such objects.

    Defined only for 2-D and 3-D images. Calculates the eigenvectors of the
    Hessian to compute the similarity of an image region to vessels, according
    to the method described in [1]_.

    Parameters
    ----------
    image : (N, M[, P]) ndarray
        Array with input image data.
    sigmas : iterable of floats, optional
        Sigmas used as scales of filter, i.e.,
        np.arange(scale_range[0], scale_range[1], scale_step)
    scale_range : 2-tuple of floats, optional
        The range of sigmas used.
    scale_step : float, optional
        Step size between sigmas.
    alpha : float, optional
        Frangi correction constant that adjusts the filter's
        sensitivity to deviation from a plate-like structure.
    beta : float, optional
        Frangi correction constant that adjusts the filter's
        sensitivity to deviation from a blob-like structure.
    gamma : float, optional
        Frangi correction constant that adjusts the filter's
        sensitivity to areas of high variance/texture/structure.
    black_ridges : boolean, optional
        When True (the default), the filter detects black ridges; when
        False, it detects white ridges.
    mode : {'constant', 'reflect', 'wrap', 'nearest', 'mirror'}, optional
        How to handle values outside the image borders.
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.

    Returns
    -------
    out : (N, M[, P]) ndarray
        Filtered image (maximum of pixels across all scales).

    Notes
    -----
    Written by Marc Schrijver, November 2001
    Re-Written by D. J. Kroon, University of Twente, May 2009, [2]_
    Adoption of 3D version from D. G. Ellis, Januar 20017, [3]_

    See also
    --------
    meijering
    sato
    hessian

    References
    ----------
    .. [1] Frangi, A. F., Niessen, W. J., Vincken, K. L., & Viergever, M. A.
        (1998,). Multiscale vessel enhancement filtering. In International
        Conference on Medical Image Computing and Computer-Assisted
        Intervention (pp. 130-137). Springer Berlin Heidelberg.
        :DOI:`10.1007/BFb0056195`
    .. [2] Kroon, D. J.: Hessian based Frangi vesselness filter.
    .. [3] Ellis, D. G.: https://github.com/ellisdg/frangi3d/tree/master/frangi
    """


    # Check image dimensions
    skimage._shared.utils.check_nD(image, [2, 3])

    # Check (sigma) scales
    sigmas = skimage.filters.ridges._check_sigmas(sigmas)

    # Rescale filter parameters
    alpha_sq = 2 * alpha ** 2
    beta_sq = 2 * beta ** 2
    gamma_sq = 2 * gamma ** 2

    # Get image dimensions
    ndim = image.ndim

    # Invert image to detect dark ridges on light background
    if black_ridges:
        image = skimage.util.invert(image)

    # Generate empty (n+1)D arrays for storing auxiliary images filtered
    # at different (sigma) scales
    filtered_array = np.zeros(sigmas.shape + image.shape)
    lambdas_array = np.zeros_like(filtered_array)

    # Filtering for all (sigma) scales
    for i, sigma in enumerate(sigmas):

        # Calculate (abs sorted) eigenvalues
        lambda1, *lambdas = compute_hessian_eigenvalues_visualize(image, sigma,
                                                        sorting='abs',
                                                        mode=mode, cval=cval,comput_skip = comput_skip,arrow_skip = arrow_skip,
                                         arrow_mult = arrow_mult, arrow_thickness = arrow_thickness, ellipse_skip = ellipse_skip,
                                         ellipse_mult = ellipse_mult, ellipse_thickness = ellipse_thickness,
                                                                  ellipse_division = ellipse_division)
        # Compute sensitivity to deviation from a plate-like
        # structure see equations (11) and (15) in reference [1]_
        r_a = np.inf if ndim == 2 else _divide_nonzero(*lambdas) ** 2

        # Compute sensitivity to deviation from a blob-like structure,
        # see equations (10) and (15) in reference [1]_,
        # np.abs(lambda2) in 2D, np.sqrt(np.abs(lambda2 * lambda3)) in 3D
        filtered_raw = np.abs(np.multiply.reduce(lambdas)) ** (1/len(lambdas))
        r_b = skimage.filters.ridges._divide_nonzero(lambda1, filtered_raw) ** 2

        # Compute sensitivity to areas of high variance/texture/structure,
        # see equation (12)in reference [1]_
        r_g = sum([lambda1 ** 2] + [lambdai ** 2 for lambdai in lambdas])

        # Compute output image for given (sigma) scale and store results in
        # (n+1)D matrices, see equations (13) and (15) in reference [1]_
        filtered_array[i] = ((1 - np.exp(-r_a / alpha_sq))
                             * np.exp(-r_b / beta_sq)
                             * (1 - np.exp(-r_g / gamma_sq)))

        lambdas_array[i] = np.max(lambdas, axis=0)

    # Remove background
    filtered_array[lambdas_array > 0] = 0

    # Return for every pixel the maximum value over all (sigma) scales
    if sigma_mode == 'max':
        r = np.max(filtered_array, axis=0)
    # Return for every pixel the mean value over all (sigma) scales
    if sigma_mode == 'mean':
        r = np.mean(filtered_array, axis=0)
    if sigma_mode == 'sum':
        r = np.sum(filtered_array, axis=0)
        
    fig = plt.figure()
    fig.set_size_inches(18.5, 10.5)
    plt.imshow(r,cmap=plt.cm.gray)
    plt.show()
    return r

def compute_hessian_eigenvalues_visualize(image, sigma, sorting='none',
                                mode='constant', cval=0,comput_skip = 1,arrow_skip = 30,
                                         arrow_mult = 10, arrow_thickness = 1, ellipse_skip = 30,
                                         ellipse_mult = 10, ellipse_thickness = 1, ellipse_division = 2):
    
    # rescales integer images to [-1, 1]
    image = skimage.util.dtype.img_as_float(image)
    # make sure float16 gets promoted to float32
#     image = image.astype(float_dtype, copy=False)

    # Make nD hessian
    hessian_elements = skimage.feature.corner.hessian_matrix(image, sigma=sigma, order='rc',
                                      mode=mode, cval=cval)
    
    # Correct for scale
    hessian_elements = [(sigma ** 2) * e for e in hessian_elements]
    
    print("Hessian element df/dx/dx")
    fig = plt.figure()
    fig.set_size_inches(18.5, 10.5)
    plt.imshow(hessian_elements[0],cmap=plt.cm.gray)
    plt.show()
    
    print("Hessian element df/dy/dx")
    fig = plt.figure()
    fig.set_size_inches(18.5, 10.5)
    plt.imshow(hessian_elements[1],cmap=plt.cm.gray)
    plt.show()
    
    print("Hessian element df/dy/dy")
    fig = plt.figure()
    fig.set_size_inches(18.5, 10.5)
    plt.imshow(hessian_elements[2],cmap=plt.cm.gray)
    plt.show()
    

    
    # Compute Hessian eigenvectors
    eig_vectors,hessian_eigenvalues = compute_eig_vectors(hessian_elements,skip = comput_skip)
    
    # Compute Hessian eigenvalues
#     global hessian_eigenvalues
#     hessian_eigenvalues = skimage.feature.corner.hessian_matrix_eigvals(hessian_elements)
#     plt.imshow(hessian_eigenvalues[0] - eig_vals2[0])*1000
    if sorting == 'abs':

        # Sort eigenvalues by absolute values in ascending order
        hessian_eigenvalues_sorted = skimage.filters.ridges._sortbyabs(hessian_eigenvalues, axis=0)

    elif sorting == 'val':

        # Sort eigenvalues by values in ascending order
        hessian_eigenvalues_sorted = np.sort(hessian_eigenvalues, axis=0)
        
        

#     # Return Hessian eigenvalues
#     print('min lambda (scaled)')
#     fig = plt.figure()
#     fig.set_size_inches(18.5, 10.5)
#     plt.imshow(hessian_eigenvalues_sorted[0],cmap=plt.cm.gray)
#     plt.show()
    
#     print('max lambda (scaled)')
#     fig = plt.figure()
#     fig.set_size_inches(18.5, 10.5)
#     plt.imshow(hessian_eigenvalues_sorted[1],cmap=plt.cm.gray)
#     plt.show()
    

    changed_pos = np.zeros((hessian_eigenvalues_sorted[0].shape))
    changed_pos[hessian_eigenvalues_sorted[0] != hessian_eigenvalues[0]] = 1
    
    for i in range(changed_pos.shape[0]):
        for j in range(changed_pos.shape[1]):
            if changed_pos[i,j] == 1:
                eig_vectors[i,j,:,:] = eig_vectors[i,j,::-1,:]
    im = drow_arrows(image,eig_vectors,hessian_eigenvalues_sorted,skip = arrow_skip, mult = arrow_mult,
                    arrow_thickness = arrow_thickness)
    fig = plt.figure()
    fig.set_size_inches(18.5, 10.5)
    plt.imshow(im,cmap=plt.cm.gray)
    plt.show()
    cv2.imwrite('arrows.jpg',im[:,:,::-1]*255)  
    im = drow_ellipse(image,eig_vectors,hessian_eigenvalues_sorted,skip = ellipse_skip,mult = ellipse_mult,
                      division = ellipse_division,ellipse_thickness = ellipse_thickness)
    fig = plt.figure()
    fig.set_size_inches(18.5, 10.5)
    plt.imshow(im,cmap=plt.cm.gray)
    plt.show()
    cv2.imwrite('ellipse.jpg',im[:,:,::-1]*255)  
    return hessian_eigenvalues_sorted

def compute_eig_vectors(hessian_elements,skip = 30):
    M00 = hessian_elements[0]
    M01 = hessian_elements[1]
    M11 = hessian_elements[2]
    eig_vectors = np.zeros((M00.shape[0],M00.shape[1],2,2))
    eig_vals = np.zeros((2,M00.shape[0],M00.shape[1]))
    for i in range(0,M00.shape[0],skip):
        for j in range(0,M00.shape[1],skip):
            e = np.linalg.eig(np.array([[M00[i,j],M01[i,j]],[M01[i,j],M11[i,j]]]))
            eig_vectors[i,j,:,:] = e[1]
            eig_vals[:,i,j] = e[0]
    return (eig_vectors,eig_vals)


def hess_matrix22_eigvals(M00, M01, M11):
    """eigenvalues of 2x2 matrix"""
    l1 = (M00+M11) / 2 - np.sqrt((M00+M11)**2-4*(M00*M11 - M01**2)) / 2
    l2 = (M00+M11) / 2 + np.sqrt((M00+M11)**2-4*(M00*M11 - M01**2)) / 2
    return l1, l2

def hess_matrix22_eigvectors(M00, M01, M11, l1, l2):
    """eigenvalues of 2x2 matrix"""
    x1 = np.ones_like(M00)
    y1 = M01/(M11-l1+sys.float_info.epsilon)
    x2 = np.ones_like(M00)
    y2 = M01/(M11-l2+sys.float_info.epsilon)
    
    x1n = x1/np.sqrt(x1**2+y1**2)
    y1n = y1/np.sqrt(x1**2+y1**2)
    x2n = x2/np.sqrt(x2**2+y2**2)
    y2n = y2/np.sqrt(x2**2+y2**2)
    
    v1 = np.concatenate((np.expand_dims(x1n,axis=2),np.expand_dims(y1n,axis=2)),axis=2)
    v2 = np.concatenate((np.expand_dims(x2n,axis=2),np.expand_dims(y2n,axis=2)),axis=2)
    return v1,v2

def compute_eig_vectors_values(hessian_elements):
    M00 = hessian_elements[0]
    M01 = hessian_elements[1]
    M11 = hessian_elements[2]
    l1,l2 = hess_matrix22_eigvals(M00, M01, M11)
    eig_vals = np.concatenate((np.expand_dims(l1,axis=2),np.expand_dims(l2,axis=2)),axis=2)
    eig_vals = skimage.filters.ridges._sortbyabs(eig_vals, axis=2)
    v1,v2 = hess_matrix22_eigvectors(M00, M01, M11, eig_vals[:,:,0], eig_vals[:,:,1])
#     eig_vals = np.concatenate((np.expand_dims(l1,axis=2),np.expand_dims(l2,axis=2)),axis=2)
    eig_vecs = np.concatenate((np.expand_dims(v1,axis=2),np.expand_dims(v2,axis=2)),axis=2)
    return (eig_vecs,eig_vals)

def sortbyabs(eig_vals, axis_vals):
    """
    Sort array along a given axis by absolute values.
    Parameters
    ----------
    array : (N, ..., M) ndarray
        Array with input image data.
    axis : int
        Axis along which to sort.
    Returns
    -------
    array : (N, ..., M) ndarray
        Array sorted along a given axis by absolute values.
    Notes
    -----
    Modified from: http://stackoverflow.com/a/11253931/4067734
    """

    index_vals = list(np.ix_(*[np.arange(i) for i in eig_vals.shape]))
    index_vals[axis_vals] = np.abs(eig_vals).argsort(axis_vals)

    return eig_vals[tuple(index_vals)]

def inter_eig_vectors_valuest(eig_vectors,eig_values,point_x,point_y):
    x1 = np.floor(point_x).astype(int)
    x2 = np.ceil(point_x).astype(int)
    y1 = np.floor(point_y).astype(int)
    y2 = np.ceil(point_y).astype(int)
    
    z11_vec1_x = eig_vectors[y1,x1,1,0]
    z11_vec1_y = eig_vectors[y1,x1,1,1]
    
    z12_vec1_x = eig_vectors[y1,x2,1,0]
    z12_vec1_y = eig_vectors[y1,x2,1,1]
    
    z21_vec1_x = eig_vectors[y2,x1,1,0]
    z21_vec1_y = eig_vectors[y2,x1,1,1]
    
    z22_vec1_x = eig_vectors[y2,x2,1,0]
    z22_vec1_y = eig_vectors[y2,x2,1,1]
    
#     z11 = eig_values[y1,x1,1]
#     z12 = eig_values[y1,x2,1]
#     z21 = eig_values[y2,x1,1]
#     z22 = eig_values[y2,x2,1]
    
    z11 = abs(eig_values[y1,x1,1])
    z12 = abs(eig_values[y1,x2,1])
    z21 = abs(eig_values[y2,x1,1])
    z22 = abs(eig_values[y2,x2,1])
    
#     if type(point_x)==list or type(point_x) == np.ndarray:
    val = []
    vec1_x = []
    vec1_y = []
    for i in range(len(point_x)):
        f_vec1_x = scipy.interpolate.interp2d([y1[i],y2[i]], [x1[i],x2[i]],
                                            [[z11_vec1_x[i],z12_vec1_x[i]],
                                             [z21_vec1_x[i],z22_vec1_x[i]]], kind='linear')
        f_vec1_y = scipy.interpolate.interp2d([y1[i],y2[i]], [x1[i],x2[i]],
                                            [[z11_vec1_y[i],z12_vec1_y[i]],
                                             [z21_vec1_y[i],z22_vec1_y[i]]], kind='linear')
        f_val = scipy.interpolate.interp2d([y1[i],y2[i]], [x1[i],x2[i]],
                                            [[z11[i],z12[i]],
                                             [z21[i],z22[i]]], kind='linear')

        vec1_x.append(f_vec1_x(point_y[i],point_x[i]))
        vec1_y.append(f_vec1_y(point_y[i],point_x[i]))
        val.append(f_val(point_y[i],point_x[i]))
#     else :
#         f = scipy.interpolate.interp2d([y1,y2], [x1,x2], [[z11,z12],[z21,z22]], kind='linear')
#         val = f(point_y,point_x)
        
    vec = np.concatenate((np.expand_dims(vec1_x,axis=2),np.expand_dims(vec1_y,axis=2)),axis=2)
    return vec,val

def inter_eig_vectors_values_fast(eig_vectors,eig_values,point_x,point_y):
    x = np.round(point_x).astype(int)
    y = np.round(point_y).astype(int)
    
    val = eig_values[y,x,:]
    vec = eig_vectors[y,x,0,:]
    
    return np.expand_dims(vec,axis = 1),val

def angle_between_vectors(v1,v2):
    theta = np.arccos(v1[0]*v2[:,:,0] + v1[1]*v2[:,:,1])
    theta[theta>np.pi/2] = np.pi/2 - theta[theta>np.pi/2] 
    return theta

def drow_arrows(image,eig_vectors,hessian_eigenvalues,skip=30,mult = 10,div = 10, arrow_thickness = 1,frame_size = 0):
    im = image.copy()
    im = abs(np.min(im))+im
    im = 255/np.max(im)*im/255
    if len(image.shape) == 2:
        im = cv2.merge([im,im,im])
    for i in range(0,im.shape[0],skip):
        for j in range(0,im.shape[1],skip):
            val1 = hessian_eigenvalues[0,i,j]
            val2 = hessian_eigenvalues[1,i,j]
            eig1 = eig_vectors[i,j,0,:]
            eig2 = eig_vectors[i,j,1,:]
            delta = abs(val2)*mult-abs(val1)*div
#             delta = mult
            pt1 = (j,i)
#             pt2 = (int(j+eig1[1]*mult1),int(i+eig1[0]*mult1))
            pt3 = (int(j+eig1[0]*delta),int(i+eig1[1]*delta))
#             im = cv2.arrowedLine(im, pt1, pt2,color = (255,0,0), thickness = arrow_thickness) 
            im = cv2.arrowedLine(im, pt1, pt3,color = (0,255,0), thickness = arrow_thickness) 
    if frame_size!=0:
        fig = plt.figure(figsize = (frame_size, frame_size))   
    plt.imshow(im)
    plt.show()
    return im

def drow_ellipse(image,eig_vectors,hessian_eigenvalues,skip=30,mult = 10,division = 2,ellipse_thickness = 1):
    im = image.copy()
    im = np.min(im)+im
    im = 255/np.max(im)*im/255
    im = cv2.merge([im,im,im])
    for i in range(0,im.shape[0],skip):
        for j in range(0,im.shape[1],skip):
            mult1 = hessian_eigenvalues[0][i,j]*mult
            mult2 = hessian_eigenvalues[1][i,j]*mult
            eig1 = eig_vectors[i,j,0,:]
            eig2 = eig_vectors[i,j,1,:]
            
            
            pt1 = (j,i)
            pt2 = (int(j+eig1[1]*mult),int(i+eig1[0]*mult))
            pt3 = (int(j+eig2[1]*mult),int(i+eig2[0]*mult))
        
            pt4 = (int(j + eig1[1] * mult1 + eig2[1] * mult2),int(i + eig1[0] * mult1 + eig2[0] * mult2))
            horizont = np.array([0,1])
            angle = arccos(eig2.dot(horizont)/(norm(eig2)*norm(horizont)))*57.3
            det = np.linalg.det(np.array([eig2,horizont]))
            if det<0:
                angle = 360-angle
            delta = abs(mult2) - abs(mult1)
            delta = delta/division
            l1 = int(abs(mult2))
            l2 = int(abs(mult2)-delta)
            axesLength = (l1, l2)
        
            im = cv2.ellipse(im, pt1, (axesLength),angle, 0, 360, (255, 0, 0), ellipse_thickness)
    return im 

def plot_sigma(image,thick_point,sigma,d = 20,mode='horisontal'):
    x = np.linspace(-20,20,100)
    y = 1/np.sqrt(2*np.pi)/sigma**5*(-sigma**2 + x**2)*np.exp(-x**2/(2*sigma**2))
    x2 = x+d
    y2 = y*d/(np.max(y)-np.min(y))
    y2 = y2+d
    plt.imshow(image[thick_point[1]-d:thick_point[1]+d,thick_point[0]-d:thick_point[0]+d],cmap='gray')
    if mode == 'horisontal':
        plt.plot(x2,y2)
    elif mode == 'vertical':
        plt.plot(y2,x2)
    plt.grid()
    plt.show()
