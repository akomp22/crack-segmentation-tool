import scipy
import numpy as np
import cracktools.tracking
import cv2
from skimage import measure

from agd import Eikonal
from agd.Metrics import Riemann
from agd.Plotting import savefig, quiver; #savefig.dirName = 'Figures/Riemannian'
from agd import LinearParallel as lp
from agd import AutomaticDifferentiation as ad
norm_infinity = ad.Optimization.norm_infinity

def edge_masks(image_gray,track,window_half_size= 40):

    edge1 = []
    edge2 = []
    step = 2
    n = 265
    center_line_length = 3
    edge_mask = np.zeros_like((image_gray),dtype = float)
    for i in range(track.shape[1]-1):
    # for i in range(n,n+1):
        start_point_x = track[1,i]
        start_point_y = track[0,i]
        a= False
        if i<track.shape[1]-center_line_length:
            end_point_x = track[1,i+center_line_length]
            end_point_y = track[0,i+center_line_length]
        else:
            a = True
            end_point_x = track[1,i-center_line_length]
            end_point_y = track[0,i-center_line_length]
        if start_point_x==end_point_x and start_point_y==end_point_y:
            continue

        ddx,ddy,l = cracktools.tracking.tang_len(start_point_x,start_point_y,end_point_x,end_point_y)
        if a == True:
            ddx = -ddx
            ddy = -ddy
        window = np.zeros((window_half_size*2,window_half_size*2))
        window = image_gray[int(start_point_x-window_half_size):int(start_point_x+window_half_size),
                                  int(start_point_y-window_half_size):int(start_point_y+window_half_size)]

        angle = np.arctan2(ddx,ddy)*57.3

        window_rotate = scipy.ndimage.rotate(window,angle,reshape=False)

        sobel2 = scipy.ndimage.sobel(window_rotate/255,axis=0)
        sobel = scipy.ndimage.gaussian_filter(window_rotate/255, 1, order=(1,0), output=None, mode='reflect', cval=0.0, truncate=4.0)
        sobel_rotate = scipy.ndimage.rotate(sobel,-angle,reshape=False)
    #     plt.imshow(window)
    #     plt.show()
    #     plt.imshow(sobel2)
    #     plt.show()
#         m = int(window_half_size)/5
        m = np.max([1,int(window_half_size/5)])
        sobel_rotate[:m,:] = 0
        sobel_rotate[-m:,:] = 0
        sobel_rotate[:,:m] = 0
        sobel_rotate[:,-m:] = 0
        
        edge_window = edge_mask[int(start_point_x-window_half_size):int(start_point_x+window_half_size),
                                  int(start_point_y-window_half_size):int(start_point_y+window_half_size)]

        edge_mask[int(start_point_x-window_half_size):
                  int(start_point_x+window_half_size),
                  int(start_point_y-window_half_size):
                  int(start_point_y+window_half_size)] = edge_window + sobel_rotate
        
    edge_mask1 = edge_mask - np.min(edge_mask)
    edge_mask2 = edge_mask1*-1-np.min(edge_mask1*-1)
    
    return edge_mask1,edge_mask2

import scipy
from agd import Eikonal
from agd.Metrics import Riemann
from agd.Plotting import savefig, quiver; #savefig.dirName = 'Figures/Riemannian'
from agd import LinearParallel as lp
from agd import AutomaticDifferentiation as ad
norm_infinity = ad.Optimization.norm_infinity

def edges_tracking(image_crop, pts_cropp, edge_mask1_cropp, edge_mask2_cropp,mu = 5,l = 1, p = 12):
    
    seeds = np.array([*pts_cropp[0][::-1]])
    tips = np.array([*pts_cropp[1][::-1]])
    b = np.array([0,image_crop.shape[0]])
    c = np.array([0,image_crop.shape[1]])
    sides = np.array([b,c])
    dims = np.array([image_crop.shape[0],image_crop.shape[1]])
    


    DxZ,DyZ = np.gradient(image_crop) 

    a11 = scipy.ndimage.gaussian_filter(mu*DxZ**2, 1, order=(0,0))
    a12 = scipy.ndimage.gaussian_filter(mu*DxZ*DyZ, 1, order=(0,0))
    a21 = scipy.ndimage.gaussian_filter(mu*DxZ*DyZ, 1, order=(0,0))
    a22 = scipy.ndimage.gaussian_filter(mu*DyZ**2, 1, order=(0,0))
    df = np.array([[1+a11,a12],[a21,1+a22]])
    metric1 = (1+edge_mask1_cropp.squeeze()*l)**p*df
    metric2 = (1+edge_mask2_cropp.squeeze()*l)**p*df
    
    
    metric = Riemann(metric1)
    hfmIn = Eikonal.dictIn({
        'model' : 'Riemann2',
        'seeds' : np.expand_dims(seeds,axis = 0),
        'arrayOrdering' : 'RowMajor',
        'tips' : np.expand_dims(tips,axis = 0),
        'metric' : metric})
    hfmIn.SetRect(sides = sides, dims = dims)
    hfmOut = hfmIn.Run()
    geos1 = [g.T for g in hfmOut['geodesics']]
    # geos1[0][:,0] = geos1[0][:,0]+y1
    # geos1[0][:,1] = geos1[0][:,1]+x1
    # track_e1 = ct.tools.track_crop_to_full(geos1[0].T,pts[0],pts[1],y_margin,x_margin)
    track_e1 = geos1[0]
    
    metric = Riemann(metric2)
    hfmIn = Eikonal.dictIn({
        'model' : 'Riemann2',
        'seeds' : np.expand_dims(seeds,axis = 0),
        'arrayOrdering' : 'RowMajor',
        'tips' : np.expand_dims(tips,axis = 0),
        'metric' : metric})
    hfmIn.SetRect(sides = sides, dims = dims)
    hfmOut = hfmIn.Run()
    geos2 = [g.T for g in hfmOut['geodesics']]
    # geos1[0][:,0] = geos1[0][:,0]+y1
    # geos1[0][:,1] = geos1[0][:,1]+x1
    # track_e2 = ct.tools.track_crop_to_full(geos2[0].T,pts[0],pts[1],y_margin,x_margin)
    track_e2 = geos2[0]
    
    return [track_e1[:,0],track_e1[:,1]], [track_e2[:,0],track_e2[:,1]]

def create_mask(image,x,y):
    flat_x = np.array(x)
    flat_y = np.array(y)

    zeros = np.ones_like(image)*255
    mask_contour = drow_mask_lines(zeros,flat_x,flat_y,color = (0,0,0),t=1,close_contur=True)
    labels = measure.label(mask_contour[:,:,0],connectivity=1)
    labels[labels!=1] = 0

    
    kernel = np.array([[0,1,0],
                      [1,1,1],
                      [0,1,0]],dtype = np.uint8)
    mask = np.array((labels),dtype = np.uint8)

    mask = np.array(mask,dtype = float)
    mask = mask*-1+1

    # mask[mask_contour[:,:,0] == 0] = 0
    mask = cv2.erode(mask, kernel, iterations=1)
    return mask

def redrow_lines(img,counturs_x,counturs_y,t,scale):
    flat_x = [item for sublist in counturs_x for item in sublist]
    flat_y = [item for sublist in counturs_y for item in sublist]
    img2 = img.copy()
    for i in range(len(flat_x)-1):
        x1 = int2(flat_x[i]-0.5)
        x2 = int2(flat_x[i+1]-0.5)
        y1 = int2(flat_y[i]-0.5)
        y2 = int2(flat_y[i+1]-0.5)
        img2 = cv2.line(img2,(x1,y1),(x2,y2),color=(0,255,0),thickness=int2(np.ceil(t*scale)))
    return (img2)

def drow_mask_lines(img,counturs_x,counturs_y,color,t=1,close_contur = False):
#     flat_x = [item for sublist in counturs_x for item in sublist]
#     flat_y = [item for sublist in counturs_y for item in sublist]
    img2 = img.copy()
    for i in range(len(counturs_x)-1):
        x1 = int2(np.round(counturs_x[i]))
        x2 = int2(np.round(counturs_x[i+1]))
        y1 = int2(np.round(counturs_y[i]))
        y2 = int2(np.round(counturs_y[i+1]))
        img2 = cv2.line(img2,(x1,y1),(x2,y2),color=color,thickness=int2(np.ceil(t)))
        
    x1 = int2(np.round(counturs_x[0]))
    x2 = int2(np.round(counturs_x[-1]))
    y1 = int2(np.round(counturs_y[0]))
    y2 = int2(np.round(counturs_y[-1]))
    if close_contur == True:
        img2 = cv2.line(img2,(x1,y1),(x2,y2),color=color,thickness=int2(np.ceil(t)))
    return (img2)

def int2(a):
    return (int(np.round(a)))