import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

##### image functions ###########
def image_load(name):
    return cv2.imread(name)[:,:,::-1]

def show_image(image,frame_size = 0, pts = None, limits_y=None, limits_x=None,color='green', marker='+',markersize=12,cmap = 'gray'):
    if frame_size!=0:
        fig = plt.figure(figsize = (frame_size, frame_size))
    if pts != None:
        for p in pts:
            plt.plot(p[0],p[1],color=color, marker=marker,markersize=markersize)
        if limits_x!=None:
            plt.xlim([np.min(np.array(pts)[:,0]) - limits_x, np.max(np.array(pts)[:,0]) + limits_x])
        if limits_y!=None:
            plt.ylim([np.max(np.array(pts)[:,1]) + limits_y, np.min(np.array(pts)[:,1]) - limits_y])
            
    plt.imshow(image,cmap = cmap)
    
def draw_track(image,track,frame_size = 0,limits_x = None,limits_y = None, track_color = 'r',track_width = 2):
    if frame_size!=0:
        fig = plt.figure(figsize = (frame_size, frame_size))
    plt.imshow(image)
    plt.plot(track[0],track[1], color=track_color, linewidth=track_width)
    if limits_x!= None:
        plt.xlim([np.min(track[0]) - limits_x, np.max(track[0]) + limits_x])
    if limits_y!= None:
        plt.ylim([np.max(track[1]) + limits_y, np.min(track[1]) - limits_y])
    plt.show()
    
def scale_image(image):
    image_out = image - abs(np.min(image))
    s = 1/image_out.max()
    image_out = image_out*s
    return image_out
###################################

##### choose points on image ##################
pts = []
def put_points(img1):
    # mouse callback function
    def line_drawing(event,x,y,flags,param):
        global pts,pt
        if event==cv2.EVENT_LBUTTONDOWN:
            pts.append(np.array([x,y]))
            cv2.line(img,(x-10,y),(x+10,y),color=(0,255,0),thickness=1)
            cv2.line(img,(x,y-10),(x,y+10),color=(0,255,0),thickness=1)

    img = img1.copy()
    cv2.namedWindow('test draw')
    cv2.setMouseCallback('test draw',line_drawing)

    while(1):
        cv2.imshow('test draw',img)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyAllWindows()
    
    return (np.array(pts))

def points(img1,scalar):
    global pts
    pts = []
    contur_points = put_points(cv2.resize(img1,\
                            (int(img1.shape[1]/scalar),int(img1.shape[0]/scalar))))
    contur_points = scalar*contur_points
    return contur_points

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

def redrow_points(img,pts,t,scale):
    img2 = img.copy()
    for p in pts:
        x1v = int2(p[0]-10)
        x2v = int2(p[0]+10)
        y1v = int2(p[1])
        y2v = int2(p[1])
        
        x1h = int2(p[0])
        x2h = int2(p[0])
        y1h = int2(p[1]-10)
        y2h = int2(p[1]+10)
        
#         img2 = cv2.line(img2,(x1,y1),(x2,y2),color=(0,255,0),thickness=int2(np.ceil(t*scale)))
        img2 = cv2.line(img2,(x1h,y1h),(x2h,y2h),color=(0,255,0),thickness=int2(np.ceil(t*scale)))
        img2 = cv2.line(img2,(x1v,y1v),(x2v,y2v),color=(0,255,0),thickness=int2(np.ceil(t*scale)))
    return (img2)

def redrow_coordinates(img,x,y,t,scale):
    img2 = img.copy()
    img2 = cv2.line(img2,(x,img2.shape[0]),(x,0),color=(0,255,0),thickness=int2(np.ceil(t*scale)))
    img2 = cv2.line(img2,(img2.shape[1],y),(0,y),color=(0,255,0),thickness=int2(np.ceil(t*scale)))
    return (img2)

def redrow_bb(img,x,y,t,scale,pts,active,c):
    img2 = img.copy()
    img2 = cv2.line(img2,(x,img2.shape[0]),(x,0),color=(0,255,0),thickness=int2(np.ceil(t*scale)))
    img2 = cv2.line(img2,(img2.shape[1],y),(0,y),color=(0,255,0),thickness=int2(np.ceil(t*scale)))
    if len(pts)>1:
        for i in range(0,len(pts)-1,2):
            x0 = int(pts[i][0])
            y0 = int(pts[i][1])
            x1 = int(pts[i+1][0])
            y1 = int(pts[i+1][1])
            color = (255,0,0)
            if len(c)>int(i/2):
                if c[int(i/2)] == 1:
                    color = (0,0,255)
                elif c[int(i/2)] == 2:
                    color = (0,255,0)
            img2 = cv2.line(img2,(x0,y0),(x1,y0),color=color,thickness=int2(np.ceil(t*scale)))
            img2 = cv2.line(img2,(x0,y0),(x0,y1),color=color,thickness=int2(np.ceil(t*scale)))
            img2 = cv2.line(img2,(x0,y1),(x1,y1),color=color,thickness=int2(np.ceil(t*scale)))
            img2 = cv2.line(img2,(x1,y0),(x1,y1),color=color,thickness=int2(np.ceil(t*scale)))
        
    if active == True and x!=None: 
        x1 = int(pts[-1][0])
        y1 = int(pts[-1][1])

        img2 = cv2.line(img2,(x,y),(x1,y),color=(255,0,0),thickness=int2(np.ceil(t*scale)))
        img2 = cv2.line(img2,(x,y),(x,y1),color=(255,0,0),thickness=int2(np.ceil(t*scale)))
        img2 = cv2.line(img2,(x,y1),(x1,y1),color=(255,0,0),thickness=int2(np.ceil(t*scale)))
        img2 = cv2.line(img2,(x1,y),(x1,y1),color=(255,0,0),thickness=int2(np.ceil(t*scale)))
    return (img2)

def drow_mask_lines(img,counturs_x,counturs_y,color,t=1):
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
    img2 = cv2.line(img2,(x1,y1),(x2,y2),color=color,thickness=int2(np.ceil(t)))
    return (img2)

def int2(a):
    return (int(np.round(a)))

class Draw():
    def counturs(self,image,scale,move_x = 0, move_y = 0):
        """
        image : array
            Image to drow on
        scale : int,float
            defins size of display window    
        """
        self.image = image
        self.image_countur = self.image.copy()
        
        self.scale = scale
        self.drawing = False
        
        self.t = 1
        self.p = 0.1
        self.pt1_x , self.pt1_y = None , None
        self.counturs_x = []
        self.counturs_y = []
        self.countur_x = []
        self.countur_y = []
        self.image2 = image.copy()
        self.dx = 0
        self.dy = 0
        self.dx1 = 0
        self.dx2 = 1
        self.dy1 = 0
        self.dy2 = 1
        self.scale2 = 1   
        self.scale2x = 1
        self.scale2y = 1
    
        
        cv2.namedWindow('draw counturs')
        cv2.moveWindow('draw counturs', move_x, move_y)
        cv2.setMouseCallback('draw counturs',self.line_drawing)

        self.image_countur = cv2.resize(self.image_countur,[int2(self.image_countur.shape[1]/scale),
                                                            int2(self.image_countur.shape[0]/scale)],
                                        interpolation = cv2.INTER_NEAREST)
        while(1):
            cv2.imshow('draw counturs',self.image_countur)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cv2.destroyAllWindows()

        flat_x = [item for sublist in self.counturs_x for item in sublist]
        flat_y = [item for sublist in self.counturs_y for item in sublist]

        flat_x = np.array(flat_x) - 0.5
        flat_y = np.array(flat_y)- 0.5

        return flat_x,flat_y
        
        
    def line_drawing(self,event,x,y,flags,param):

        if event==cv2.EVENT_LBUTTONDOWN:
            self.drawing=True
            
            self.pt1_x,self.pt1_y=self.dx1+x*self.scale*self.scale2x,self.dy1+y*self.scale*self.scale2y
            self.countur_x = []
            self.countur_y = []
            self.countur_x.append(self.pt1_x)
            self.countur_y.append(self.pt1_y)
            self.image2 = redrow_lines(self.image,self.counturs_x,self.counturs_y,1,np.mean([self.scale2x,self.scale2y]))
            if len(self.counturs_x)>0:
                cv2.line(self.image_countur,(int2((self.counturs_x[-1][-1]-self.dx1)/self.scale/self.scale2x),
                                             int2((self.counturs_y[-1][-1]-self.dy1)/self.scale/self.scale2y)),
                        (int2(x),int2(y)),color=(0,255,0),thickness=self.t)


        elif event==cv2.EVENT_MOUSEMOVE:
            if self.drawing==True:
                cv2.line(self.image_countur,(int2((self.pt1_x-self.dx1)/self.scale/self.scale2x),
                                             int2((self.pt1_y-self.dy1)/self.scale/self.scale2y)),(int2(x),int2(y)),
                        color=(0,255,0),thickness=self.t)
                self.pt1_x,self.pt1_y=self.dx1+x*self.scale*self.scale2x,self.dy1+y*self.scale*self.scale2y
                self.countur_x.append(self.pt1_x)
                self.countur_y.append(self.pt1_y)
        elif event==cv2.EVENT_LBUTTONUP:
            self.drawing=False
            cv2.line(self.image_countur,(int2((self.pt1_x-self.dx1)/self.scale/self.scale2x),
                                         int2((self.pt1_y-self.dy1)/self.scale/self.scale2y)),
                     (int2(x),int2(y)),color=(0,255,0),thickness=self.t)
            self.counturs_x.append(self.countur_x)
            self.counturs_y.append(self.countur_y)
            self.image2 = redrow_lines(self.image,self.counturs_x,self.counturs_y,1,np.mean([self.scale2x,self.scale2y]))

        elif event==cv2.EVENT_RBUTTONDOWN:
            if self.drawing==False and len(self.counturs_x)>0:
                self.counturs_x.remove(self.counturs_x[-1])
                self.counturs_y.remove(self.counturs_y[-1])
                self.image2 = redrow_lines(self.image,self.counturs_x,self.counturs_y,1,1)
                self.image_countur = cv2.resize(self.image2[self.dy1:-self.dy2,self.dx1:-self.dx2,:],
                                                [int2(self.image.shape[1]/self.scale/self.scale2),
                                                 int2(self.image.shape[0]/self.scale/self.scale2)],
                            interpolation = cv2.INTER_NEAREST)

        elif event==cv2.EVENT_MOUSEWHEEL and flags>0:

            rx,ry = x/self.image_countur.shape[1],y/self.image_countur.shape[0]

            ddx = (self.image.shape[1]-(self.dx1+self.dx2))*self.p
            self.dx1 = np.max([int2(self.dx1+ddx*rx),0])
            self.dx2 = np.max([int2(self.dx2+ddx*(1-rx)),1])

            ddy = (self.image.shape[0]-(self.dy1+self.dy2))*self.p
            self.dy1 = np.max([int2(self.dy1+ddy*ry),0])
            self.dy2 = np.max([int2(self.dy2+ddy*(1-ry)),1])

            self.scale2x = 1-(self.dx1+self.dx2)/self.image.shape[1]
            self.scale2y = 1-(self.dy1+self.dy2)/self.image.shape[0]
    #         image2 = redrow_lines(image,counturs_x,counturs_y,t,scale*scale2x)
            self.image_countur = self.image2[self.dy1:-self.dy2,self.dx1:-self.dx2,:]
            self.image_countur = cv2.resize(self.image_countur,[int2(self.image_countur.shape[1]/self.scale/self.scale2x),
                                                    int2(self.image_countur.shape[0]/self.scale/self.scale2y)],
                                                    interpolation = cv2.INTER_NEAREST)
        elif event==cv2.EVENT_MOUSEWHEEL:
            rx,ry = x/self.image_countur.shape[1],y/self.image_countur.shape[0]

            ddx = (self.image.shape[1]-(self.dx1+self.dx2))*self.p
            self.dx1 = np.max([int2(self.dx1-ddx*rx),0])
            self.dx2 = np.max([int2(self.dx2-ddx*(1-rx)),1])

            ddy = (self.image.shape[0]-(self.dy1+self.dy2))*self.p
            self.dy1 = np.max([int2(self.dy1-ddy*ry),0])
            self.dy2 = np.max([int2(self.dy2-ddy*(1-ry)),1])

            self.scale2x = 1-(self.dx1+self.dx2)/self.image.shape[1]
            self.scale2y = 1-(self.dy1+self.dy2)/self.image.shape[0]
    #         image2 = redrow_lines(image,counturs_x,counturs_y,t,scale*scale2x)
            self.image_countur = self.image2[self.dy1:-self.dy2,self.dx1:-self.dx2,:]
            self.image_countur = cv2.resize(self.image_countur,[int2(self.image_countur.shape[1]/self.scale/self.scale2x),
                                                    int2(self.image_countur.shape[0]/self.scale/self.scale2y)],
                                                    interpolation = cv2.INTER_NEAREST)
            
    def points(self,image,scale,t = 5,move_x = 0, move_y = 0):
        """
        image : array
            Image to drow on
        scale : int,float
            defins size of display window    
        """
        self.image = image
        self.image_countur = self.image.copy()
        self.scale = scale
        
        self.t = t
        self.p = 0.1
        self.pt1_x , self.pt1_y = None , None
        self.pts = []
        self.pt_x = []
        self.pt_y = []
        self.image2 = image.copy()
        self.dx = 0
        self.dy = 0
        self.dx1 = 0
        self.dx2 = 1
        self.dy1 = 0
        self.dy2 = 1
        self.scale2 = 1   
        self.scale2x = 1
        self.scale2y = 1
    
        
        cv2.namedWindow('draw points')
        cv2.moveWindow('draw points', move_x, move_y)
        cv2.setMouseCallback('draw points',self.put_points)

        self.image_countur = cv2.resize(self.image_countur,[int2(self.image_countur.shape[1]/scale),
                                                            int2(self.image_countur.shape[0]/scale)],
                                        interpolation = cv2.INTER_NEAREST)
        while(1):
            cv2.imshow('draw points',self.image_countur)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cv2.destroyAllWindows()

#         flat_x = [item for sublist in self.counturs_x for item in sublist]
#         flat_y = [item for sublist in self.counturs_y for item in sublist]

#         flat_x = np.array(flat_x) - 0.5
#         flat_y = np.array(flat_y)- 0.5

        return self.pts
        
        
    def put_points(self,event,x,y,flags,param):

        if event==cv2.EVENT_LBUTTONDOWN:
            
            self.pt1_x,self.pt1_y=self.dx1+x*self.scale*self.scale2x,self.dy1+y*self.scale*self.scale2y
            self.pts.append(np.array([self.pt1_x,self.pt1_y]))
            self.image2 = redrow_points(self.image,self.pts,self.t,np.mean([self.scale2x,self.scale2y]))
            cv2.line(self.image_countur,(int2(x-10),int2(y)),(int2(x+10),int2(y)),color=(0,255,0),thickness=1)
            cv2.line(self.image_countur,(int2(x),int2(y-10)),(int2(x),int2(y+10)),color=(0,255,0),thickness=1)
#             self.image_countur = self.image2.copy()

        elif event==cv2.EVENT_RBUTTONDOWN:
            if len(self.pts)>0:
                
                self.pts = self.pts[:-1]
                self.image2 = redrow_points(self.image,self.pts,self.t,np.mean([self.scale2x,self.scale2y]))
                self.image_countur = cv2.resize(self.image2[self.dy1:-self.dy2,self.dx1:-self.dx2,:],
                                                [int2(self.image.shape[1]/self.scale/self.scale2),
                                                 int2(self.image.shape[0]/self.scale/self.scale2)],
                            interpolation = cv2.INTER_NEAREST)

        elif event==cv2.EVENT_MOUSEWHEEL and flags>0:

            rx,ry = x/self.image_countur.shape[1],y/self.image_countur.shape[0]

            ddx = (self.image.shape[1]-(self.dx1+self.dx2))*self.p
            self.dx1 = np.max([int2(self.dx1+ddx*rx),0])
            self.dx2 = np.max([int2(self.dx2+ddx*(1-rx)),1])

            ddy = (self.image.shape[0]-(self.dy1+self.dy2))*self.p
            self.dy1 = np.max([int2(self.dy1+ddy*ry),0])
            self.dy2 = np.max([int2(self.dy2+ddy*(1-ry)),1])

            self.scale2x = 1-(self.dx1+self.dx2)/self.image.shape[1]
            self.scale2y = 1-(self.dy1+self.dy2)/self.image.shape[0]
    #         image2 = redrow_lines(image,counturs_x,counturs_y,t,scale*scale2x)
            self.image_countur = self.image2[self.dy1:-self.dy2,self.dx1:-self.dx2,:]
            self.image_countur = cv2.resize(self.image_countur,[int2(self.image_countur.shape[1]/self.scale/self.scale2x),
                                                    int2(self.image_countur.shape[0]/self.scale/self.scale2y)],
                                                    interpolation = cv2.INTER_NEAREST)
        elif event==cv2.EVENT_MOUSEWHEEL:
            rx,ry = x/self.image_countur.shape[1],y/self.image_countur.shape[0]

            ddx = (self.image.shape[1]-(self.dx1+self.dx2))*self.p
            self.dx1 = np.max([int2(self.dx1-ddx*rx),0])
            self.dx2 = np.max([int2(self.dx2-ddx*(1-rx)),1])

            ddy = (self.image.shape[0]-(self.dy1+self.dy2))*self.p
            self.dy1 = np.max([int2(self.dy1-ddy*ry),0])
            self.dy2 = np.max([int2(self.dy2-ddy*(1-ry)),1])

            self.scale2x = 1-(self.dx1+self.dx2)/self.image.shape[1]
            self.scale2y = 1-(self.dy1+self.dy2)/self.image.shape[0]
    #         image2 = redrow_lines(image,counturs_x,counturs_y,t,scale*scale2x)
            self.image_countur = self.image2[self.dy1:-self.dy2,self.dx1:-self.dx2,:]
            self.image_countur = cv2.resize(self.image_countur,[int2(self.image_countur.shape[1]/self.scale/self.scale2x),
                                                    int2(self.image_countur.shape[0]/self.scale/self.scale2y)],
                                                    interpolation = cv2.INTER_NEAREST)
            
    def bounding_box(self,image,scale,t = 5, move_x = 0, move_y = 0):
        """
        image : array
            Image to drow on
        scale : int,float
            defins size of display window    
        """
        self.image = image
        self.image_countur = self.image.copy()
        self.scale = scale
        
        self.t = t
        self.p = 0.1
        self.pt1_x , self.pt1_y = None , None
        self.pts = []
        self.c = []
        self.pt_x = []
        self.pt_y = []
        self.image2 = image.copy()
        self.dx = 0
        self.dy = 0
        self.dx1 = 0
        self.dx2 = 1
        self.dy1 = 0
        self.dy2 = 1
        self.scale2 = 1   
        self.scale2x = 1
        self.scale2y = 1
        self.active = False
    
        
        cv2.namedWindow('draw bb')
        cv2.moveWindow('draw bb', move_x, move_y)
        cv2.setMouseCallback('draw bb',self.bb)

        self.image_countur = cv2.resize(self.image_countur,[int2(self.image_countur.shape[1]/scale),
                                                            int2(self.image_countur.shape[0]/scale)],
                                        interpolation = cv2.INTER_NEAREST)
        while(1):
            cv2.imshow('draw bb',self.image_countur)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            if len(self.c)<int(len(self.pts)/2):
                if cv2.waitKey(1) & 0xFF == 49:
                    self.c.append(1)
                if cv2.waitKey(1) & 0xFF == 50:
                    self.c.append(2)
                
        cv2.destroyAllWindows()

#         flat_x = [item for sublist in self.counturs_x for item in sublist]
#         flat_y = [item for sublist in self.counturs_y for item in sublist]

#         flat_x = np.array(flat_x) - 0.5
#         flat_y = np.array(flat_y)- 0.5

        return self.pts,self.c

    def bb(self,event,x,y,flags,param):

        if event==cv2.EVENT_LBUTTONDOWN:
            if self.active == False:
                self.active = True
                self.pt1_x,self.pt1_y=self.dx1+x*self.scale*self.scale2x,self.dy1+y*self.scale*self.scale2y
                self.pts.append(np.array([self.pt1_x,self.pt1_y]))
    #             self.image2 = redrow_points(self.image,self.pts,1,1)
#                 cv2.line(self.image_countur,(int2(x-10),int2(y)),(int2(x+10),int2(y)),color=(0,255,0),thickness=1)
#                 cv2.line(self.image_countur,(int2(x),int2(y-10)),(int2(x),int2(y+10)),color=(0,255,0),thickness=1)
            elif self.active == True:
                self.active = False
                x1,y1=self.dx1+x*self.scale*self.scale2x,self.dy1+y*self.scale*self.scale2y
                self.pt1_x,self.pt1_y=self.dx1+x*self.scale*self.scale2x,self.dy1+y*self.scale*self.scale2y
                self.pts.append(np.array([self.pt1_x,self.pt1_y]))
                self.image2 = redrow_bb(self.image,int(x1),int(y1),self.t,np.mean([self.scale2x,self.scale2y]),
                                        self.pts,self.active,self.c)
#                 self.c.append(input('class (1-crack, 2-corrosion):'))
    #             self.image2 = redrow_points(self.image,self.pts,1,1)
#                 cv2.line(self.image_countur,(int2(x-10),int2(y)),(int2(x+10),int2(y)),color=(0,255,0),thickness=1)
#                 cv2.line(self.image_countur,(int2(x),int2(y-10)),(int2(x),int2(y+10)),color=(0,255,0),thickness=1)
                
#             self.image_countur = self.image2.copy()
        

        elif event==cv2.EVENT_RBUTTONDOWN:
            if len(self.pts)>0:
                if self.active == True:
                    self.pts = self.pts[:-1]
                    self.active = False
                elif self.active == False:
                    self.pts = self.pts[:-2]
                    self.c = self.c[:-1]
                if len(self.pts)>0:
                    self.image2 = redrow_bb(self.image,None,None,self.t,np.mean([self.scale2x,self.scale2y]),
                                        self.pts,self.active,self.c)
                    self.image_countur = cv2.resize(self.image2[self.dy1:-self.dy2,self.dx1:-self.dx2,:],
                                                    [int2(self.image.shape[1]/self.scale/self.scale2),
                                                     int2(self.image.shape[0]/self.scale/self.scale2)],
                                interpolation = cv2.INTER_NEAREST)

                    
        if event==cv2.EVENT_MOUSEMOVE:
            x1,y1=self.dx1+x*self.scale*self.scale2x,self.dy1+y*self.scale*self.scale2y
            self.image2 = redrow_coordinates(self.image,int(x1),int(y1),self.t,np.mean([self.scale2x,self.scale2y]))
            self.image2 = redrow_bb(self.image,int(x1),int(y1),self.t,np.mean([self.scale2x,self.scale2y]),
                                    self.pts,self.active,self.c)
            self.image_countur = cv2.resize(self.image2[self.dy1:-self.dy2,self.dx1:-self.dx2,:],
                                            [int2(self.image.shape[1]/self.scale/self.scale2),
                                             int2(self.image.shape[0]/self.scale/self.scale2)],
                        interpolation = cv2.INTER_NEAREST)
                

        elif event==cv2.EVENT_MOUSEWHEEL and flags>0:

            rx,ry = x/self.image_countur.shape[1],y/self.image_countur.shape[0]

            ddx = (self.image.shape[1]-(self.dx1+self.dx2))*self.p
            self.dx1 = np.max([int2(self.dx1+ddx*rx),0])
            self.dx2 = np.max([int2(self.dx2+ddx*(1-rx)),1])

            ddy = (self.image.shape[0]-(self.dy1+self.dy2))*self.p
            self.dy1 = np.max([int2(self.dy1+ddy*ry),0])
            self.dy2 = np.max([int2(self.dy2+ddy*(1-ry)),1])

            self.scale2x = 1-(self.dx1+self.dx2)/self.image.shape[1]
            self.scale2y = 1-(self.dy1+self.dy2)/self.image.shape[0]
    #         image2 = redrow_lines(image,counturs_x,counturs_y,t,scale*scale2x)
            self.image_countur = self.image2[self.dy1:-self.dy2,self.dx1:-self.dx2,:]
            self.image_countur = cv2.resize(self.image_countur,[int2(self.image_countur.shape[1]/self.scale/self.scale2x),
                                                    int2(self.image_countur.shape[0]/self.scale/self.scale2y)],
                                                    interpolation = cv2.INTER_NEAREST)
            
            
        elif event==cv2.EVENT_MOUSEWHEEL:
            rx,ry = x/self.image_countur.shape[1],y/self.image_countur.shape[0]

            ddx = (self.image.shape[1]-(self.dx1+self.dx2))*self.p
            self.dx1 = np.max([int2(self.dx1-ddx*rx),0])
            self.dx2 = np.max([int2(self.dx2-ddx*(1-rx)),1])

            ddy = (self.image.shape[0]-(self.dy1+self.dy2))*self.p
            self.dy1 = np.max([int2(self.dy1-ddy*ry),0])
            self.dy2 = np.max([int2(self.dy2-ddy*(1-ry)),1])

            self.scale2x = 1-(self.dx1+self.dx2)/self.image.shape[1]
            self.scale2y = 1-(self.dy1+self.dy2)/self.image.shape[0]
    #         image2 = redrow_lines(image,counturs_x,counturs_y,t,scale*scale2x)
            self.image_countur = self.image2[self.dy1:-self.dy2,self.dx1:-self.dx2,:]
            self.image_countur = cv2.resize(self.image_countur,[int2(self.image_countur.shape[1]/self.scale/self.scale2x),
                                                    int2(self.image_countur.shape[0]/self.scale/self.scale2y)],
                                                    interpolation = cv2.INTER_NEAREST)
            
#     def show_image(self,image,scale):
#         """
#         image : array
#             Image to drow on
#         scale : int,float
#             defins size of display window    
#         """
#         self.image = image
#         self.image_countur = self.image.copy()
#         self.scale = scale
    
        
#         cv2.namedWindow('image')

#         self.image_countur = cv2.resize(self.image_countur,[int2(self.image_countur.shape[1]/scale),
#                                                             int2(self.image_countur.shape[0]/scale)],
#                                         interpolation = cv2.INTER_NEAREST)
#         while(1):
#             cv2.imshow('image',self.image_countur)
#             if cv2.waitKey(1) & 0xFF == 27:
#                 break
                
#         cv2.destroyAllWindows()

# #         flat_x = [item for sublist in self.counturs_x for item in sublist]
# #         flat_y = [item for sublist in self.counturs_y for item in sublist]

# #         flat_x = np.array(flat_x) - 0.5
# #         flat_y = np.array(flat_y)- 0.5
            
            
            

def image_crop(image,start_point,end_point,pts,sides1 = 10,sides2 = 10):
    """Function cropps input image with a rectengular 
    box making "sides" pixels indent from endpoints."""

    y_bound1 = int(np.max([int(np.min([start_point[1],end_point[1]]))-sides1,0]))
    y_bound2 = int(np.max([int(np.max([start_point[1],end_point[1]]))+sides1,0]))
    x_bound1 = int(np.max([int(np.min([start_point[0],end_point[0]]))-sides2,0]))
    x_bound2 = int(np.max([int(np.max([start_point[0],end_point[0]]))+sides2,0]))
    img_cropp = image[y_bound1:y_bound2,x_bound1:x_bound2,:]
    pts_cropp = []
    for pt in pts:
        pts_cropp.append(pt-[x_bound1,y_bound1])
    return img_cropp,pts_cropp

def track_crop_to_full(track_crop,start_point,end_point,sides1,sides2):
    y_bound1 = int(np.max([int(np.min([start_point[1],end_point[1]]))-sides1,0]))
    x_bound1 = int(np.max([int(np.min([start_point[0],end_point[0]]))-sides2,0]))
    track_x = np.array(track_crop[0]) + x_bound1
    track_y = np.array(track_crop[1]) + y_bound1
    return [track_x.squeeze(),track_y.squeeze()]

def get_files(folder = 'cracktools/crackimages',formats = ['png','jpg'],basename = True):
    files = []
    for f in formats:
        for file in glob.glob(folder+"/*."+f):
            if basename == True:
                files.append(os.path.basename(file))
            elif basename == False:
                files.append(file)
    return files
