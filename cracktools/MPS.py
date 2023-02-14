import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
# import itertools
from copy import deepcopy
from agd.Metrics import Riemann 
from agd import Eikonal
import numpy as np

class MPSStructure():
    def __init__(self,size1,size2):
        self.nodes = np.zeros([size1,size2],dtype = Node)
        self.connections = np.zeros([size1,size2,8],dtype = Connection)
        self.directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        self.reversed_directions = [(1, 1), (1, 0), (1, -1), (0, 1), (0, -1), (-1, 1), (-1, 0), (-1, -1)]
        self.size1 = size1
        self.size2 = size2
        for i in range(size1):
            for j in range(size2):
                for n in range(8):
                    ii = i + self.directions[n][0]
                    jj = j + self.directions[n][1]
                    if ii>=0 and jj>=0 and ii<self.nodes.shape[0] and jj<self.nodes.shape[1]:
                        self.connections[i,j,n] = Connection(self.nodes[i,j],self.nodes[ii,jj])
                    else :
                        self.connections[i,j,n] = Connection()
                        
    def update(self):
        for i in range(self.size1):
            for j in range(self.size2):
                for n in range(8):
                    ii = i + self.directions[n][0]
                    jj = j + self.directions[n][1]
                    if ii>=0 and jj>=0 and ii<self.nodes.shape[0] and jj<self.nodes.shape[1]:
                        self.connections[i,j,n] = Connection(self.nodes[i,j],self.nodes[ii,jj])
    
    def apply_Tc(self,kc):
        checked_connections_for_mean_cost = []
        costs = []
        for i in range(self.connections.shape[0]):
            for j in range(self.connections.shape[1]):
                for n in range(self.connections.shape[2]):
                    if self.connections[i,j,n] not in checked_connections_for_mean_cost and self.connections[i,j,n].active:
                        costs.append(self.connections[i,j,n].cost) 
                        checked_connections_for_mean_cost.append(self.connections[i,j,n]) 
        self.mean_c = sum(costs)/len(costs)
        self.std_c = np.std(costs)
        
        Tc = self.mean_c - kc*self.std_c
        for i in range(self.connections.shape[0]):
            for j in range(self.connections.shape[1]):
                for n in range(self.connections.shape[2]):
                    if cells.connections[i,j,n].active:
                        if cells.connections[i,j,n].cost>Tc:
                            cells.connections[i,j,n].active = False
                            
    def activate_connections(self):
        for i in range(self.connections.shape[0]):
            for j in range(self.connections.shape[1]):
                for n in range(self.connections.shape[2]):
                    ii = i + self.directions[n][0]
                    jj = j + self.directions[n][1]
                    if ii>=0 and jj>=0 and ii<self.nodes.shape[0] and jj<self.nodes.shape[1]:
                        if type(self.connections[i,j,n].cost) == np.float64:
                            self.connections[i,j,n].active = True
                            
    def find_skeleton(self):
        self.checked_connections = []
        self.skeleton = dict({})
        for i in range(self.connections.shape[0]):
            for j in range(self.connections.shape[1]):
                self.skeleton[i,j] = self._find_childs([i,j])
                
    def _find_childs(self,d):
        childs = dict({})
        for n in range(self.connections.shape[2]):
            if self.connections[d[0],d[1],n].active:
                if self.connections[d[0],d[1],n] not in self.checked_connections:
                    self.checked_connections.append(self.connections[d[0],d[1],n])
                    next_childs = self._find_childs([d[0]+self.directions[n][0],d[1]+self.directions[n][1]])
                    childs[d[0]+self.directions[n][0],d[1]+self.directions[n][1]] = next_childs
        return childs
    
    def find_junctions(self):
        self.checked_nodes = []
        self.junctions = []
        for i in range(self.nodes.shape[0]):
            for j in range(self.nodes.shape[1]):
                if self.skeleton[i,j] != {}:
                    if [i,j] not in self.checked_nodes:
                        self.checked_nodes.append([i,j])
                    elif [i,j] not in self.junctions: 
                        self.junctions.append([i,j])
                self._find_junctions_helper(self.skeleton[i,j],[i,j])
            
    def _find_junctions_helper(self,skeleton_part,p):
        i = p[0]
        j = p[1]
        for k in skeleton_part.keys():
            if [k[0],k[1]] not in self.checked_nodes:
                self.checked_nodes.append([k[0],k[1]])
            elif [k[0],k[1]] not in self.junctions: 
                self.junctions.append([k[0],k[1]])
                
            self._find_junctions_helper(skeleton_part[k],k)
#             if k in self.checked_nodes:

    def find_skeleton_linear(self):
        self.skeleton_linear = deepcopy(self.skeleton)
        for i in range(self.connections.shape[0]):
            for j in range(self.connections.shape[1]):
                self._linearize_skeleton(cells.skeleton[i,j],[i,j])
                
        for i in range(self.connections.shape[0]):
            for j in range(self.connections.shape[1]):
                self._cut_junctions(self.skeleton_linear[i,j],[i,j])
                
    def _linearize_skeleton(self,skeleton_element,d):
        for k in skeleton_element.keys():
            if [d[0],d[1]] in self.junctions:
                self.skeleton_linear[d[0],d[1]][k[0],k[1]] = skeleton_element[k[0],k[1]]
            self._linearize_skeleton(skeleton_element[k[0],k[1]],k)
            
    def _cut_junctions(self,skeleton_element,d):
        for k in skeleton_element.keys():
            if [k[0],k[1]] in self.junctions:
                skeleton_element[k[0],k[1]] = {}
            else :
                self._cut_junctions(skeleton_element[k[0],k[1]],[k[0],k[1]])
                
    def apply_Tc2(self,kc2):
        checked_connections_for_mean_cost = []
        costs = []
        for i in range(self.connections.shape[0]):
            for j in range(self.connections.shape[1]):
                cost = self.skeleton_element_cost(self.skeleton_linear[i,j],[i,j])
                costs.append(cost)
        self.mean_c2 = sum(costs)/len(costs)
        self.std_c2 = np.std(costs)
        
        Tc = self.mean_c2 - kc2*self.std_c2
        for i in range(self.connections.shape[0]):
            for j in range(self.connections.shape[1]):
                cost = self.skeleton_element_cost(self.skeleton_linear[i,j],[i,j])
                if cost>Tc:
                    self.skeleton_linear[i,j] = {}

    def plt_skeleton(self,skeleton_part,p,linewidth = 1):
        i = p[0]
        j = p[1]
        keys = skeleton_part.keys()
        for k in keys:
            ii = k[0] - i
            jj = k[1] - j
            n = self.directions.index((ii,jj))
            plt.plot(self.connections[i,j,n].path[:,1],self.connections[i,j,n].path[:,0],'r-',markersize = 1,
                     linewidth = linewidth)
            self.plt_skeleton(skeleton_part[k],k,linewidth)
            
    def _skeleton_track(self,skeleton_part,p):
        i = p[0]
        j = p[1]
        keys = skeleton_part.keys()
        for k in keys:
            ii = k[0] - i
            jj = k[1] - j
            n = self.directions.index((ii,jj))
            self.Track_x = np.concatenate([self.Track_x,self.connections[i,j,n].path[:,0]])
            self.Track_y = np.concatenate([self.Track_y,self.connections[i,j,n].path[:,1]])
            self._skeleton_track(skeleton_part[k],k)
            
    def find_skeleton_track(self):
        self.Track_x = np.empty(shape = [0])
        self.Track_y = np.empty(shape = [0])
        for i in range(self.connections.shape[0]):
            for j in range(self.connections.shape[1]):
                self._skeleton_track(self.skeleton_linear[i,j],[i,j])
        
    def skeleton_element_length(self,skeleton_part,p):
        l = 0
        c = 0
        i = p[0]
        j = p[1]
        keys = skeleton_part.keys()
        for k in keys:
            ii = k[0] - i
            jj = k[1] - j
            n = self.directions.index((ii,jj))
            l = l + cells.connections[i,j,n].length + self.skeleton_element_length(skeleton_part[k],k)
        return l
    
    def skeleton_element_cost(self,skeleton_part,p):
        c = 0
        i = p[0]
        j = p[1]
        keys = skeleton_part.keys()
        for k in keys:
            ii = k[0] - i
            jj = k[1] - j
            n = self.directions.index((ii,jj))
            c = c + cells.connections[i,j,n].cost + self.skeleton_element_cost(skeleton_part[k],k)
        return c
        
        
    def apply_Ts(self,Ts):
        for i in range(self.connections.shape[0]):
            for j in range(self.connections.shape[1]):
                l = self.skeleton_element_length(self.skeleton[i,j],[i,j])
                if l<Ts:
                    self.skeleton[i,j] = {}
                    
class Connection():
    def __init__(self,node1 = [],node2 = []):
        self.node1 = node1
        self.node2 = node2
        self.active = False
        self.cost = []
        self.length = []
        self.path = []
        
    def update_connection(self,path,cost,length):
        self.cost = cost
        self.length = length
        self.path = path
        self.active = True
        
    
class Node():
    def __init__(self,image_gray,x1,x2,y1,y2,i,j):
        self.x1 = np.max([x1,0])
        self.x2 = np.min([x2,image_gray.shape[0]])
        self.y1 = np.max([y1,0])
        self.y2 = np.min([y2,image_gray.shape[1]])
        self.i = i
        self.j = j
        
        self.min_point_val = image_gray[self.x1:self.x2,self.y1:self.y2].min()
        self.min_point_pos_cell = np.argwhere(image_gray[self.x1:self.x2,self.y1:self.y2] == self.min_point_val)[0]
        self.min_point_pos_image = self.min_point_pos_cell + np.array([self.x1,self.y1])
    
        self.active = True

    def Min_value_threshold(self,Te):
        if self.min_point_val>Te:
            self.active = False
        
        
def Dijkstars(seed,tip,cost):

    b = np.array([0,cost.shape[0]])
    c = np.array([0,cost.shape[1]])
    sides = np.array([b,c])
    dims = np.array([cost.shape[0],cost.shape[1]])
    
    mu = 0
    l = 1
    p = 2
    if cost.shape[0] == 1:
        cost1 = np.concatenate([cost,cost],0)
        DxZ,DyZ = np.gradient(cost1)
    if cost.shape[1] == 1:
        cost1 = np.concatenate([cost,cost],1)
        DxZ,DyZ = np.gradient(cost1) 
    else :
        DxZ,DyZ = np.gradient(cost) 
    a11 = scipy.ndimage.gaussian_filter(mu*DxZ**2, 1, order=(0,0))
    a12 = scipy.ndimage.gaussian_filter(mu*DxZ*DyZ, 1, order=(0,0))
    a21 = scipy.ndimage.gaussian_filter(mu*DxZ*DyZ, 1, order=(0,0))
    a22 = scipy.ndimage.gaussian_filter(mu*DyZ**2, 1, order=(0,0))

    df = np.array([[1+a11,a12],[a21,1+a22]])
    metric1 = (0.0001+cost*l)**p*df
    
    metric = Riemann(metric1)
    hfmIn = Eikonal.dictIn({
        'model' : 'Riemann2',
        'seeds' : np.expand_dims(seed,axis = 0),
        'arrayOrdering' : 'RowMajor',
        'tips' : np.expand_dims(tip,axis = 0),
        'metric' : metric})
    hfmIn['order']=2
    hfmIn.SetRect(sides = sides, dims = dims)
    hfmOut = hfmIn.Run()
    geos1 = [g.T for g in hfmOut['geodesics']]
    path = np.array(geos1[0],dtype = int)
    path1 = np.unique(path,axis = 0)
    val = np.sum(cost[path1[:,0],path1[:,1]])
    return geos1[0],val

def init_points(cells, image_gray, P, ke):
    Te = np.mean(image_gray) - ke*np.std(image_gray)
    for i in range(0, int(np.ceil(image_gray.shape[0])), P):
        ii = int(np.ceil(i/P))
        for j in range(0, int(np.ceil(image_gray.shape[1]) ), P):
            jj = int(np.ceil(j/P))
            cell = Node(image_gray, i, i+P, j, j+P,ii,jj)
            cell.Min_value_threshold(Te)
            cells.nodes[ii,jj] = cell

    cells.update()
    return cells

def calc_paths(cells,image_gray):
    k = 0
    paths = []
    for i in range(cells.connections.shape[0]):
        for j in range(cells.connections.shape[1]):
            if cells.nodes[i,j].active:
                for n in range(cells.connections.shape[2]):
                    if cells.connections[i,j,n].node1 == []:
                        continue

                    center_node = cells.connections[i,j,n].node1
                    side_node = cells.connections[i,j,n].node2
                    if side_node.active == False:
                        continue
                    x1 = np.min([side_node.x1,center_node.x1])
                    x2 = np.max([side_node.x2,center_node.x2])
                    y1 = np.min([side_node.y1,center_node.y1])
                    y2 = np.max([side_node.y2,center_node.y2])

                    seed = center_node.min_point_pos_image - np.array([x1,y1])
                    tip = side_node.min_point_pos_image - np.array([x1,y1])

                    cost_function = image_gray[x1:x2,y1:y2]
                    if np.sum(np.array(cost_function.shape) == 1) >=1:
                        continue
                    path,cost = Dijkstars(seed,tip,cost_function)
                    path[:,0] = path[:,0] + x1
                    path[:,1] = path[:,1] + y1
                    l = np.unique(np.array(path).astype(int),axis = 0).shape[0]
                    c = cost/l
                    cells.connections[i,j,n].update_connection(path,c,l)
                    n_1 = cells.directions[n]
                    n_2 = cells.reversed_directions.index(n_1)
                    ii = side_node.i
                    jj = side_node.j
                    cells.connections[ii,jj,n_2] = cells.connections[i,j,n]

                    k = k+1
    return cells