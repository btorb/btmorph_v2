import sys,os,copy
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

v_to_check = [0]
col_to_check = []

def prim_mst_x(cost_matrix,distance_matrix,\
                   v_new,all_edges,path_Ls,bf,
                   out_degree_of,
                   bifs_only=True):
    global col_to_check
    """
    Pick the vetices with lowest cost, make an edge and return 
    the upated list of covered vertices as edges
    """

    # add the bf*path_L to the cost matrix
    # add to each column in row n (that is not np.inf)
    cost_copy = copy.copy(cost_matrix)
    for row in range(distance_matrix.shape[0]):
        cost_matrix[row,:]=cost_matrix[row,:]+bf*path_Ls[row]
    
    mini = np.min(cost_matrix[v_to_check,:])

    
    #n,m = np.unravel_index( np.argmin(cost_matrix),dims=cost_matrix.shape  )
    n,m = np.unravel_index( np.argmin(cost_matrix[v_to_check,:]),\
                                dims=cost_matrix[v_to_check,:].shape  )
    # print("checking this bit now:\n{}".format(cost_matrix[v_new,:]))
    # print("({},{}) is argmin at {}".format(n,m,mini))
    # print("v_new={}".format(v_new))
    n = v_to_check[n]
    print("[{} added]({},{}) is argmin at {}".format(len(v_new),n,m,mini))

    v_new.append(m)
    all_edges.append((n,m))
    out_degree_of[n]=out_degree_of[n]+1
    path_Ls[m]=path_Ls[n]+distance_matrix[n,m]

    # add m, as usual
    v_to_check.append(m)

    # use the un-bf-ed cost_matrix again
    cost_matrix = cost_copy
    
    # exclude cycles the potential to form loops/cycles
    cost_matrix[v_new,m]=np.inf
    cost_matrix[m,v_new]=np.inf

    # if bifs_only: exclude vertices with out_degree >=2
    ind = np.where(out_degree_of>=2)
    cost_matrix[ind,:]=np.inf
    cost_matrix[:,ind]=np.inf

    # remove bifurcation also from the v_to_check
    for ppp in ind[0]:
        try:
            v_to_check.remove(ppp)
        except:
            pass
    for ppp in ind[0]:
        try:
            col_to_check.remove(ppp)
        except:
            pass
    
    #print("v_to_check:{}, v_new:{}".format(v_to_check,v_new))

    # print("prim_mst_x updated, new vertex: {} and edge: ({},{})".format(n,n,m))
    # print("out_degrees: {}".format(out_degree_of))
    # print("path_Ls: {}".format(path_Ls))
    # print("cost_matrix to be returned:\n{}".format(cost_matrix))

    return v_new, all_edges, cost_matrix, path_Ls, out_degree_of
    

class MSTSynth(object):
    def __init__(self):
        pass

    def _load_dots(self,dots_file,down_sample_factor=1):
        #data = np.loadtxt(dots_file)
        data = np.genfromtxt(dots_file)
        if down_sample_factor == 1 :
            return data
        down_size = data.shape[0] / down_sample_factor
        return data[np.random.randint(data.shape[0],size=down_size)]

    def connect_dots_to_tree(self,dots_file,bf=1.6,out_f="test.swc",\
                             vicinity=50):
        global col_to_check
        """
        Parameters
        ----------
        vicinity : real
            Distances between points larger than this value become np.inf
            to increase sparseness of matrix
        """
        data = self._load_dots(dots_file)
        no_points = data.shape[0]
        print("found {} data points".format(no_points))

        col_to_check=range(no_points)

        # compute original Euclidean distance matrix between all points
        distance_matrix = np.tile(np.inf,(no_points,no_points))
        for i in range(no_points):
            for j in range(i,no_points):#range(i,no_points):
                if i==j:
                    distance_matrix[i,i]=np.inf
                else:
                    p1 = data[i,:]
                    p2 = data[j,:]
                    d = np.sqrt(  (p1[0]-p2[0])**2 + \
                                      (p1[1]-p2[1])**2 + \
                                      (p1[2]-p2[2])**2 )                
                    distance_matrix[i,j]=d
                    distance_matrix[j,i]=d
        print("Original, complete distance matrix: \n{}".format(distance_matrix))

        vertices, edges, new_cost_matrix,path_Ls,out_degree_of \
          = prim_mst_x(distance_matrix,\
                                             distance_matrix,\
                                             v_new=[0],all_edges=[],\
                                             path_Ls=np.zeros(no_points),\
                                             bf=bf,\
                                             out_degree_of=np.zeros(no_points,dtype=int))
        while(len(vertices) < no_points):
            t1 = datetime.now()
            vertices, edges, new_cost_matrix,path_Ls,out_degree_of = \
                prim_mst_x(new_cost_matrix,\
                               distance_matrix,\
                               v_new=vertices,all_edges=edges,\
                               path_Ls=path_Ls,bf=bf,\
                               out_degree_of=out_degree_of)
            t2 = datetime.now()
            print("step took {}s".format((t2-t1).total_seconds()))

        # in accordance to the NeuroMorpho.org standard
        # to_write =  "1 1 0 0 0 2 -1\n"
        # to_write += "2 1 0 2 0 2 1\n"
        # to_write += "3 1 0 -1 0 2 1\n"
        p0 = data[0,:]
        to_write =  "1 1 {} {} {} 2 -1\n".format(p0[0],p0[1],p0[2])
        to_write += "2 1 {} {} {} 2 1\n".format(p0[0],p0[1]+p0[1]/2,p0[2])
        to_write += "3 1 {} {} {} 2 1\n".format(p0[0],p0[1]-p0[1]/2,p0[2]) 
        index = 4
        
        for i in range(1,no_points): # assume that point 1 is the soma...
            p = data[i,:]
            to_write += str(index) + " 3  " \
              + str(p[0])+ " " + str(p[1]) + " " + str(p[2]) \
              + " 1 "
            for edge in edges :
                if edge[1] == i :
                    # SWC starts at 1 + offset from three-point soma, offset=3
                    if i==1:
                        to_write += '1\n'
                    else:
                        to_write += str(edge[0]+3) + '\n'
            index = index +  1
        #print("to_write:\n{}".format(to_write))
        out_file = open(out_f,'w')
        out_file.write(to_write)
        out_file.flush()
        out_file.close()            

if __name__=="__main__":
    print("starting...")
    f = sys.argv[1]
    try:
        bf = float(sys.argv[2])
    except Exception:
        bf = 1.6
    try:
        out_f = sys.argv[3]
    except Exception:
        out_f = "test.swc"
    synth = MSTSynth()
    synth.connect_dots_to_tree(f,bf=bf,out_f=out_f)
    

