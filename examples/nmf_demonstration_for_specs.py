"""
Code to demonstrate the internal structure of NMF files.
Comments are provided in the code to document some details

B. Torben-Nielsen (btorbennielsen@gmail.com)
"""

import h5py
import os
import numpy as np

def create_example_B():
    cdir = os.path.dirname(os.path.abspath(__file__))
    f = h5py.File(cdir+"/example_B.nmf", "w")
    print("f.name: {}".format(f))

    # add time_lapse/1
    ids = [1,2,3]
    s_types = [1,3,3]
    parents = [-1,1,2]
    xs = [0,0,-10]
    ys = [0,10,20]
    zs = [0,0,0]
    rs = [5,1,1]    
    t_parents=[-1,-1,-1] # indicating no temporal parent
    t_events=["new_point","new_point","new_point"] # string entries for now
    tl_group = f.create_group("time_lapse")
    tl_group_1 = tl_group.create_group("1")
    tl_group_1.create_dataset("index",data=ids,dtype="int32")
    tl_group_1.create_dataset("type",data=s_types,dtype="int8")
    tl_group_1.create_dataset("x",data=xs,dtype="float64")
    tl_group_1.create_dataset("y",data=ys,dtype="float64")
    tl_group_1.create_dataset("z",data=zs,dtype="float64")
    tl_group_1.create_dataset("r",data=rs,dtype="float64")
    tl_group_1.create_dataset("parent_index",data=parents,dtype="int32")
    tl_group_1.create_dataset("t_parent_index",data=t_parents,dtype="int32")
    tl_group_1.create_dataset("t_event",data=t_events,dtype="S20")

    # add time_lapse/2
    ids = [1,2,3,4]
    s_types = [1,3,3,3]
    parents = [-1,1,2,2]
    xs = [0,0,-10,10]
    ys = [0,10,20,20]
    zs = [0,0,0,0]
    rs = [5,1,1,1]
    t_parents=[1,2,3,-1] # indicating no temporal parent
    t_events=["existing_point","existing_point","existing_point","new_point"]
    tl_group_2 = tl_group.create_group("2")
    tl_group_2.create_dataset("index",data=ids,dtype="int32")
    tl_group_2.create_dataset("type",data=s_types,dtype="int8")
    tl_group_2.create_dataset("x",data=xs,dtype="float64")
    tl_group_2.create_dataset("y",data=ys,dtype="float64")
    tl_group_2.create_dataset("z",data=zs,dtype="float64")
    tl_group_2.create_dataset("r",data=rs,dtype="float64")
    tl_group_2.create_dataset("parent_index",data=parents,dtype="int32")
    tl_group_2.create_dataset("t_parent_index",data=t_parents,dtype="int32")
    tl_group_2.create_dataset("t_event",data=t_events,dtype="S20")

    # store the final structure in /swc
    # Hard-link the last structure to the /swc group.
    # HDF5 does this without copying the data, but through a real link
    f["swc"] = tl_group_2
    f["swc"].attrs["soma_type"]=np.string_("1_point_soma")    

    f.close()

def create_example_C():
    cdir = os.path.dirname(os.path.abspath(__file__))
    f = h5py.File(cdir+"/example_C.nmf", "w")
    print("f.name: {}".format(f))

    # add time_lapse/1
    ids = [1,2,3]
    s_types = [1,3,3]
    parents = [-1,1,2]
    xs = [0,0,-0]
    ys = [0,10,20]
    zs = [0,0,0]
    rs = [5,1,1]    
    t_parents=[-1,-1,-1] # indicating no temporal parent
    t_events=["new_point","new_point","new_point"] # string entries for now
    tl_group = f.create_group("time_lapse")
    tl_group_1 = tl_group.create_group("1")
    tl_group_1.create_dataset("index",data=ids,dtype="int32")
    tl_group_1.create_dataset("type",data=s_types,dtype="int8")
    tl_group_1.create_dataset("x",data=xs,dtype="float64")
    tl_group_1.create_dataset("y",data=ys,dtype="float64")
    tl_group_1.create_dataset("z",data=zs,dtype="float64")
    tl_group_1.create_dataset("r",data=rs,dtype="float64")
    tl_group_1.create_dataset("parent_index",data=parents,dtype="int32")
    tl_group_1.create_dataset("t_parent_index",data=t_parents,dtype="int32")
    tl_group_1.create_dataset("t_event",data=t_events,dtype="S20")

    # add time_lapse/2
    ids = [1,2,3,]
    s_types = [1,3,3,]
    parents = [-1,1,2,]
    xs = [0,0,0]
    ys = [-10,10,30]
    zs = [0,0,0]
    rs = [5,1,1]    
    t_parents=[1,2,3] # indicating all points existed but moved
    t_events=["moving_point","moving_point","moving_point"] # string entries for now
    tl_group_2 = tl_group.create_group("2")
    tl_group_2.create_dataset("index",data=ids,dtype="int32")
    tl_group_2.create_dataset("type",data=s_types,dtype="int8")
    tl_group_2.create_dataset("x",data=xs,dtype="float64")
    tl_group_2.create_dataset("y",data=ys,dtype="float64")
    tl_group_2.create_dataset("z",data=zs,dtype="float64")
    tl_group_2.create_dataset("r",data=rs,dtype="float64")
    tl_group_2.create_dataset("parent_index",data=parents,dtype="int32")
    tl_group_2.create_dataset("t_parent_index",data=t_parents,dtype="int32")
    tl_group_2.create_dataset("t_event",data=t_events,dtype="S20")

    # Hard-link the last structure to the /swc group.
    # HDF5 does this without copying the data, but through a real link
    f["swc"] = tl_group_2
    f["swc"].attrs["soma_type"]=np.string_("1_point_soma")

    f.close()

def create_static_subcell():
    cdir = os.path.dirname(os.path.abspath(__file__))
    f = h5py.File(cdir+"/example_static_subcell.nmf", "w")
    print("f.name: {}".format(f))

    # add time_lapse/1
    ids = [1,2,3,4,5,6,7,8,9,10,11,12]
    s_types = [3,3,3,3,3,3,3,3,3,3,3,3]
    parents = [-1,1,2,3,4,5,6,7,4,9,10,11]
    xs = [149,140,136,130,125,118,112,105,145,153,161,167]
    ys = [98,112,130,140,152,164,175,184,150,166,172,178]
    zs = [0,0,0,0,0,0,0,0,0,0,0,0]
    rs = [0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6]

    # create the, here static, morphology.
    swc_grp = f.create_group("swc")
    swc_grp.attrs["soma_type"]=np.string_("1_point_soma")
    swc_grp.create_dataset("index",data=ids,dtype="int32")
    swc_grp.create_dataset("type",data=s_types,dtype="int8")
    swc_grp.create_dataset("x",data=xs,dtype="float64")
    swc_grp.create_dataset("y",data=ys,dtype="float64")
    swc_grp.create_dataset("z",data=zs,dtype="float64")
    swc_grp.create_dataset("r",data=rs,dtype="float64")
    swc_grp.create_dataset("parent_index",data=parents,dtype="int32")

    # add annotations for subcellular domains as visualized with
    # additional markers in distinct channels
    # two channels used as illustration
    chan0_grp = f.create_group("channel_0")
    chan0_grp.attrs["channel"]=np.string_("GFP")
    chan0_data = [0,0.25,0.25,0.5,0.5,1,1,0.25,0.5,0.75,0.75,0.75]
    chan0_grp.create_dataset("intensity",data=chan0_data,dtype="float32")
    
    chan1_grp = f.create_group("channel_1")
    chan1_grp.attrs["channel"]=np.string_("RFP")
    chan1_data = [0,0.25,0.25,0,1,1,0,0.25,0.75,1,1,0.5]
    chan1_grp.create_dataset("intensity",data=chan1_data,dtype="float32") 
    
if __name__=="__main__":
    create_example_B()
    create_example_C()
    create_static_subcell()
