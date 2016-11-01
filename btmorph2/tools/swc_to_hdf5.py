"""
Converter to translate SWC format to NMF format. 
(NMF: Neuronal Morphology Format; a temporary working name)

Benjamin Torben-Nielsen, btorbennielsen@gmail.com
"""

import h5py
import numpy as np
from btmorph2.btstructs import NeuronMorphology

def swc_to_nmf(file_name,out_file=None):
    """
    Load an SWC file and store in NTF format based on HDF5.
    Uses NeuronMorphology to load file as it automatically deals
    with most soma-types outlined on NeuroMorpho.org
    (http://neuromorpho.org/SomaFormat.html)

    Parameters
    ----------
    file_name : string
        path to SWC file
    out_file : string
        path to output file. Default is None and will simply add
        .ntf to the 'file_name'
    """
    structure = NeuronMorphology(input_file=file_name)
    all_nodes = structure.tree.get_nodes()
    ids = [n.index for n in all_nodes]
    parents = [n.parent.index if n.parent != None else -1 for n in all_nodes]
    xs = [n.content['p3d'].xyz[0] for n in all_nodes]
    ys = [n.content['p3d'].xyz[1] for n in all_nodes]
    zs = [n.content['p3d'].xyz[2] for n in all_nodes]
    rs = [n.content['p3d'].radius for n in all_nodes]
    s_types = [n.content['p3d'].segtype for n in all_nodes]
    # for i,p in zip(ids,parents):
    #     print("{} has parent: {}".format(i,p))

    # print("xs: {}".format(xs))
    
    # create HDF5 NTF file
    if out_file == None:
        out_file = file_name+".nmf"
    f = h5py.File(out_file, "w")
    print("f.name: {}".format(f))
    
    swc_grp = f.create_group("swc")
    swc_grp.create_dataset("index",data=ids,dtype="int32")
    swc_grp.create_dataset("type",data=s_types,dtype="int8")
    swc_grp.create_dataset("x",data=xs,dtype="float64")
    swc_grp.create_dataset("y",data=ys,dtype="float64")
    swc_grp.create_dataset("z",data=zs,dtype="float64")
    swc_grp.create_dataset("r",data=rs,dtype="float64")
    swc_grp.create_dataset("parent_index",data=parents,dtype="int32")
    # add meta-data about the SWC soma-type
    """internally used codes for soma types are btmorph specific
    From btstructs.Tree._determine_soma_type
    soma_type : int
        Integer indicating one of the supported SWC soma formats.
        0: 1-point soma
        1: Default three-point soma
        2: multiple cylinder description,
        3: otherwise [not suported in btmorph]
    """
    soma_code = structure.tree.soma_type
    if soma_code==0:
        swc_grp.attrs["soma_type"]=np.string_("1_point_soma")
    elif soma_code==1:
        swc_grp.attrs["soma_type"]=np.string_("3_point_soma")
    elif soma_code==2:
        swc_grp.attrs["soma_type"]=np.string_("multiple_cylinder_soma")
    else:
        swc_grp.attrs["soma_type"]=np.string_("unsupported_soma")

    # close the HDF5 file
    f.close()

def curate_ntf():
    pass

if __name__=="__main__":
    import sys
    fn = sys.argv[1]
    swc_to_nmf(file_name=fn)
    
