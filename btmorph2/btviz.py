"""
Basic visualization of neurite morphologies using matplotlib.

Usage is restricted to morphologies in the sWC format with the three-point soma `standard <http://neuromorpho.org/neuroMorpho/SomaFormat.html>`_

B. Torben-Nielsen
"""
import sys,time
from matplotlib.cm import get_cmap
from Crypto.Protocol.AllOrNothing import isInt
sys.setrecursionlimit(10000)

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.animation as animation

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

import btmorph2
from btmorph2 import config
from numpy import mean,cov,double,cumsum,dot,linalg,array,rank
from pylab import plot,subplot,axis,stem,show,figure, Normalize
""" internal constants required for the dendrogram generation """
H_SPACE = 20
V_SPACE = 0
C = 'k'

max_width = 0
max_height = 0


def plot_2D(neuron, color_scheme="default", color_mapping=None,
            synapses=None, save_image=None, depth='y',show_radius=True):
    """
    2D matplotlib plot of a neuronal moprhology. Projection can be in XY and XZ.
    The SWC has to be formatted with a "three point soma".
    Colors can be provided

    Parameters
    -----------
    color_scheme: string
        "default" or "neuromorpho". "neuronmorpho" is high contrast
    color_mapping: list[float] or list[list[float,float,float]]
        Default is None. If present, this is a list[N] of colors
        where N is the number of compartments, which roughly corresponds to the
        number of lines in the SWC file. If in format of list[float], this list
        is normalized and mapped to the jet color map, if in format of 
        list[list[float,float,float,float]], the 4 floats represt R,G,B,A 
        respectively and must be between 0-255. When not None, this argument
        overrides the color_scheme argument(Note the difference with segments).
    synapses : vector of bools
        Default is None. If present, draw a circle or dot in a distinct color 
        at the location of the corresponding compartment. This is a 
        1xN vector.
    save_image: string
        Default is None. If present, should be in format "file_name" without 
        an extension, and an animation will be saved as this filename.
    depth : string
        Default 'y' means that X represents the superficial to deep axis. \
        Otherwise, use 'z' to conform the mathematical standard of having the Z axis.
    show_radius : boolean
        True (default) to plot the actual radius. If set to False,
        the radius will be taken from `btmorph2\config.py`
    """

    # print "scheme: ", config.c_scheme_nm
    if show_radius==False:
        plot_radius = config.fake_radius

    # my_color_list = ['r','g','b','c','m','y','k','g','DarkGray']
    if color_scheme == 'default':
        my_color_list = config.c_scheme_default['neurite']
    elif color_scheme == 'neuromorpho':
        my_color_list = config.c_scheme_nm['neurite']
    print 'my_color_list: ', my_color_list
    
    scalarMap = None
    # setting up for a colormap
    
    if color_mapping is not None:
        if isinstance(color_mapping[0], int):
            jet = plt.get_cmap('jet')
            norm = colors.Normalize(np.min(color_mapping), np.max(color_mapping))
            scalarMap = cm.ScalarMappable(norm=norm, cmap=jet)

            Z = [[0, 0], [0, 0]]
            levels = np.linspace(np.min(color_mapping), np.max(color_mapping), 100)

            CS3 = plt.contourf(Z, levels, cmap=jet)
            plt.clf()

    min_depth=100

    if depth is "y":
        ax = 1
    elif depth is "z":
        ax = 2

    # ax = plt.subplot2grid((2,2), (0, 0))

    index = 0
    for node in neuron.tree:
        if index < 3:
            pass
        else:
            C = node
            P = node.parent
            if show_radius==False:
                line_width = plot_radius
            else:
                line_width = C.content['p3d'].radius
            if color_mapping is None:
                pl = plt.plot([P.content['p3d'].xyz[0], C.content['p3d'].xyz[0]], [P.content['p3d'].xyz[ax],C.content['p3d'].xyz[ax]], my_color_list[C.content['p3d'].segtype-1], linewidth=line_width, zorder=1)
            else:
    
                if isinstance(color_mapping[0], int):
                    c = scalarMap.to_rgba(color_mapping[index])
                elif isinstance(color_mapping[0], list):
                    c = [float(x) / 255 for x in color_mapping[index]]
                pl = plt.plot([P.content['p3d'].xyz[0], C.content['p3d'].xyz[0]], [P.content['p3d'].xyz[ax], C.content['p3d'].xyz[ax]], c=c , linewidth=line_width, zorder=1)
    
            # add the synapses
            if synapses is not None:
                if synapses[index]:
                    pl = plt.scatter([C.content['p3d'].xyz[0]], C.content['p3d'].xyz[ax], c='r', s = 75, zorder=2)

            min_depth = C.content['p3d'].xyz[ax] if C.content['p3d'].xyz[ax] < min_depth else min_depth

        index += 1
        # TODO: insert synapses here
    plt.xlabel("X")
    if ax is 1: 
        plt.ylabel("Y")
    else: 
        plt.ylabel("Z")

    plt.tight_layout()
    plt.axis("equal")

    # color bar? in case `color_mapping` is used
    if color_mapping is not None:
        if isinstance(color_mapping[0], int):
            cb = plt.colorbar(CS3) # bit of a workaround, but it seems to work
            ticks_f = np.linspace(np.min(color_mapping),np.max(color_mapping),5 )
            ticks_i = map(int, ticks_f)
            cb.set_ticks(ticks_i)

    # set the bg color
    fig = plt.gcf()
    ax = fig.gca()
    if color_scheme == 'default':
        ax.set_axis_bgcolor(config.c_scheme_default['bg'])
    elif color_scheme == 'neuromorpho':
        ax.set_axis_bgcolor(config.c_scheme_nm['bg'])

    if save_image is not None:
        plt.savefig(save_image)
    else:
        plt.show()


def plot_3D(neuron, color_scheme="default", color_mapping=None,
            synapses=None, save_image="animation",show_radius=True):
    """
    3D matplotlib plot of a neuronal morphology. The SWC has to be formatted with a "three point soma".
    Colors can be provided and synapse location marked

    Parameters
    -----------
    color_scheme: string
        "default" or "neuromorpho". "neuronmorpho" is high contrast
    color_mapping: list[float] or list[list[float,float,float]]
        Default is None. If present, this is a list[N] of colors
        where N is the number of compartments, which roughly corresponds to the
        number of lines in the SWC file. If in format of list[float], this list
        is normalized and mapped to the jet color map, if in format of 
        list[list[float,float,float,float]], the 4 floats represt R,G,B,A 
        respectively and must be between 0-255. When not None, this argument
        overrides the color_scheme argument(Note the difference with segments).
    synapses : vector of bools
        Default is None. If present, draw a circle or dot in a distinct color 
        at the location of the corresponding compartment. This is a 
        1xN vector.
    save_image: string
        Default is None. If present, should be in format "file_name.extension",
        and figure produced will be saved as this filename.
    show_radius : boolean
        True (default) to plot the actual radius. If set to False,
        the radius will be taken from `btmorph2\config.py`
    """

    if show_radius==False:
        plot_radius = config.fake_radius
    
    if color_scheme == 'default':
        my_color_list = config.c_scheme_default['neurite']
    elif color_scheme == 'neuromorpho':
        my_color_list = config.c_scheme_nm['neurite']
    else:
        raise Exception("Not valid color scheme")
    print 'my_color_list: ', my_color_list

    fig, ax = plt.subplots()

    if color_mapping is not None:
        if isinstance(color_mapping[0], int):
            jet = plt.get_cmap('jet')
            norm = colors.Normalize(np.min(color_mapping), np.max(color_mapping))
            scalarMap = cm.ScalarMappable(norm=norm, cmap=jet)

            Z = [[0, 0], [0, 0]]
            levels = np.linspace(np.min(color_mapping), np.max(color_mapping), 100)
            CS3 = plt.contourf(Z, levels, cmap=jet)
            plt.clf()

    ax = fig.gca(projection='3d')

    index = 0

    for node in neuron.tree: # not ordered but that has little importance here
        # draw a line segment from parent to current point
        c_x = node.content['p3d'].xyz[0]
        c_y = node.content['p3d'].xyz[1]
        c_z = node.content['p3d'].xyz[2]
        c_r = node.content['p3d'].radius

        if index < 3:
            pass
        else:
            parent = node.parent
            p_x = parent.content['p3d'].xyz[0]
            p_y = parent.content['p3d'].xyz[1]
            p_z = parent.content['p3d'].xyz[2]
            # p_r = parent.content['p3d'].radius
            # print 'index:', index, ', len(cs)=', len(color_mapping)
            if show_radius==False:
                line_width = plot_radius
            else:
                line_width = c_r/2.0
            
            if color_mapping is None:
                ax.plot([p_x, c_x], [p_y, c_y], [p_z, c_z], my_color_list[node.content['p3d'].segtype - 1], linewidth=line_width)
            else:
                if isinstance(color_mapping[0], int):
                    c = scalarMap.to_rgba(color_mapping[index])
                elif isinstance(color_mapping[0], list):
                    c = [float(x) / 255 for x in color_mapping[index]]

                ax.plot([p_x, c_x], [p_y, c_y], [p_z, c_z], c=c, linewidth=c_r/2.0)
            # add the synapses
        if synapses is not None:
            if synapses[index]:
                ax.scatter(c_x, c_y, c_z, c='r')

        index += 1

    minv, maxv = neuron.get_boundingbox()
    minv = min(minv)
    maxv = max(maxv)
    ax.auto_scale_xyz([minv, maxv], [minv, maxv], [minv, maxv])

    index = 0

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if color_mapping is not None:
        if isinstance(color_mapping[0], int):
            cb = plt.colorbar(CS3) # bit of a workaround, but it seems to work
            ticks_f = np.linspace(np.min(color_mapping)-1, np.max(color_mapping)+1, 5)
            ticks_i = map(int, ticks_f)
            cb.set_ticks(ticks_i)

    # set the bg color
    fig = plt.gcf()
    ax = fig.gca()
    if color_scheme == 'default':
        ax.set_axis_bgcolor(config.c_scheme_default['bg'])
    elif color_scheme == 'neuromorpho':
        ax.set_axis_bgcolor(config.c_scheme_nm['bg'])

    if save_image is not None:
        plt.savefig(save_image)

    plt.show()

    return fig

def animate(neuron, color_scheme="default", color_mapping=None,
            synapses=None, save_image=None, axis="z"):
    """
    3D matplotlib plot of a neuronal morphology. The SWC has to be formatted with a "three point soma".
    Colors can be provided and synapse location marked

    Parameters
    -----------
    color_scheme: string
        "default" or "neuromorpho". "neuronmorpho" is high contrast
    color_mapping: list[float] or list[list[float,float,float]]
        Default is None. If present, this is a list[N] of colors
        where N is the number of compartments, which roughly corresponds to the
        number of lines in the SWC file. If in format of list[float], this list
        is normalized and mapped to the jet color map, if in format of 
        list[list[float,float,float,float]], the 4 floats represt R,G,B,A 
        respectively and must be between 0-255. When not None, this argument
        overrides the color_scheme argument(Note the difference with segments).
    synapses : vector of bools
        Default is None. If present, draw a circle or dot in a distinct color 
        at the location of the corresponding compartment. This is a 
        1xN vector.
    save_image: string
        Default is None. If present, should be in format "file_name.extension",
        and figure produced will be saved as this filename.

    """

    if color_scheme == 'default':
        my_color_list = config.c_scheme_default['neurite']
    elif color_scheme == 'neuromorpho':
        my_color_list = config.c_scheme_nm['neurite']
    else:
        raise Exception("Not valid color scheme")
    print 'my_color_list: ', my_color_list

    fig, ax = plt.subplots()

    if color_mapping is not None:
        if isinstance(color_mapping[0], int):
            jet = plt.get_cmap('jet')
            norm = colors.Normalize(np.min(color_mapping), np.max(color_mapping))
            scalarMap = cm.ScalarMappable(norm=norm, cmap=jet)

            Z = [[0, 0], [0, 0]]
            levels = np.linspace(np.min(color_mapping), np.max(color_mapping), 100)
            CS3 = plt.contourf(Z, levels, cmap=jet)
            plt.clf()

    ax = fig.gca(projection='3d')

    index = 0

    for node in neuron.tree: # not ordered but that has little importance here
        # draw a line segment from parent to current point
        c_x = node.content['p3d'].xyz[0]
        c_y = node.content['p3d'].xyz[1]
        c_z = node.content['p3d'].xyz[2]
        c_r = node.content['p3d'].radius

        if index < 3:
            pass
        else:
            parent = node.parent
            p_x = parent.content['p3d'].xyz[0]
            p_y = parent.content['p3d'].xyz[1]
            p_z = parent.content['p3d'].xyz[2]
            # p_r = parent.content['p3d'].radius
            # print 'index:', index, ', len(cs)=', len(color_mapping)
            if color_mapping is None:
                ax.plot([p_x, c_x], [p_y, c_y], [p_z, c_z], my_color_list[node.content['p3d'].segtype - 1], linewidth=c_r/2.0)
            else:
                if isinstance(color_mapping[0], int):
                    c = scalarMap.to_rgba(color_mapping[index])
                elif isinstance(color_mapping[0], list):
                    c = [float(x) / 255 for x in color_mapping[index]]

                ax.plot([p_x, c_x], [p_y, c_y], [p_z, c_z], c=c, linewidth=c_r/2.0)
            # add the synapses
        if synapses is not None:
            if synapses[index]:
                ax.scatter(c_x, c_y, c_z, c='r')

        index += 1

    minv, maxv = neuron.get_boundingbox()
    minv = min(minv)
    maxv = max(maxv)
    ax.auto_scale_xyz([minv, maxv], [minv, maxv], [minv, maxv])

    index = 0

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if color_mapping is not None:
        if isinstance(color_mapping[0], int):
            cb = plt.colorbar(CS3) # bit of a workaround, but it seems to work
            ticks_f = np.linspace(np.min(color_mapping)-1, np.max(color_mapping)+1, 5)
            ticks_i = map(int, ticks_f)
            cb.set_ticks(ticks_i)

    # set the bg color
    fig = plt.gcf()
    ax = fig.gca()
    if color_scheme == 'default':
        ax.set_axis_bgcolor(config.c_scheme_default['bg'])
    elif color_scheme == 'neuromorpho':
        ax.set_axis_bgcolor(config.c_scheme_nm['bg'])

    anim = animation.FuncAnimation(fig, _animate_rotation,fargs=(ax,), frames=60)
    anim.save(save_image + ".gif", writer='imagemagick', fps=4)

    # anim.save(save_image + ".gif", writer='ffmpeg', fps=4)
    

    return fig

def _animate_rotation(nframe,fargs):
    fargs.view_init(elev=0, azim=nframe*6)

def plot_3D_Forest(neurons, color_scheme="default", save_image=None):
    """
    3D matplotlib plot of a neuronal morphology. The Forest has to be formatted with a "three point soma".
    Colors can be provided and synapse location marked

    Parameters
    -----------
    color_scheme: string
        "default" or "neuromorpho". "neuronmorpho" is high contrast
    save_image: string
        Default is None. If present, should be in format "file_name.extension",
        and figure produced will be saved as this filename.
    """
    my_color_list = ['r','g','b','c','m','y','r--','b--','g--']

    # resolve some potentially conflicting arguments
    if color_scheme == 'default':
        my_color_list = config.c_scheme_default['neurite']
    elif color_scheme == 'neuromorpho':
        my_color_list = config.c_scheme_nm['neurite']
    else:
        raise Exception("Not valid color scheme")
    print 'my_color_list: ', my_color_list

    fig, ax = plt.subplots()

    ax = fig.gca(projection='3d')

    for neuron in neurons:
        index = 0
        for node in neuron.tree:
            c_x = node.content['p3d'].xyz[0]
            c_y = node.content['p3d'].xyz[1]
            c_z = node.content['p3d'].xyz[2]
            c_r = node.content['p3d'].radius

            if index < 3:
                pass
            else:
                parent = node.parent
                p_x = parent.content['p3d'].xyz[0]
                p_y = parent.content['p3d'].xyz[1]
                p_z = parent.content['p3d'].xyz[2]
                # p_r = parent.content['p3d'].radius
                # print 'index:', index, ', len(cs)=', len(color_mapping)

                ax.plot([p_x, c_x], [p_y, c_y], [p_z, c_z], my_color_list[node.content['p3d'].segtype - 1], linewidth=c_r/2.0)
            index += 1

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

    if save_image is not None:
        plt.savefig(save_image)

    return fig

def plot_dendrogram(neuron,transform='plain',shift=0,c='k',radius=True,rm=20000.0,ra=200,outN=None) :
    """
    Generate a dendrogram from an SWC file. The SWC has to be formatted with a "three point soma"

    Parameters
    -----------
    file_name : string
        File name of the SWC file to plots
    transform : string
        Either 'plain' or 'lambda'. Plain means no transform while 'lambda' performs an elecotrtonic transform
    shift : float
        Offset in the x-direction
    c : string
        Color ('r','g', 'b', ...)
    radius : boolean
        Plot a wire (False) dendrogram or one with the thickness of the processes (True)
    rm : float
       Membrane resistance. Only needed when transform = 'lambda'
    rm : float
       Axial resistance. Only needed when transform = 'lambda'
    outN : string
        File name of the output file. Extension of this file sets the file type
    """
    global C, RM, RA, max_width, max_height # n.a.s.t.y.
    
    swc_tree = neuron.tree
    RM = rm
    RA = ra
    C = c
    max_height = 0
    max_width = 0
    plt.clf()
    print 'Going to build the dendrogram. This might take a while...'
    ttt = time.time()
    _expand_dendrogram(swc_tree.root,swc_tree,shift,0,radius=radius,transform=transform)
    if(transform == 'plain') :
        plt.ylabel('L (micron)')
    elif(transform == 'lambda') :
        plt.ylabel('L (lambda)')
    print (time.time() - ttt), ' later the dendrogram was finished. '

    print 'max_widht=%f, max_height=%f' % (max_width,max_height)
    x_bound = (max_width / 2.0) + (0.1*max_width)
    max_y_bound = max_height + 0.1*max_height
    plt.axis([-1.0*x_bound,x_bound,-0.1*max_height,max_y_bound])

    plt.plot([x_bound,x_bound],[0,100],'k', linewidth=5) # 250 for MN, 100 for granule

    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    
    if(outN != None) :
        plt.savefig(outN)

def _expand_dendrogram(cNode,swc_tree,off_x,off_y,radius,transform='plain') :
    global max_width,max_height # middle name d.i.r.t.y.
    '''
    Gold old fashioned recursion... sys.setrecursionlimit()!
    '''
    place_holder_h = H_SPACE
    max_degree = swc_tree.degree_of_node(cNode)
    required_h_space = max_degree * place_holder_h
    start_x = off_x-(required_h_space/2.0)
    if(required_h_space > max_width) :
        max_width = required_h_space
    
    if swc_tree.is_root(cNode) :
        print 'i am expanding the root'
        cNode.children.remove(swc_tree.get_node_with_index(2))
        cNode.children.remove(swc_tree.get_node_with_index(3))
    
    for cChild in cNode.children :
        l = _path_between(swc_tree,cChild,cNode,transform=transform)
        r = cChild.content['p3d'].radius

        cChild_degree = swc_tree.degree_of_node(cChild)
        new_off_x = start_x + ( (cChild_degree/2.0)*place_holder_h )
        new_off_y = off_y+(V_SPACE*2)+l
        r = r if radius  else 1
        plt.vlines(new_off_x,off_y+V_SPACE,new_off_y,linewidth=r,colors=C)
        if((off_y+(V_SPACE*2)+l) > max_height) :
            max_height = off_y+(V_SPACE*2)+l

        _expand_dendrogram(cChild,swc_tree,new_off_x,new_off_y,radius=radius,transform=transform)

        start_x = start_x + (cChild_degree*place_holder_h)
        plt.hlines(off_y+V_SPACE,off_x,new_off_x,colors=C)

def _path_between(swc_tree,deep,high,transform='plain') :
    path = swc_tree.path_to_root(deep)
    pl = 0
    pNode = deep
    for node in path[1:] :
        pPos = pNode.content['p3d'].xyz
        cPos = node.content['p3d'].xyz
        pl = pl + np.sqrt(np.sum((cPos-pPos)**2))
        #pl += np.sqrt( (pPos.x - cPos.x)**2 + (pPos.y - cPos.y)**2 + (pPos.z - cPos.z)**2 )
        pNode = node
        if(node == high) : break
        
    if(transform == 'plain'):
        return pl
    elif(transform == 'lambda') :
        DIAM = (deep.content['p3d'].radius*2.0 + high.content['p3d'].radius*2.0) /2.0 # naive...
        c_lambda = np.sqrt(1e+4*(DIAM/4.0)*(RM/RA))
        return pl / c_lambda
        
