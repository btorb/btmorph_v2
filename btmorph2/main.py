
import numpy as np
from IPython.utils.coloransi import ColorScheme

import glob
import btmorph2
import matplotlib.pyplot as plt

for f in glob.glob('genetic_interaction_screen_01/*.swc'):
    n1 = btmorph2.NeuronMorphology(f)
    if n1.total_length() <= 1.0:
        print("no length")
        continue
    orders = [n1.order_of_node(node) for node in n1._all_nodes]
    out_f = f.split(".swc")[0]+"_order_color.png"
    print("{}".format(out_f))
    n1.plot_2D(color_mapping=orders, save_image=out_f)
    plt.close()


'''
swc_neuron = btmorph2.NeuronMorphology("C:/Users/Sam/Downloads/sam_forest_test_HN_P60/sam_forest_test_HN_P60/0000C8.swc")

swc_neuron.plot_2D()
swc_neuron.plot_2D(depth="Z")
swc_neuron.plot_3D()
pop = btmorph2.PopulationMorphology(swc_neuron)
pop.add_neuron(swc_neuron)
'''
'''
n = btmorph2.NeuronMorphology("C:/Users/Sam/Downloads/sam_forest_test_HN_P60/sam_forest_test_HN_P60/0000C8.swc")


xoff = n.tree.root.content['p3d'].xyz[0]
yoff = n.tree.root.content['p3d'].xyz[1]
zoff = n.tree.root.content['p3d'].xyz[2]

xoff = 0
yoff = 0
zoff = 0

f = btmorph2.ForestStructure(n, (-xoff, -yoff, -zoff))

f.add_neuron(btmorph2.NeuronMorphology("C:/Users/Sam/Downloads/sam_forest_test_HN_P60/sam_forest_test_HN_P60/00FF00.swc"),(-xoff,-yoff,-zoff))
f.add_neuron(btmorph2.NeuronMorphology("C:/Users/Sam/Downloads/sam_forest_test_HN_P60/sam_forest_test_HN_P60/1E6CCD.swc"),(-xoff,-yoff,-zoff))
f.add_neuron(btmorph2.NeuronMorphology("C:/Users/Sam/Downloads/sam_forest_test_HN_P60/sam_forest_test_HN_P60/4BB36B.swc"),(-xoff,-yoff,-zoff))
f.add_neuron(btmorph2.NeuronMorphology("C:/Users/Sam/Downloads/sam_forest_test_HN_P60/sam_forest_test_HN_P60/4C1CF2.swc"),(-xoff,-yoff,-zoff))
f.add_neuron(btmorph2.NeuronMorphology("C:/Users/Sam/Downloads/sam_forest_test_HN_P60/sam_forest_test_HN_P60/5CC8BF.swc"),(-xoff,-yoff,-zoff))
f.add_neuron(btmorph2.NeuronMorphology("C:/Users/Sam/Downloads/sam_forest_test_HN_P60/sam_forest_test_HN_P60/9B67B8.swc"),(-xoff,-yoff,-zoff))
f.add_neuron(btmorph2.NeuronMorphology("C:/Users/Sam/Downloads/sam_forest_test_HN_P60/sam_forest_test_HN_P60/32B26B.swc"),(-xoff,-yoff,-zoff))
f.add_neuron(btmorph2.NeuronMorphology("C:/Users/Sam/Downloads/sam_forest_test_HN_P60/sam_forest_test_HN_P60/56AE0B.swc"),(-xoff,-yoff,-zoff))
f.add_neuron(btmorph2.NeuronMorphology("C:/Users/Sam/Downloads/sam_forest_test_HN_P60/sam_forest_test_HN_P60/17245B.swc"),(-xoff,-yoff,-zoff))
f.add_neuron(btmorph2.NeuronMorphology("C:/Users/Sam/Downloads/sam_forest_test_HN_P60/sam_forest_test_HN_P60/22563F.swc"),(-xoff,-yoff,-zoff))
f.add_neuron(btmorph2.NeuronMorphology("C:/Users/Sam/Downloads/sam_forest_test_HN_P60/sam_forest_test_HN_P60/81200F.swc"),(-xoff,-yoff,-zoff))
f.add_neuron(btmorph2.NeuronMorphology("C:/Users/Sam/Downloads/sam_forest_test_HN_P60/sam_forest_test_HN_P60/741851.swc"),(-xoff,-yoff,-zoff))
f.add_neuron(btmorph2.NeuronMorphology("C:/Users/Sam/Downloads/sam_forest_test_HN_P60/sam_forest_test_HN_P60/940117.swc"),(-xoff,-yoff,-zoff))
f.add_neuron(btmorph2.NeuronMorphology("C:/Users/Sam/Downloads/sam_forest_test_HN_P60/sam_forest_test_HN_P60/A35CFB.swc"),(-xoff,-yoff,-zoff))
f.add_neuron(btmorph2.NeuronMorphology("C:/Users/Sam/Downloads/sam_forest_test_HN_P60/sam_forest_test_HN_P60/B84C73.swc"),(-xoff,-yoff,-zoff))
f.add_neuron(btmorph2.NeuronMorphology("C:/Users/Sam/Downloads/sam_forest_test_HN_P60/sam_forest_test_HN_P60/B450F4.swc"),(-xoff,-yoff,-zoff))
f.add_neuron(btmorph2.NeuronMorphology("C:/Users/Sam/Downloads/sam_forest_test_HN_P60/sam_forest_test_HN_P60/C0DCC0.swc"),(-xoff,-yoff,-zoff))
f.add_neuron(btmorph2.NeuronMorphology("C:/Users/Sam/Downloads/sam_forest_test_HN_P60/sam_forest_test_HN_P60/F0D487.swc"),(-xoff,-yoff,-zoff))
f.add_neuron(btmorph2.NeuronMorphology("C:/Users/Sam/Downloads/sam_forest_test_HN_P60/sam_forest_test_HN_P60/FEDCF1.swc"),(-xoff,-yoff,-zoff))
f.add_neuron(btmorph2.NeuronMorphology("C:/Users/Sam/Downloads/sam_forest_test_HN_P60/sam_forest_test_HN_P60/FF0000.swc"),(-xoff,-yoff,-zoff))
f.add_neuron(btmorph2.NeuronMorphology("C:/Users/Sam/Downloads/sam_forest_test_HN_P60/sam_forest_test_HN_P60/FF00FF.swc"),(-xoff,-yoff,-zoff))
f.add_neuron(btmorph2.NeuronMorphology("C:/Users/Sam/Downloads/sam_forest_test_HN_P60/sam_forest_test_HN_P60/FF0080.swc"),(-xoff,-yoff,-zoff))
f.add_neuron(btmorph2.NeuronMorphology("C:/Users/Sam/Downloads/sam_forest_test_HN_P60/sam_forest_test_HN_P60/FFFF00.swc"),(-xoff,-yoff,-zoff))
f.add_neuron(btmorph2.NeuronMorphology("C:/Users/Sam/Downloads/sam_forest_test_HN_P60/sam_forest_test_HN_P60/FFFF80.swc"),(-xoff,-yoff,-zoff))

print 'no_of_neurons'
print f.no_of_neurons()
raw_input("Press Enter to continue...")
print 'no_of_bifurcations'
print f.no_of_bifurcations()
raw_input("Press Enter to continue...")
print "no_terminals"
print f.no_terminals()
raw_input("Press Enter to continue...")
print "no_stems"
print f.no_stems()
raw_input("Press Enter to continue...")
print "total_length"
print f.total_length()
raw_input("Press Enter to continue...")
print "total_surface"
print f.total_surface()
raw_input("Press Enter to continue...")
print "total_volume"
print f.total_volume()
raw_input("Press Enter to continue...")
print "total_dimensions_verbose"
print f.total_dimensions_verbose()
raw_input("Press Enter to continue...")
print "global_horton_strahler"
print f.global_horton_strahler()
raw_input("Press Enter to continue...")
print "get_diameters"
print f.get_diameters()
raw_input
("Press Enter to continue...")

# f.add_neuron(btmorph2.NeuronMorphology('C:/Users/Sam/workspace/btmorph_v2/tests/v_e_moto1.CNG.swc'), (-xoff,-yoff,-zoff))
# f.add_neuron(btmorph2.NeuronMorphology('C:/Users/Sam/workspace/btmorph_v2/examples/new/2005-01-25-A1.CNG.swc'),(-xoff,-yoff,-zoff))

f.plot_3DGL(multisample=True, fast=True)
#f.animationGL(filename="testhighres", zoom=2.75, displaysize=(1920,1080))
'''

'''
# Animate Example
swc_neuron1 = NeuronMorphology('C:/Users/Sam/workspace/btmorph_v2/tests/v_e_moto1.CNG.swc')

swc_neuron1.animationGL(filename='TestX', displaysize=(800,600), radius=3, axis='x')
swc_neuron1.animationGL(filename='TestY', displaysize=(800,600), radius=3, axis='y')
swc_neuron1.animationGL(filename='TestZ', displaysize=(800,600), radius=3, axis='z')

swc_neuron1.animationGL(filename='TestXPCA', displaysize=(800,600), radius=3, axis='x', pcaproject=True)
swc_neuron1.animationGL(filename='TestYPCA', displaysize=(800,600), radius=3, axis='y', pcaproject=True)
swc_neuron1.animationGL(filename='TestZPCA', displaysize=(800,600), radius=3, axis='z', pcaproject=True)
'''

'''
f = btmorph2.ForestStructure("C:/Users/Sam/Downloads/sam_forest_test_HN_P60/sam_forest_test_HN_P60/",(0,0,0))
f.plot_3DGL(multisample=True, fast=True)


p = btmorph2.PopulationMorphology("C:/Users/Sam/Downloads/sam_forest_test_HN_P60/sam_forest_test_HN_P60/")
print p.global_horton_strahler()
'''
#swc_neuron1 = btmorph2.NeuronMorphology('C:/Users/Sam/workspace/btmorph_v2/tests/v_e_moto1.CNG.swc')

#swc_neuron1.animationGL(filename='TestX', displaysize=(800,600), zoom=20, axis='x')
#swc_neuron1.plot_2D(depth='z')
#swc_neuron1.plot_3D()

#swc_neuron1.plot_3DGL()

#swc_neuron1 = btmorph2.NeuronMorphology('C:/Users/Sam/workspace/btmorph_v2/tests/v_e_moto1.CNG.swc', translate_origin=[500,500,500])
#swc_neuron1.plot_3DGL()


'''
swc_neuron1.plot_3DGL(pcaproject=True)

'''

#swc_neuron1 = btmorph2.NeuronMorphology('C:/Users/Sam/workspace/btmorph_v2/examples/new/041018-zA.CNG.swc')

#swc_neuron1 = btmorph2.NeuronMorphology('C:/Users/Sam/workspace/btmorph_v2/examples/data/1220882a.CNG.swc')
#swc_neuron1.plot_3D()
#swc_neuron1.animate()
'''
synapses = [False for i in range(swc_neuron1.no_nodes())]
synapses[4] = True

colors = []
r = []
g = []
for node in swc_neuron1.tree:
    #colors.append(swc_neuron1.order_of_node(node))
    r.append(swc_neuron1.order_of_node(node))
    g.append(swc_neuron1.degree_of_node(node))

rm = max(r)
r2 = [255-((255/rm) * x) for x in r]
gm = max(g)
g2 = [255-((255/gm) * x) for x in g]
for i in range(len(r2)):
    colors.append([r2[i],g2[i],0,255])


swc_neuron1.plot_3D(color_mapping=colors)
swc_neuron1.animate(color_mapping=colors, save_image="Test")
swc_neuron1.plot_2D(color_mapping=colors)
'''
'''swc_neuron1.plot_2D(color_mapping=colors, synapses=synapses)
swc_neuron1.plot_2D()
'''
#swc_neuron1.plot_3DGL()

'''
swc_neuron1 = btmorph2.NeuronMorphology('C:/Users/Sam/workspace/btmorph_v2/examples/new/041018-zA.CNG.swc', pca_translate=True )
swc_neuron1.plot_3DGL()
'''

#swc_neuron1 = btmorph2.NeuronMorphology('C:/Users/Sam/workspace/btmorph_v2/examples/new/2005-01-25-A1.CNG.swc')
#swc_neuron1.plot_3DGL()


#swc_neuron1 = btmorph2.NeuronMorphology('C:/Users/Sam/workspace/btmorph_v2/examples/new/cell_0.swc', translate_origin=[0,0,0])

# swc_neuron1.plot_3D(synapses=[True] * len(swc_neuron1.get_tree().get_nodes()))
#swc_neuron1.plot_3DGL()



#swc_neuron1 = btmorph2.NeuronMorphology('C:/Users/Sam/workspace/btmorph_v2/examples/new/PRC2080328I.CNG.swc')
#swc_neuron1.plot_3DGL()





#print swc_neuron1.global_horton_strahler()
