################
Tutorial
################

This is a brief, hands-on tutorial explaining how to use btmorph to load SWC files, analyse them by computing morphometric measures and compare morphologies to one another. A short description is also provided on how to use btmorph in further scripting.

We recommend to use IPython. In a terminal type, change to the :code:`examples` directory and type :code:`ipython --pylab -i`. Then, you can either type the command in the listings below or directly copy them. Copy and pasting of the code snippets can be done by copying them and typing into the ipython prompt :code:`%paste` `(Magic functions) <http://ipython.org/ipython-doc/rel-1.1.0/interactive/tutorial.html>`_.

.. note:: Make use of "magic functions" in IPython. Copy the code from this pages and type ``%paste`` in the IPython session. All code will be pasted with correct layout and directly executed.

Analyzing morphometric data
---------------------------

This tutorial assumes you are in the :code:`examples` directory of the btmorph package (ath the top level, issue :code:`cd examples`). Several exemplar SWC files are contained in the package [#f1]_.
::
   
  import btmorph2
  import numpy
  import matplotlib.pyplot as plt

  neuron1 = btmorph2.NeuronMorphology("data/v_e_moto1.CNG.swc")

  """ get the total length, a scalar morphometric feature """
  total_length = neuron1.total_length()
  print 'Total neurite length=%f' % total_length

  """ get the topological measure of leaf node in the tree"""
  no_terminals = neuron1.no_terminals()
  print 'Number of terminals=%f' % no_terminals


In case you do this tutorial in one Ipython session (:code:`Ipython --pylab`), you don't need to load all libraries all the time and you can just continue to copy and paste code into the prompt. Therefore, loading of the library is omitted in the code listing below.

Now probe a vector morphometric, for instance the segment length. Clearly, summing all segments lengths together should give us the total segment length as before.
::

  bif_nodes = neuron1._bif_points
  term_nodes = neuron1._end_points
  all_nodes = bif_nodes + term_nodes
  total_length = 0
  all_segment_lengths = []
  for node in all_nodes :
      all_segment_lengths.append( neuron1.get_segment_pathlength(node)  )
      total_length = total_length + all_segment_lengths[-1]
  print 'total_length=', total_length

Now you can plot a histogram of the segment length distribution::

  plt.hist(all_segment_lengths)
  plt.xlabel('Segment length (micron)')
  plt.ylabel('count')

which should produce an image as illustrated below:

.. image:: figures/v_e_moto1_segment_lengths.png
  :scale: 50

As a slightly more complicated example, we can also check the path length and Euclidean distance at which bifurcations occur. Plotting the number of bifurcations as a function of euclidean distance is roughly the same as the *Sholl analysis*. ::

  bif_path_lengths = []
  bif_euclidean_lengths = []
  bif_contractions = []
  for node in neuron1._bif_points :
      bif_path_lengths.append(neuron1.get_pathlength_to_root(node))
      bif_euclidean_lengths.append(neuron1.get_Euclidean_length_to_root(node))
      bif_contractions.append( bif_euclidean_lengths[-1] / bif_path_lengths[-1]  )
  plt.hist(bif_euclidean_lengths)
  plt.title('(Almost) Sholl analysis')
  plt.xlabel('euclidean distance (micron)')
  plt.ylabel('count / crossings')

which produces the following image:

.. image:: figures/v_e_moto1_sholl.png
  :scale: 50

Clearly, in the above figure we can distinguish the bimodal distribution introduced by some the basal and oblique dendrites on the one hand, and the distal apical dendrites on the other.

Finally, to visually inspect both morphologies we could plot them::

  plt.figure()
  neuron1.plot_2D()
  plt.figure()
  neuron1.plot_dendrogram()

.. |2D| image:: figures/v_e_moto1_2D.png
  :scale: 50

.. |dendro| image:: figures/v_e_moto1_dendrogram.png
  :scale: 50

+---------+-----------+
| |2D|    | |dendro|  |
+---------+-----------+

Potential extensions
--------------------

There are also hooks in :code:`btmorph2` to access other features. NeuronMorphology objects provide direct access to the tree data structure. In case of a NeuronMorphology object ``n``, the following hooks exist.

- ``n._all_nodes``: list with all nodes in the tree
- ``n._bif_points``: list with bifurcating nodes in the tree
- ``n._end_points``: list with terminal (=leaf) nodes in the tree
- ``n._tree``: Tree structure. Can be used to compute various graph-theoretical features.

For instance, it is straight-forward to save a cloud on which measurement related to the spatial distribution of points (for instance, the moments of the point cloud) can be measured.::

  bx,by,bz = [],[],[]
  for node in neuron1._bif_points :
      n = node.get_content()['p3d']
      bx.append(n.xyz[0])
      by.append(n.xyz[1])
      bz.append(n.xyz[2])
  bif_cloud = [bx,by,bz]
  # save as txt...
  np.savetxt('bif_cloud.txt',bif_cloud) 
  #... or as pickle
  import pickle
  pickle.dump(bif_cloud,open('bif_cloud.pkl','w'))

Note that in this example only bifurcation points are considered. Through the ``neuron1.tree.get_nodes()`` or ``neuron1._all_points`` all points can be retrieved.

The cloud data can now be loaded and plotted (and serve for further analysis)
::

  import pickle
  bc = pickle.load(open('bif_cloud.pkl'))
  for i in range(len(bc[0])) :
      plt.plot(bc[0][i],bc[1][i],'ro')

  # or
  plt.scatter(bx,by)

.. image:: figures/v_e_moto1_bifcloud.png
  :scale: 50


Comparison of morphologies
--------------------------

Validation of morphologies boils down -in the simplest one-dimensional case and in a statistical sense- to the comparison of data vectors. The idea is visually illustrated below. The method outlined here can be easily extended to conditional data, that is, N-dimensional data capturing relations between data point using adequate statistical tools.


One-to-one validation
~~~~~~~~~~~~~~~~~~~~~

Two neurons are compared to each other. On a one to one basis there is little statistical ground to compare the scalar properties with each other. However, the vector features (for instance, segment lengths) can be compared. In this example we do the fairly senseless thing of showing the difference between a hippocampal granule cell and a spinal cord motor neuron (used before).
::

  import btmorph2
  import numpy
  import matplotlib.pyplot as plt

  v1_tree = btmorph2.NeuronMorphology("data/v_e_moto1.CNG.swc")

  granule_tree = btmorph2.NeuronMorphology("data/1220882a.CNG.swc")

  v1_bif_nodes = v1_tree._bif_points
  granule_bif_nodes = granule_tree._bif_points

  v1_bif_segment_lengths = []
  granule_bif_segment_lengths = []
  
  for node in v1_bif_nodes:
      v1_bif_segment_lengths.append( v1_tree.get_segment_pathlength(node)  )
  for node in granule_bif_nodes:
      granule_bif_segment_lengths.append( granule_tree.get_segment_pathlength(node)  )

And compare the two vectors (visually and by performing the Kruskal-Wallis H-test):
::

  import scipy
  import scipy.stats
  hist(v1_bif_segment_lengths,color='r',alpha=0.5,label="v_e_moto1")
  hist(granule_bif_segment_lengths,color='b',alpha=0.5,label="granule")
  legend(loc=0)
  res = scipy.stats.ks_2samp(v1_bif_segment_lengths,granule_bif_segment_lengths)
  print 'K-S=%f, p_value=%f' % (res[0], res[1])

A figure will be generated and the output will appear: ``K-S=0.609631, p_value=0.000023``

.. image:: figures/compare_segments.png
  :scale: 50

According to the `manual <http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html#scipy.stats.kruskal>`_: "if the K-S statistic is small or the p-value is high, then we cannot reject the hypothesis that the distributions of the two samples are the same."


Many-to-many validation
~~~~~~~~~~~~~~~~~~~~~~~

The comparison of two population can be done in exactly the same way as described above. The scalar properties of each neuron in the population make up a vector of values. Hence, the vector of one population can be compared against the vector associated with another population. In the case of vector features, all features can be appended to one vector per population.


Working with populations
------------------------

New in btmorph v2 is the concept of a population, a set of NeuronMorphology objects.
Currently, only a limited set of morphometrics is directly implemented for the population,
but we hope to add more soon. Also, specialized plotting functions for
populations will be developed.
::
   import btmorph2
   pop = btmorph2.PopulationMorphology("data/population/")
   Ls = pop.total_length()
   plt.hist(Ls)
   
To investigate or plot just a single neuron from the population:
::
   n1 = pop.neurons[1]
   n1.plot_2D() # for instance

..
   Wrappers for btmorph
   --------------------

   We provide basic wrappers that perform standard, of-the-shelf analysis of neurons. Two wrappers are available.

   - ``btmorph.perform_2D_analysis``. Collects morphometric features of birufcatiuon and terminal points and stores the results in files. For each of these points the path length to the soma, euclidean distance from the soma, degree, order, partition asymmetry and segment length are recorded. Hence, one can correlate, for instance, the segment length with the centrifugal order (= two-dimensional). Higher order correlation can be used at will as well. (See API)

   - ``btmorph.perform_1D_population_analysis``. Collects all morphometric features of one population in vectors and writes the result to files. (see API)


References

.. [#f1] v_e_moto1 is downloaded from `here <http://neuromorpho.org/neuroMorpho/neuron_info.jsp?neuron_name=v_e_moto1>`_ and originates from a study linked on `pubmed <http://www.ncbi.nlm.nih.gov/pubmed/3819010>`_.
