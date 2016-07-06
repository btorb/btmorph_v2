.. raw:: html

    <style> .red {color:white;background-color:red} </style>
    <style> .green {color:white;background-color:green} </style>


#####################
Validation & testing
#####################

.. _comparison:

Comparison with L-Measure
--------------------------

Btmorph is compared to the "golden standard" L-Measure through the publicly available data at `NeuroMorpo.org <NeuroMorpho.org>`_.

In most cases the results obtained with the btmorph library are similar; there are some slight differences that reflect slight implementation details and some measures are interpreted differently; implementation details of L-Measure can be found `(here) <http://cng.gmu.edu:8080/Lm/help/index.htm>`_ and the meaning of the morphometrics displayed on NeuroMorpho.org are explained `here <http://neuromorpho.org/neuroMorpho/myfaq.jsp>`_.

We explain the similarities and differences by means of an exemplar analysis performed on one
morphology: `v_e_moto1` `(from here) <http://neuromorpho.org/neuroMorpho/neuron_info.jsp?neuron_name=v_e_moto1>`_. 


.. role:: red
.. role:: green


.. tabularcolumns:: |l|l|p{5cm}|

+---------------------+-----------------+---------------------------+
|Morphometric feature | NeuroMorpho.org | btmorph                   |
+=====================+=================+===========================+
| Soma Surface        | 45216 μm2       | :red:`45238` μm2 [#f0]_   |
+---------------------+-----------------+---------------------------+
| # Stems             | 10              | :green:`10`               |
+---------------------+-----------------+---------------------------+
| # Bifurcations      | 122             | :green:`122`              |
+---------------------+-----------------+---------------------------+
| # Branches          | 254             | :green:`254` [#f1]_       |
+---------------------+-----------------+---------------------------+
| Overall Width       |  1804.67 μm     | 2588.0 μm [#f2]_          |
+---------------------+-----------------+---------------------------+
| Overall Height      |  2259.98 μm     | 2089.0 μm [#f2]_          |
+---------------------+-----------------+---------------------------+
| Overall Depth       |  1701.72 μm     | 2306.0 μm [#f2]_          |
+---------------------+-----------------+---------------------------+
| Average Diameter    |  2.2 μm         | :green:`2.2` μm [#f3]_    |
+---------------------+-----------------+---------------------------+
| Total Length        |  78849.1 μm     | :green:`78849.1` μm       |
+---------------------+-----------------+---------------------------+
| Total Surface       |  512417 μm2     | :green:`512417` μm2       |
+---------------------+-----------------+---------------------------+
| Total Volume        |  390413 μm3     | :green:`390412` μm3       |
+---------------------+-----------------+---------------------------+
| Max Euclidean       |                 |                           |
| Distance            | 765.73 μm       | :red:`1531 μm` [#f4]_     |
+---------------------+-----------------+---------------------------+
| Max Path Distance   | 873.56 μm       | :red:`1817` μm [#f5]_     |
+---------------------+-----------------+---------------------------+
| Max Branch Order    | 3.69            | :green:`3.83` [#f6]_      |
+---------------------+-----------------+---------------------------+
| Average Contraction | 0.94            | :green:`0.9359` [#f7]_    |
+---------------------+-----------------+---------------------------+
| Total Fragmentation | 559             | :green:`599` [#f8]_       |
+---------------------+-----------------+---------------------------+
| Partition Asymmetry | 0.43            | :green:`0.43` [#f9]_      |
+---------------------+-----------------+---------------------------+
| Average Rall's      |                 |                           |
| Ratio               |1.25             | :green:`1.25`             |
+---------------------+-----------------+---------------------------+
| Average Bifurcation |                 |                           |
| Angle Local         | 46.83°          | :green:`46.83°`           |
+---------------------+-----------------+---------------------------+
| Average Bifurcation |                 |                           |
| Angle Remote        |  45.74°         | :green:`45.7 °`           |
+---------------------+-----------------+---------------------------+

.. [#f0] In accordance with the three-point soma format, the somatic surface is computed as :math:`A = 4 \times \pi \times r^2`.
.. [#f1] Computed by `neuron1.no_bifurcations() + neuron.no_terminals()`
.. [#f2] We compute the raw, untranslated extend in X,Y and Z dimension. This is different from aligning the axis with the first three principal components and including 95% of the data as is done in L-Measure and NeuroMorpho.org.
.. [#f3] Computed by `np.mean(neuron1.get_diameters())`
.. [#f4] Unclear how the NeuroMorpho.org value is generated. We compute the euclidean distance between each terminal point and the soma. A visual inspection shows that our value is correct.

.. [#f5] See [#f4]_
.. [#f6] This is actually not the maximum as listed on the NeuroMorpho website but the average of either all points, or the bifurcation points.
.. [#f7] Computed as follows: 
:: 

   eds = []
   pls = []
   for node in neuron1._end_points:
       eds.append(neuron.get_segment_Euclidean_length(node))
       pls.append(neuron1.get_segment_pathlength(node))
   mean(array(eds)/array(pls))

.. [#f8] The fragmentation can be computed by :code:`len(neuron1._all_nodes)-3`, where 3 is subtracted to discount the three soma points.

.. [#f9] Computed as follows:
::

   pas = []
   for node in neuron1._bif_points:
       pas.append(neuron1.partition_asymmetry(node))
   mean(pas)

.. _unit_testing:

Unit testing
------------

Unit-testing refers to testing of elementary pieces of code in a computer program `(Wikipedia) <http://en.wikipedia.org/wiki/Unit_testing>`_. Testing is done using the Python testing framework, called nose tests. In these tests, we compare the outcome of our library to similar outcomes generated by L-Measure that are accessible through the `NeuroMorpho.org <www.neuromorpho.org>`_ website. Note that there are some differences in design and definition of the features as listed :ref:`comparison`.

Unit-tests of this library are provided in the ``tests`` directory and can be run by
::

    nosetests -v --nocapture tests/structs_test.py

.. note:: Run the unit-tests after change to the code to ensure a) backward compatibility and b) correctness of the results.

