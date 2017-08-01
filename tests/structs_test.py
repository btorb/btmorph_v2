'''
Test routines for the btstructs.py file
'''
from __future__ import division
from __future__ import print_function

from past.utils import old_div
from btmorph2 import NeuronMorphology
import numpy as np
from nose.tools import with_setup
import os


def test_soma_type_3ps():
    """
    Test if SWC 3-point soma  description is correctly recognized
    """
    swc_neuron1 = NeuronMorphology("tests/v_e_moto1.CNG.swc")
    assert(swc_neuron1.tree.soma_type == 1)

def test_soma_type_1ps():
    """
    Test if SWC 1-point soma  description is correctly recognized: v_e_purk2.CNG.swc
    """
    swc_neuron1 = NeuronMorphology("tests/v_e_purk2.CNG.swc")
    assert(swc_neuron1.tree.soma_type == 0)


def test_soma_type_1ps():
    """
    Test if SWC 1-point soma description is correctedly recognized: 1220882a.CNG.swc
    """
    swc_neuron1 = NeuronMorphology("tests/soma_types/1220882a.CNG.swc")
    assert(swc_neuron1.tree.soma_type == 0)

def test_soma_type_mc():
    """
    Test if SWC multiple cylinder soma  description is correctly recognized
    """
    swc_neuron1 = NeuronMorphology("tests/soma_types/l22.CNG.swc")
    assert(swc_neuron1.tree.soma_type == 2)


def test_NeuronMorphology_multipleTrees():
    """
    Test if initializing a NeuronMorphology object with an SWC file with multiple trees raises and error
    """

    input_file = "tests/117.v3dpbd/08_117.v3dpbd_neutube_updated.swc"

    try:
        nrn = NeuronMorphology(input_file)
    except ValueError as ve:
        assert str(ve) == "Given SWC File {} has more than one trees".format(input_file)


def test_empty_SWC():
    """
    Test if passing empty SWC file raises the appropriate error
    :return: 
    """

    testFile = "tests/empty.swc"

    try:
        nrn = NeuronMorphology(testFile)
    except ValueError as ve:
        assert str(ve) == "Improper data in SWC file {}".format(testFile)

def test_nonUTF8_SWC():
    """
    Test parsing SWC file with non utf-8 encoded chars
    :return: 
    """

    testFile = "tests/nonUTF8.swc"


    nrn = NeuronMorphology(testFile, correctIfSomaAbsent=True, ignore_type=True)

    assert len(nrn.tree.get_nodes()) == 18




def test_load_swc():
    '''
    Test whether SWC files are correctly loaded
    '''
    swc_neuron1 = NeuronMorphology('tests/v_e_moto1.CNG.swc')
    all_nodes1 = swc_neuron1.tree.get_nodes()

    print(('\n len(swc_neuron1)', len(all_nodes1)))

    assert(len(all_nodes1) == 562)


def test_load_and_write_swc():
    '''
    Test whether SWC trees are correctly written to file
    '''

    expectedOutput = 'tests/moto_1_outputted.swc'
    if os.path.isfile(expectedOutput):
        os.remove(expectedOutput)
    swc_neuron1 = NeuronMorphology('tests/v_e_moto1.CNG.swc')
    swc_neuron1.tree.write_SWC_tree_to_file(expectedOutput)
    swc_neuron2 = NeuronMorphology('tests/moto_1_outputted.swc')
    print(('len(swc_neuron2)', len(swc_neuron2.tree.get_nodes())))

    assert(len(swc_neuron2.tree.get_nodes()) == 562)


def test_load_swc_mcs1():
    '''
    Test whether SWC files with multiple-cylinder soma format are read correctly (l22)
    '''
    swc_neuron1 = NeuronMorphology('tests/soma_types/l22.CNG.swc')
    all_nodes1 = swc_neuron1.tree.get_nodes()
    print(('\nlen(swc_neuron1)', len(all_nodes1)))
    assert(len(all_nodes1) == 1595)
    assert(1416 < swc_neuron1.approx_soma() < 1417)


def test_load_swc_mcs2():
    '''
    Test whether SWC files with multiple-cylinder soma format are read correctly (ri05)
    '''
    swc_neuron1 = NeuronMorphology('tests/soma_types/ri05.CNG.swc')
    all_nodes1 = swc_neuron1.tree.get_nodes()
    print(('\nlen(swc_neuron1)', len(all_nodes1)))
    assert(len(all_nodes1) == 8970)
    assert(503 < swc_neuron1.approx_soma() < 504)


def test_global_bifurcations():
    """
    Bifurcation count
    """
    swc_neuron1 = NeuronMorphology('tests/v_e_moto1.CNG.swc')
    no_bifurcations = swc_neuron1.no_bifurcations()
    print(('no_bifurcations=%f' % no_bifurcations))
    assert(no_bifurcations == 122)


def test_global_terminals():
    """
    Terminal point count
    """
    swc_neuron1 = NeuronMorphology('tests/v_e_moto1.CNG.swc')
    no_terminals = swc_neuron1.no_terminals()
    print(('no_terminals=%f' % no_terminals))
    assert(no_terminals == 132)


def test_global_stems():
    """
    Stem count
    """
    swc_neuron1 = NeuronMorphology('tests/v_e_moto1.CNG.swc')
    no_stems = swc_neuron1.no_stems()
    print(('no_stems=%s' % no_stems))
    assert(no_stems == 10)


def test_global_totallength():
    """
    Total length count
    """
    swc_neuron1 = NeuronMorphology('tests/v_e_moto1.CNG.swc')
    total_length = swc_neuron1.total_length()
    print(('total length=%f' % (total_length)))
    assert(78849 < total_length < 78850)


def test_global_somasurface():
    """
    Soma surface
    """
    swc_neuron1 = NeuronMorphology('tests/v_e_moto1.CNG.swc')
    soma_surface = swc_neuron1.approx_soma()
    print(('soma surface=%f' % soma_surface))
    assert(45238 < soma_surface < 45239)


def test_segment_length():
    """
    Compute total length as sum of incoming segment lengths
    """
    swc_neuron1 = NeuronMorphology('tests/v_e_moto1.CNG.swc')
    bif_nodes = swc_neuron1._bif_points
    term_nodes = swc_neuron1._end_points
    all_nodes = bif_nodes + term_nodes
    total_length = 0
    for node in all_nodes:
        total_length = total_length + swc_neuron1.get_segment_pathlength(node)
    print(('total_length=', total_length))
    assert(78849 < total_length < 78850)


def test_terminal_lengths():
    """
    Check terminal point lengths
    """
    swc_neuron1 = NeuronMorphology('tests/v_e_moto1.CNG.swc')
    term_path_lengths = []
    term_euclidean_lengths = []
    term_contractions = []
    for node in swc_neuron1._end_points:
        term_path_lengths.append(swc_neuron1.get_pathlength_to_root(node))
        term_euclidean_lengths.append(swc_neuron1.get_Euclidean_length_to_root(node))
        term_contractions.append(old_div(term_euclidean_lengths[-1],
                                 term_path_lengths[-1]))
    print(('min/max path: %f - %f' % (min(term_path_lengths),
                                     max(term_path_lengths))))
    print(('min/max euclid: %f - %f' % (min(term_euclidean_lengths),
                                       max(term_euclidean_lengths))))
    print(('min/max contraction: %f - %f' % (min(term_contractions),
                                            max(term_contractions))))
    assert(1531 < max(term_euclidean_lengths) < 1532)
    assert(1817 < max(term_path_lengths) < 1819)


def test_degree():
    """
    Topological degree
    """
    swc_neuron1 = NeuronMorphology('tests/v_e_moto1.CNG.swc')
    max_degree = swc_neuron1.degree_of_node(swc_neuron1.tree.root)
    print(('max_degree = ', max_degree))
    assert(max_degree == 134)


def test_order():
    """
    Topological order
    """
    swc_neuron1 = NeuronMorphology('tests/v_e_moto1.CNG.swc')
    max_order = -1
    min_order = 100000
    for node in swc_neuron1.tree.get_nodes():  # skip the root
        order = swc_neuron1.tree.order_of_node(node)
        if order > max_order:
            max_order = order
        if order < min_order:
            min_order = order
    print(('min_order=%f, max_order=%f' % (min_order, max_order)))
    assert(max_order == 9)


def test_partition_asymmetry():
    """
    Parition asymmetry
    """
    swc_neuron1 = NeuronMorphology('tests/v_e_moto1.CNG.swc')
    pa = []
    for node in swc_neuron1._bif_points:
        pa.append(swc_neuron1.partition_asymmetry(node))
    avg_pa = np.mean(pa)
    max_pa = max(pa)
    min_pa = min(pa)
    print(('avg_pa=%f, min_pa=%f, max_pa=%f' % (avg_pa, min_pa, max_pa)))
    assert(0.43 < avg_pa < 0.45)


def test_surface():
    """
    Total neurite surface
    """
    swc_neuron1 = NeuronMorphology('tests/v_e_moto1.CNG.swc')
    total_surf = swc_neuron1.total_surface()
    print(('total_surf= %f' % (total_surf[0])))
    assert(512417 < total_surf[0] < 512419)


def test_volume():
    """
    Total neurite volume
    """
    swc_neuron1 = NeuronMorphology('tests/v_e_moto1.CNG.swc')
    total_vol = swc_neuron1.total_volume()
    print(('total_volume = %f' % (total_vol[0])))
    assert(390412 < total_vol[0] < 390414)


def ttest_bifurcation_sibling_ratio_local():
    swc_neuron1 = NeuronMorphology('tests/v_e_moto1.CNG.swc')
    ratios = []
    for node in swc_neuron1._bif_points:
        ratio = swc_neuron1.bifurcation_sibling_ratio(node, where='local')
        ratios.append(ratio)
    print(('mean(ratios_local)=', np.mean(ratios)))
    assert(1.31 < np.mean(ratios) < 1.32)


def ttest_bifurcation_sibling_ratio_remote():
    swc_neuron1 = NeuronMorphology('tests/v_e_moto1.CNG.swc')
    ratios = []
    for node in swc_neuron1._bif_points:
        ratio = swc_neuron1.bifurcation_sibling_ratio(node, where='remote')
        ratios.append(ratio)
    print(('mean(ratios_remote)=', np.mean(ratios)))
    assert(1.16 < np.mean(ratios) < 1.17)


def test_bifurcation_amplitude_local():
    swc_neuron1 = NeuronMorphology('tests/v_e_moto1.CNG.swc')
    all_ampl = []
    for node in swc_neuron1._bif_points:
        ampl = swc_neuron1.bifurcation_angle_vec(node, where='local')
        all_ampl.append(ampl)
    print(('min=%f max(ample)=%f, mean(ampl)=%f' % (np.min(all_ampl),
                                                   np.max(all_ampl),
                                                   np.mean(all_ampl))))
    assert(46.8 < np.mean(all_ampl) < 46.9)


def test_bifurcation_amplitude_remote():
    swc_neuron1 = NeuronMorphology('tests/v_e_moto1.CNG.swc')
    all_ampl = []
    for node in swc_neuron1._bif_points:
        ampl = swc_neuron1.bifurcation_angle_vec(node, where='remote')
        all_ampl.append(ampl)
    print(('min=%f max(ample)=%f, mean(ampl)=%f' % (np.min(all_ampl),
                                                   np.max(all_ampl),
                                                   np.mean(all_ampl))))
    assert(45.7 < np.mean(all_ampl) < 45.8)


def test_ralls_power_brute():
    swc_neuron1 = NeuronMorphology('tests/v_e_moto1.CNG.swc')
    all_n = []
    for node in swc_neuron1._bif_points:
        n = swc_neuron1.bifurcation_ralls_power_brute(node)
        if n is None or n is -1:
            pass
        else:
            # print "N: ", n
            all_n.append(n)
    print(('min_p=%f,avg_p=%f media=%f, max_p=%f' % (np.min(all_n),
                                                    np.mean(all_n),
                                                    np.median(all_n),
                                                    np.max(all_n))))
    assert(1.77 <= np.mean(all_n) < 1.80)


def test_ralls_power_fmin():
    """
    scipy.optimize.fminsearch for rall's power
    """
    swc_neuron1 = NeuronMorphology('tests/v_e_moto1.CNG.swc')
    all_p = []
    for node in swc_neuron1._bif_points:
        p = swc_neuron1.bifurcation_ralls_power_fmin(node)
        all_p.append(p)
        # print node, '-> p=', p
    all_p = np.array(all_p)
    all_pp = []
    for n in all_p:
        if not np.isnan(n):
            all_pp.append(n)
    print(('min_p=%f,avg_p=%f media=%f, max_p=%f' % (np.min(all_pp),
                                                    np.mean(all_pp),
                                                    np.median(all_pp),
                                                    np.max(all_pp))))
    # p = stats.bifurcation_ralls_ratio(stats._bif_points[1])
    avg_rr = np.mean(all_pp)
    assert(1.68 < avg_rr < 1.70)


def test_ralls_ratio_classic():
    swc_neuron1 = NeuronMorphology('tests/v_e_moto1.CNG.swc')
    all_n = []
    for node in swc_neuron1._bif_points:
        n = swc_neuron1.bifurcation_rall_ratio_classic(node, where='local')
        if n is None or n is -1:
            pass
        else:
            # print "N: ", n
            all_n.append(n)
    print(('min_p=%f,avg_p=%f media=%f, max_p=%f' % (np.min(all_n),
                                                    np.mean(all_n),
                                                    np.median(all_n),
                                                    np.max(all_n))))
    assert(1.25 <= np.mean(all_n) < 1.26)

""" New fucntions by Irina - test"""
test_neurons = []


def setup_func_small_tree():
    """
    Setup function for Horton-Strahler number testing
    """
    global test_neurons
    # 0 - Only soma tree
    test_neurons.append(NeuronMorphology("tests/soma_only.swc"))
    # 1 - Wiki test tree
    test_neurons.append(NeuronMorphology("tests/horton-strahler_" +
                                       "test_wiki_3pointsoma.swc"))


def teardown_func_small_tree():
    """
    Teardown function for Horton-Strahler number testing
    """
    global test_neurons
    test_neurons = []


@with_setup(setup_func_small_tree, teardown_func_small_tree)
def test_local_horton_strahler():
    swc_neuron1 = NeuronMorphology('tests/v_e_moto1.CNG.swc')

    global test_neurons
    # Trivial cases
    assert(-1 == swc_neuron1.local_horton_strahler(None))
    # Real test
    all_nodes = test_neurons[1].tree.get_nodes()
    for node in all_nodes:
        r = int(node.content['p3d'].radius)
        assert(r == swc_neuron1.local_horton_strahler(node))
    pass


@with_setup(setup_func_small_tree, teardown_func_small_tree)
def test_global_horton_strahler():
    global test_neurons
    assert(4 == test_neurons[1].global_horton_strahler())
    pass


def setup_func_small_tree_lac():
    """
    Setup function for tree initialization and loading
    """
    global test_neurons
    # 0 - Only soma tree
    # test_trees.append(btmorph.STree2().read_SWC_tree_from_file("tests/soma_only.swc"))
    # 1 - Wiki test tree moto_1_outputted
    # test_trees.append(btmorph.STree2().read_SWC_tree_from_file("tests/horton-strahler_test_wiki_3pointsoma.swc"))
    test_neurons.append(NeuronMorphology("tests/moto_1_outputted.swc"))


def teardown_func_small_tree_lac():
    """
    Teardown function for tree initialization and loading
    """
    global test_neurons
    test_neurons = []


def breadth_first_iterator_test():
    """
    Testing Breadth first iterator generator
    """

    swcFile = "tests/horton-strahler_test_wiki.swc"
    nrn = NeuronMorphology(swcFile)
    bfsInds = [node.index for node in nrn.tree.breadth_first_iterator_generator()]
    assert bfsInds == [1, 2, 21, 22, 3, 4, 5, 12, 6, 9, 13, 14, 7, 8, 10, 11, 15, 18, 16, 17, 19, 20]


def depth_first_iterator_test():
    """
    Testing Depth first iterator generator
    """

    swcFile = "tests/horton-strahler_test_wiki.swc"
    nrn = NeuronMorphology(swcFile)
    dfsInds = [node.index for node in nrn.tree.depth_first_iterator_generator()]
    assert dfsInds == [1, 22, 21, 2, 4, 12, 14, 18, 20, 19, 15, 17, 16, 13, 5, 9, 11, 10, 6, 8, 7, 3]


if __name__ == "__main__":
    # test_NeuronMorphology_multipleTrees()
    test_empty_SWC()