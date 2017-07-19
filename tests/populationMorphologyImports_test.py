from btmorph2 import PopulationMorphology
import tempfile
import os
import filecmp

def dummyImport():

    inDir = "tests/117.v3dpbd/06_117.v3dpbd_MOST.swc"
    pm = PopulationMorphology(inDir, correctIfSomaAbsent=True)
    pass

def multiTreeSWCImport_test():

    """
    Test the initialization of PopulationMorphology with an SWC file with multiple trees
    """

    swc = "tests/117.v3dpbd/03_117.v3dpbd_NeuroGPSTree.swc"

    pm = PopulationMorphology(swc, correctIfSomaAbsent=True)
    assert len(pm.neurons) == 260

def multiTreeSWC_write_test():
    """
    Test write function of PopulationMorphology
    """

    inDir = "tests/popMorphTest"
    expectedFile = os.path.join("tests/popMorphTestResult", "expectedOP.swc")
    opFile = os.path.join("tests/popMorphTestResult", "test.swc")
    if os.path.isfile(opFile):
        os.remove(opFile)
    pm = PopulationMorphology(inDir, correctIfSomaAbsent=True)
    assert len(pm.neurons) == 227
    pm.write_to_SWC_file(opFile)
    assert filecmp.cmp(opFile, expectedFile, shallow=False)

def multiTreeSWC_write_read_test():
    """
    Test write and read function of PopulationMorphology
    """
    inDir = "tests/popMorphTest"
    opFile = os.path.join("tests/popMorphTestResult", "test.swc")
    if os.path.isfile(opFile):
        os.remove(opFile)
    pm = PopulationMorphology(inDir, correctIfSomaAbsent=True)
    assert len(pm.neurons) == 227
    pm.write_to_SWC_file(opFile)
    pm1 = PopulationMorphology(opFile)
    assert len(pm1.neurons) == 227
    nodesCount1 = [len(x.tree.get_nodes()) for x in pm1.neurons]
    nodesCount = [len(x.tree.get_nodes()) for x in pm.neurons]

    assert sorted(nodesCount) == sorted(nodesCount1)

if __name__ == "__main__":

    # dummyImport()
    # multiTreeSWCImport_test()
    # multiTreeSWC_write()
    multiTreeSWC_write_read_test()

