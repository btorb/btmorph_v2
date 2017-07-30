from btmorph2 import PopulationMorphology
import os
import filecmp
from btmorph2.transforms import compose_matrix
import numpy as np

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

def affineTransform_test():
    """
    Testing affine transform function of Population Morphology
    """

    affineTransformMatrix = compose_matrix(scale=(0.8, 1.2, 0.95),
                                           translate=(-10, 5, 4),
                                           angles=np.deg2rad((15, -6, -30)))
    testFile = "tests/transTest/22_117.v3dpbd_Rayshooting.swc"
    expectedOP = "tests/transTest/22_117.v3dpbd_Rayshooting_trans.swc"

    pm = PopulationMorphology(testFile, correctIfSomaAbsent=True)
    newPM = pm.affineTransform(affineTransformMatrix)

    testOP = "tests/transTest/22_117.v3dpbd_Rayshooting_testOutput.swc"
    if os.path.isfile(testOP):
        os.remove(testOP)
    newPM.write_to_SWC_file(testOP)
    assert filecmp.cmp(testOP, expectedOP)

def emptyPM_test():
    """
    Testing initializing an empty PopulationMorphology
    :return: 
    """
    pm = PopulationMorphology()
    assert len(pm.neurons) == 0



if __name__ == "__main__":

    # dummyImport()
    # multiTreeSWCImport_test()
    # multiTreeSWC_write()
    # multiTreeSWC_write_read_test()
    affineTransform_test()
