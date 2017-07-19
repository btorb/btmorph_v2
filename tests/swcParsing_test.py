from btmorph2.SWCParsing import SWCParsing

def testCC():
    """
    Test if connected components are being detected properly
    """

    swcFile = "tests/117.v3dpbd/03_117.v3dpbd_NeuroGPSTree.swc"
    temp = SWCParsing(swcFile)
    assert temp.numberOfTrees() == len(temp.checkAndReturnFeasibleGraphsWithTypeLineNumber())

def test_soma_type_3ps():
    """
    Test if SWC 3-point soma  description is correctly recognized
    """
    temp = SWCParsing("tests/v_e_moto1.CNG.swc")
    swcTree = temp.checkAndReturnFeasibleGraphsWithTypeLineNumber()[0]
    soma_type = SWCParsing.determine_soma_type(swcTree)
    assert(soma_type == 1)


def test_soma_type_1ps1():
    """
    Test if SWC 1-point soma  description is correctly recognized
    """
    temp = SWCParsing("tests/v_e_purk2.CNG.swc")
    swcTree = temp.checkAndReturnFeasibleGraphsWithTypeLineNumber()[0]
    soma_type = SWCParsing.determine_soma_type(swcTree)
    assert(soma_type == 0)


def test_soma_type_1ps2():
    """
    Test if SWC 1-point soma description is correctedly recognized
    """
    temp = SWCParsing("tests/soma_types/1220882a.CNG.swc")
    swcTree = temp.checkAndReturnFeasibleGraphsWithTypeLineNumber()[0]
    soma_type = SWCParsing.determine_soma_type(swcTree)
    assert(soma_type == 0)


def test_soma_type_mc():
    """
    Test if SWC multiple cylinder soma  description is correctly recognized
    """
    temp = SWCParsing("tests/soma_types/l22.CNG.swc")
    swcTree = temp.checkAndReturnFeasibleGraphsWithTypeLineNumber()[0]
    soma_type = SWCParsing.determine_soma_type(swcTree)
    assert(soma_type == 2)

if __name__ == "__main__":
    test_soma_type_1ps1()
    test_soma_type_1ps2()
    test_soma_type_3ps()
    test_soma_type_mc()