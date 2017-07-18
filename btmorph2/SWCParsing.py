import numpy as np
from auxFuncs import readSWC_numpy, getDuplicates
import networkx as nx



class SWCParsing(object):
    """
    Class to assist in SWC file parsing
    """

    def __init__(self, swcFile):

        self.headr, self.swcData = readSWC_numpy(swcFile)
        self.swcFile = swcFile

    def numberOfTrees(self):

        parents = self.swcData[:, 6]
        return max(1, np.array(parents < 0, int).sum())

    # def get_LineNumberTypeTrees(self):
    #
    #     parents = self.swcData[:, 6]
    #     lineNumbers = np.arange(self.swcData.shape[0])
    #     sortingInds = parents.argsort()
    #     lineNumbersSorted = lineNumbers[sortingInds]
    #     swcDataParentIDSortedAsc = self.swcData[sortingInds]
    #
    #     swcLineNumberTypeTrees = {}
    #
    #     for lineNumber, row in zip(lineNumbersSorted, swcDataParentIDSortedAsc):
    #
    #         index = int(row[0])
    #         nodeType = int(row[1])
    #         parentID = int(row[6])
    #
    #         if parentID < 0:
    #
    #             tempTree = Tree()
    #             tempTree.create_node(identifier=index, tag=nodeType, data=lineNumber)
    #             swcLineNumberTypeTrees[index] = tempTree
    #
    #         else:
    #
    #             treeBelonging = [x for x, y in swcLineNumberTypeTrees.iteritems()
    #                              if parentID in y.nodes.keys()]
    #
    #             if len(treeBelonging) == 1:
    #
    #                 swcLineNumberTypeTrees[treeBelonging[0]].create_node(identifier=index,
    #                                                                      tag=nodeType,
    #                                                                      parent=parentID,
    #                                                                      data=lineNumber)
    #             else:
    #                 logging.warning("Isolated node with index {} in {}".format(index,
    #                                                                        self.swcFile))
    #     return swcLineNumberTypeTrees

    def checkAndReturnFeasibleGraphsWithTypeLineNumber(self):

        allGraphMulti = nx.MultiDiGraph()

        for lineNumber, row in zip(xrange(self.swcData.shape[0]), self.swcData):

            index = int(row[0])
            nodeType = int(row[1])
            parentID = int(row[6])

            allGraphMulti.add_node(n=index, type=nodeType, lineNumber=lineNumber)
            allGraphMulti.add_node(n=parentID)
            allGraphMulti.add_edge(u=parentID, v=index)

        duplicateEdges = getDuplicates(allGraphMulti.edges())

        if duplicateEdges:
            raise ValueError("Duplicate connections detected! \n "
                             "File: {} \n "
                             "Index pairs of nodes with duplicate connections: \n "
                             "{}".format(self.swcFile, duplicateEdges))

        allGraph = nx.DiGraph(allGraphMulti)

        assert nx.is_directed_acyclic_graph(allGraph), "File {} has cyclic connections!".format(self.swcFile)

        # simpleCycles = [x for x in nx.simple_cycles(allGraph)]

        # if simpleCycles:
        #     raise ValueError("Cyclic connections detected! \n "
        #                      "File: {} \n "
        #                      "Index Lists of nodes forming cyclic connections: \n "
        #                      "{}".format(self.swcFile, simpleCycles))

        ccGraphs = list(nx.weakly_connected_component_subgraphs(allGraph))

        return ccGraphs

    @staticmethod
    def determine_soma_type(graph):

        assert nx.is_arborescence(graph), "graph must be an acyclic directed tree\n" \
                                          "see http://networkx.readthedocs.io/en/networkx-1.11/reference/algorithms.tree.html"

        rootNodes = [n for n, d in graph.in_degree().items() if d == 0]

        assert len(rootNodes) == 1, "graph has more than one roots"

        rootParent = rootNodes[0]

        rootNode = graph.successors(rootParent)[0]


        if graph.node[rootNode]["type"] == 1:

            # Collect, along with that of the root, IDs of all nodes that have a
            # continuous path to the root that consists only of soma nodes.
            somaConnectedNodeIDs = []

            for parentNode, childNodeList in nx.dfs_successors(graph, rootNode).items():

                parentType = graph.node[parentNode]["type"]

                for childNode in childNodeList:

                    childType = graph.node[childNode]["type"]
                    if parentType == 1 and childType == 1:

                        somaConnectedNodeIDs.append(childNode)

            # if there are zero or one child(ren) of the root that are of type 1 => 1-point soma
            if len(somaConnectedNodeIDs) in [0, 1]:
                return 0

            if len(somaConnectedNodeIDs) == 2:

                # if there are two children of the root that are of type 1 => 3-point soma
                if graph.predecessors(somaConnectedNodeIDs[0])[0] is rootNode and \
                                graph.predecessors(somaConnectedNodeIDs[1])[0] is rootNode:

                    return 1

                # if there are two non-child nodes connected to the root that are of type 1 => multiple cylinder soma
                else:

                    return 2

            # if there are more than two nodes connected to the root of type 1 => multiple cylinder soma
            elif len(somaConnectedNodeIDs) > 2:

                return 2

        # if root type is not 1 => no soma
        else:

            return 3

    def getSWCDatasetsTypes(self):

        swcGraphs = self.checkAndReturnFeasibleGraphsWithTypeLineNumber()

        toReturn = {}

        for graph in swcGraphs:

            treeSomaType = self.determine_soma_type(graph)

            treeNodeLineNumbers = [graph.node[x]["lineNumber"] for
                                   x in nx.topological_sort(graph)[1:]]

            treeData = self.swcData[treeNodeLineNumbers, :]

            toReturn[treeSomaType] = treeData

        return toReturn
























