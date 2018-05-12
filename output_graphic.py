import matplotlib.pyplot as plt
import networkx as nx
import preprocessing as pp
import graphical_lasso as gl
import numpy as np


def graph_from_precision_matrix(precision, sources):
    edges = []
    nodes = {}
    node_degree = {}
    node_id = 1
    for i in range(len(sources)):
        nodes[sources[i]] = node_id
        node_degree[node_id] = 0
        node_id += 1

    for (i, j) in zip(*np.where(precision > 0)):
        if i > j:
            edges.append([nodes[sources[i]], nodes[sources[j]]])
            node_degree[i+1] += 1
            node_degree[j+1] += 1
    G = nx.make_small_graph(["edgelist", "source graph", len(nodes), edges])
    return G, nodes, node_degree


def draw_graph(G, nodes, node_degree, topic):
    node_labels = {value-1: key for key, value in nodes.items()}
    plt.figure(figsize=(24, 6))
    # strip url.
    nodes = {}
    for i in node_labels:
        node_labels[i] = node_labels[i].split(".")[0]
        nodes[node_labels[i]] = i

    sorted_nodes = sorted(node_degree.items(), key=lambda x: -x[1])
    sorted_node_ids = list(zip(*sorted_nodes)[0])
    # pos = nx.shell_layout(G, [[0] + sorted_node_ids[:16],
    #                           sorted_node_ids[16:]
    #                          ], scale=1.0)
    pos = nx.kamada_kawai_layout(G, scale=.5)

    # nodes
    nx.draw_networkx_nodes(G, pos,
                           node_color='black',
                           node_size=1,
                           alpha=0.0)
    # edges
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=1.0)

    for i in pos:
        pos[i][0] -= 0.0

    print(nodes.keys())
    if topic == "isis":
        pos[nodes["wikinews"]][0] -= .1
        pos[nodes["wikinews"]][1] += .1
        pos[nodes["bloomberg"]][0] -= .1
        pos[nodes["bloomberg"]][1] -= .1
        pos[nodes["techcrunch"]][0] += 0.1
        pos[nodes["techcrunch"]][1] += 0.2
        pos[nodes["independent"]][0] += 0.05
        pos[nodes["independent"]][1] += 0.15
        pos[nodes["businessinsider"]][0] -= 0.02
    else:
        pos[nodes["wikinews"]][0] -= .15
        pos[nodes["nytimes"]][1] += -.05
        pos[nodes["techcrunch"]][0] -= 0.2
        pos[nodes["middleeasteye"]][0] += -0.01
        pos[nodes["middleeasteye"]][1] += -0.01

    nx.draw_networkx_labels(G, pos, node_labels,
                            font_size=20,
                            font_color="white",
                            bbox=dict(
                                boxstyle="square,pad=0.3",
                                fc="black",
                                ec="white",
                                lw=1,
                                alpha=0.9
                            ))

    plt.axis('off')
    plt.savefig(topic+".pdf")


def main(topic="isis"):
    # get A
    A, sources = pp.get_A_and_labels(topic)
    print(np.unique(A))
    A = A.astype('float64')
    if topic == "brexit":
        # add noise to brexit.
        A += np.random.randn(A.shape[0], A.shape[1])*1e-16

    # compute precision
    lasso = gl.GraphicalLasso(convergence_threshold=1e-6, lambda_param=1e-5/4)
    precision = lasso.execute(A)

    # plot precision
    G, nodes, node_degree = graph_from_precision_matrix(precision, sources)
    draw_graph(G, nodes, node_degree, topic)


if __name__ == "__main__":
    main("isis")
    main("brexit")
