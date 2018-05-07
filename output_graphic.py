import matplotlib.pyplot as plt
import networkx as nx
import preprocessing as pp
import graphical_lasso as gl
import numpy as np


def get_precision(A):
    lasso = gl.GraphicalLasso(convergence_threshold=1e-6, lambda_param=1e-6)
    precision = lasso.execute(A)
    return precision


def graph_from_precision_matrix(precision, sources):
    edges = []
    nodes = {}
    edge_labels = []
    node_id = 1
    for (i, j) in zip(*np.where(precision > 0)):
        if i > j:
            if sources[i] not in nodes:
                nodes[sources[i]] = node_id
                node_id += 1
            if sources[j] not in nodes:
                nodes[sources[j]] = node_id
                node_id += 1
            edge_labels.append([sources[i], sources[j]])
            edges.append([nodes[sources[i]], nodes[sources[j]]])
    G = nx.make_small_graph(["edgelist", "source graph", len(nodes), edges])
    node_labels = {value-1: key for key, value in nodes.items()}

    return G, node_labels


def draw_graph(G, node_labels):
    pos = nx.spring_layout(G, iterations=500)  # positions for all nodes

    # nodes
    nx.draw_networkx_nodes(G, pos,
                           node_color='black',
                           node_size=5,
                           alpha=0.1)
    # edges
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=1.0)

    nx.draw_networkx_labels(G, pos, node_labels,
                            font_size=12,
                            font_color="white",
                            bbox=dict(
        boxstyle="square,pad=0.3",
        fc="black",
        ec="white",
        lw=1,
        alpha=0.5
    ))

    plt.axis('off')
    plt.show()


def main(topic="isis"):
    # get A
    A, sources = pp.get_A_and_labels(topic)
    print(np.unique(A))
    # compute precision
    precision = get_precision(A)

    # plot precision
    G, node_labels = graph_from_precision_matrix(precision, sources)
    draw_graph(G, node_labels)


if __name__ == "__main__":
    main("brexit")

