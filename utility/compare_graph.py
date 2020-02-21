import networkx as nx
from sys import argv
from itertools import product

def main(args):
    """
    usage : python3 compare_graph.py original_graph.csv reconstructed_graph.csv
    """
    original, reconst = args[0],args[1]
    go = read_graph(original)
    gr = read_graph(reconst)
    
    n  = go.number_of_nodes()

    eo = set(go.edges())
    er = set(gr.edges())

    tp = len(eo & er)             #True Positive
    fp = len(er - eo)             #False Positive
    fn = len(eo - er)             #False Negative
    tn = n * (n-1) - tp - fp - fn #True Negative

    accuracy  = (tp + tn) / (n * (n-1))
    precision = tp / (tp + fp)
    recall    = tp / (tp + fn)
    f1_score  = (2 * recall * precision) / (recall + precision)

    print(f"f1_score    : {f1_score}")
    print(f"precision   : {precision} = {tp} / {tp + fp}")
    print(f"recall      : {recall} = {tp} / {tp + fn}")
    print(f"accuracy    : {accuracy} = {tp + tn} / {n * (n-1)}")
    print(f"tp          : {tp}")
    print(f"fp          : {fp}")
    print(f"fn          : {fn}")
    print(f"tn          : {tn}")

def read_graph(fname):
    g = nx.DiGraph()
    with open(fname) as infh:
        for line in infh:
            l = line.strip().split(',')
            for i in range(len(l)-1):
                g.add_edge(l[i],l[i+1])
    print("#edges {}, #vertices {}".format(g.number_of_edges(),g.number_of_nodes()))

    return g

if __name__ == "__main__":
    main(argv[1:])
