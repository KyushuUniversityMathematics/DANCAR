import networkx as nx
from sys import argv
from itertools import product

def main(args):
    original, reconst = args[0],args[1]
    go = read_graph(original)
    gr = read_graph(reconst)
    
    n  = go.number_of_nodes()

    eo = set(go.edges())
    er = set(gr.edges())

    tp = len(eo & er)             #枝があり、枝があると予測
    fp = len(er - eo)             #枝がなく、枝があると予測
    fn = len(eo - er)             #枝があり、枝がないと予測
    tn = n * (n-1) - tp - fp - fn #枝がなく、枝がないと予測

    accuracy  = (tp + tn) / (n * (n-1))
    precision = tp / (tp + fp)
    recall    = tp / (tp + fn)
    f1_score  = (2 * recall * precision) / (recall + precision)

    print("f1_score  :",f1_score)
    print("precision :",precision)
    print("recall    :",recall)
    print("accuracy  :",accuracy)

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
