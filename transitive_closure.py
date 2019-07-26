import networkx as nx
import argparse
def read_graph(filename):
    g = nx.DiGraph()
    g_trans = nx.DiGraph()
    with open(filename) as infh:
        for line in infh:
            l = line.strip().split(',')
            l = [i for i in l if i!='']
            g.add_edges_from([(l[i],l[i+1]) for i in range(len(l)-1)])
    return g

def write_transitive_closure(g,filename):
    g_trans = nx.DiGraph()
    f = open(filename, 'w')
    for v in g.nodes():
        reachable_from_v = nx.descendants(g,v)
        # print(v,reachable_from_v)
        for w in reachable_from_v:
            print(f"{v},{w}", file=f)
            g_trans.add_edge(v,w)
    f.close()
    return g_trans
def write_negative_edges(g_trans,filename):
    f = open(filename, 'w')
    for v in g_trans.nodes():
        for w in g_trans.nodes():
            if (v,w) not in g_trans.edges() and v != w:
                print(f"{v},{w}", file=f)
                # print(v,w)
    f.close()
def main():
    parser = argparse.ArgumentParser(description='Disk Embedding')
    parser.add_argument('--input', '-i', default="./dag_pos.csv",help='input graph')
    parser.add_argument('--output', '-o', default="./dag_full_transitive.csv",help='output graph')
    parser.add_argument('--negative', '-n', default="./dag_negative.csv",help='negative graph')
    args = parser.parse_args()

    g = read_graph(args.input)
    g_trans = write_transitive_closure(g,args.output)
    write_negative_edges(g_trans,args.negative)
if __name__ == "__main__":
    main()