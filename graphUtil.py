import networkx as nx
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import argparse

# read graph from csv
def read_graph(fname,offset=0):
    vert = set()
    edge = set()
    with open(fname) as infh:
        for line in infh:
            l = line.strip().split(',')
            vert.add(l[0])
            for i in range(len(l)-1):
                edge.add((l[i],l[i+1]))
                vert.add(l[i+1])
    vlist = np.array(list(vert),dtype=np.int32)
    elist = np.array(list(edge),dtype=np.int32)
    return(vlist-offset,elist-offset)

def plot_digraph(edge,fname):
    G = nx.DiGraph()
    G.add_edges_from(edge)
    plt.figure(figsize=(15,15))
    pos = nx.fruchterman_reingold_layout(G)
    nx.draw_networkx(G,pos,node_color="#5050ff",font_size=0,node_size=75)
    plt.savefig(fname)

# plot results (works only with dim=2)
def plot_disks(disks,fname):
    fig = plt.figure()
    ax = plt.axes()
    cmap = plt.get_cmap("Dark2")
    dim = (disks.shape[1]-1)//2
    min_r = np.min(disks[:,0])
    for i,v in enumerate(disks):
        # disk
        c = patches.Circle(xy=(v[1+dim], v[2+dim]), radius=v[0], fc=cmap(int(i%10)),alpha=0.4)
        ax.add_patch(c)
        ax.text(v[1], v[2], i, size = 20, color = cmap(int(i%10)))
        # disk boundary
        c = patches.Circle(xy=(v[1+dim], v[2+dim]), radius=v[0], ec='black', fill=False)
        ax.add_patch(c)
        # anchor
        c = patches.Circle(xy=(v[1], v[2]), radius=min_r/100, ec='black', fill=True)
        ax.add_patch(c)
    plt.axis('scaled')
    ax.set_aspect('equal')
    plt.savefig(fname)
    plt.close()

# reconstruct digraph from arrangements
def reconstruct(disks, dag=0.0, max_edges=1e7):
    dim = (disks.shape[1]-1)//2
    x = disks[:,1:(dim+1)]
    c = disks[:,(dim+1):]
    G = []
#    dm = np.sum((np.expand_dims(x,axis=0) - np.expand_dims(c,axis=1))**2,axis=2)
#    dm += np.max(r2)*np.eye(len(dm))
    for i in range(len(x)):
        d = np.sqrt(np.sum((c[i]-x)**2,axis=1))
        d[i] += disks[i,0]
        E = [(i,j) for j in np.where(d+dag*disks[:,0] < disks[i,0])[0]]
        G.extend(E)
    return(np.array(G,dtype=np.int32))

# read graph from csv
# def read_graph_old(fname):
#     g = nx.DiGraph()
#     g_trans = set()
#     with open(fname) as infh:
#         for line in infh:
#             l = line.strip().split(',')
#             g.add_edges_from([(l[i],l[i+1]) for i in range(len(l)-1)])
#     pos_edge = []
#     reachable = {}
#     for v in g.nodes():
#         reachable[v] = nx.descendants(g,v)
#         for w in reachable[v]:
#             pos_edge.append((v,w))
#             g_trans.add((v,w))
#         reachable[v].add(v)
#         g_trans.add((v,v))
#     neg_edge = []
#     super_neg_edge = []
#     for v in g.nodes():
#         for w in g.nodes():
#             if (v,w) not in g_trans:
#                 neg_edge.append((v,w))
#                 # pair of nodes with no common descendant
#                 if not (reachable[v] & reachable[w]):
#                     super_neg_edge.append((v,w))
#     print("#edges {}, #vertices {}".format(len(pos_edge),len(g.nodes())))
#     return (len(g.nodes()),pos_edge,neg_edge,super_neg_edge)

def check_anchor_containment(disks):
    dim = (disks.shape[1]-1)//2
    r2 = disks[:,0]**2
    x = disks[:,1:(dim+1)]
    c = disks[:,(dim+1):]
    violation = 0
    for i in range(len(x)):
        if np.sum((x[i]-c[i])**2) > r2[i]:
            violation += 1
    return(violation,len(x))

def compare_graph(go,gr,output=True):
    n  = go.number_of_nodes()
    eo = set(go.edges())
    er = set(gr.edges())
    tp, fp, fn = len(eo & er), len(er - eo), len(eo - er)
    tn = n * (n-1) - tp - fp - fn

    accuracy  = (tp + tn) / (n * (n-1))
    precision = tp / max((tp + fp),1)
    recall    = tp / max((tp + fn),1)
    f1_score  = (2 * recall * precision) / max((recall + precision),1)

    if output:
        print("\n\n Confusion Matrix")
        print(f"{tp}\t{fp}")
        print(f"{fn}\t{tn}")
        print(f"f1_score    : {f1_score}")
        print(f"precision   : {precision} = {tp} / {tp + fp}")
        print(f"recall      : {recall} = {tp} / {tp + fn}")
        print(f"accuracy    : {accuracy} = {tp + tn} / {n * (n-1)}")

    return(f1_score,precision,recall,accuracy)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Digraph Embedding')
    parser.add_argument('--input', '-i', help='Path to digraph description file')
    parser.add_argument('--reconstructed', '-r', default="reconstructed.csv", help='Path to digraph description file')
    parser.add_argument('--coordinates', '-c', help='Path to coordinate file')
    parser.add_argument('--dag', type=float, default=0, help='0:non-acyclic, 1:acyclic')
    args = parser.parse_args()

    if args.coordinates:
        print("reconstructing...")
        dat = np.loadtxt(args.coordinates,delimiter=",")
        redge = reconstruct(dat,dag=args.dag)
        if args.reconstructed:
            np.savetxt(args.reconstructed,redge,fmt='%i',delimiter=",")
        print("Anchor violation/All vertices = ",check_anchor_containment(dat))
        reconstG = nx.from_edgelist(redge)
    else:
        reconstG = nx.from_edgelist(read_graph(args.reconstructed)[1],nx.DiGraph())

    if args.input:
        print("comparing...")
        compare_graph(nx.from_edgelist(read_graph(args.input)[1],nx.DiGraph()),reconstG)
