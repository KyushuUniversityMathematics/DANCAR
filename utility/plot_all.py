import argparse
import numpy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def mk_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input",type=str)
    parser.add_argument("-f", "--folder_path",type=str,help='folder_path')
    parser.add_argument("-e", "--edge_file",type=str,help='input_file', default="dag_pos.csv")
    parser.add_argument("-pe", "--positive_edges_file",type=str,help='input_file', default="dag_full_transitive.csv")
    parser.add_argument("-ne", "--negative_edges_file",type=str,help='input_file', default="dag_negative.csv")
    parser.add_argument("-s", "--save", action='store_true', help="whether file save")
    args = parser.parse_args()
    return args

def plot_all(disks,min_radius,args):
    fig = plt.figure()
    ax = plt.axes()

    # fc = face color, ec = edge color
    # cmap = plt.get_cmap("tab10")
    cmap = plt.get_cmap("Dark2")

    for i,v in enumerate(disks):
        xy = tuple(v[1:])
        # print(v,xy)
        # c = patches.Circle(xy=xy, radius=v[0]-min_radius+1, fc=color[i],alpha=0.3)
        c = patches.Circle(xy=xy, radius=v[0], fc=cmap(int(i%10)),alpha=0.4)
        ax.add_patch(c)
        ax.text(xy[0], xy[1], i, size = 20, color = cmap(int(i%10)))
        c = patches.Circle(xy=xy, radius=v[0], ec='black', fill=False)
        ax.add_patch(c)

    plt.axis('scaled')
    ax.set_aspect('equal')

    if args.save == True:
        plt.savefig(f"plot.png")
    else:
        plt.show()
    plt.close()

def plot_edge(edges,disks,min_radius,args,folder_name):
    cmap = plt.get_cmap("Dark2")
    os.makedirs(f"{args.folder_path}/{folder_name}",exist_ok=True)
    for edge in edges:
        fig = plt.figure()
        ax = plt.axes()

        for node in edge:
            radius,node_x,node_y = disks[node]
            ax.text(node_x, node_y, node, size = 20, color = cmap(int(node%10)))
            c = patches.Circle(xy=(node_x,node_y), radius=radius, fc=cmap(int(node%10)),alpha=0.4)
            ax.add_patch(c)

        plt.axis('scaled')
        ax.set_aspect('equal')
        plt.title(f'{edge[0]} -> {edge[1]}')

        if args.save == True:
            plt.savefig(f"{args.folder_path}/{folder_name}/{edge[0]}_{edge[1]}.png")
        else:
            plt.show()    
        plt.close()

def load_csv(filepath,data_type):
    return [tuple(map(data_type, line.split(','))) for line in open(filepath)]

def main():
    args = mk_args()
#    edges = load_csv(args.edge_file,int)
#    positive_edges = load_csv(args.positive_edges_file,int)
#    negative_edges = set(load_csv(args.negative_edges_file,int)) - {(w,v) for v,w in positive_edges}
    disks = numpy.loadtxt(args.input,delimiter=",")
    min_radius = 0#min(disk[0] for disk in disks)

    plot_all(disks,min_radius,args)
#    plot_edge(edges,disks,min_radius,args,'edges')
#    plot_edge(positive_edges,disks,min_radius,args,'positive')
#    plot_edge(negative_edges,disks,min_radius,args,'negative')


if __name__ == '__main__':
    main()