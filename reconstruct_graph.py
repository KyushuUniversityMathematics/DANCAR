#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
from itertools import combinations as cmb
import networkx as nx
from simple_table import SimpleTable
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--disk', '-d', help='Path to disk file',default=None)
    # parser.add_argument('--dim', '-d', type=int, default=2,
    #                     help='Embedding dimension')
    parser.add_argument('--graph', '-g', help='Path to graph file',default=None)
    # parser.add_argument('--num', '-n', type=int, default=10,
    #                     help='number of disks')
    parser.add_argument('--outdir', '-o', help='Path to output',default="out")

    args = parser.parse_args()
    return args

def read_graph(filename):
    vert = set()
    edge = set()
    for line in open(filename):
        l = tuple(map(int, line.strip().split(',')))
        vert |= set(l)
        edge |= set(zip(l,l[1:]))
    # print("original : #edges {}, #vertices {}".format(len(edge),len(vert)))
    return len(vert),edge

def full_transitive(edge):
    g = nx.DiGraph()
    g.add_edges_from(edge)
    reachable = {(u,v) for u in g.node for v in nx.descendants(g,u)}
    # print("closure  : #edges {}, #vertices {}".format(len(reachable),g.number_of_nodes()))
    return reachable

def reconstruct_graph(filename):
    disks = {} #idx -> (r,x,y)
    edge = set()

    for idx, line in enumerate(open(filename)):
        disks[idx] = tuple(map(float, line.strip().split(',')))
    
    for u, v in cmb(disks, 2):
        ru, xu, yu = disks[u]
        rv, xv, yv = disks[v]
        dist_squared = (xu-xv) ** 2 + (yu-yv) ** 2
        if dist_squared < ru: #v in B(u)
            edge.add((u,v))
        if dist_squared < rv: #u in B(v)
            edge.add((v,u))
    
    vert = set(disks.keys())

    # print("reconst  : #edges {}, #vertices {}".format(len(edge),len(vert)))
    return len(vert),edge

def display_difference(reconst, original,n_vert, title):
    false_negative = int(n_vert * (n_vert-1) / 2) - len(reconst | original)

    recall    = len(reconst & original) / len(original)
    precision = len(reconst & original) / len(reconst)
    f1_score  = recall * precision * 2 / (precision+recall)
    table = SimpleTable()
    table.set_headers((title, 'positive', 'negative', 'precision'))
    table.add_row(('predicted positive',f'{len(reconst & original)}',f'{len(reconst - original)}', f'{precision:.5f}'))
    table.add_row(('predicted negative',f'{len(original - reconst)}',f'{false_negative}',''))
    table.add_row(('recall',f'{recall:.5f}','',f'f1={f1_score:.5f}'))
    print(table)

    # print(f",true,false")
    # print(f"positive,{len(reconst & original)},{len(reconst - original)}")
    # print(f"negative,{len(original - reconst)},{false_negative}")

def main():
    print()
    args = get_arguments()
    os.makedirs(args.outdir, exist_ok=True)

    n_vert, edges_reconst  = reconstruct_graph(args.disk)
    _,      edges_original = read_graph(args.graph)
    edges_full = full_transitive(edges_original)
    
    #display the difference between 2 graphs
    display_difference(edges_reconst, edges_original, n_vert, "ORIGINAL")
    display_difference(edges_reconst, edges_full, n_vert, "CLOSURE")

if __name__ == '__main__':
    main()