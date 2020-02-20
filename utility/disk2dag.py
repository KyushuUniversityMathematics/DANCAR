#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_all(disks,fname):
    fig = plt.figure()
    ax = plt.axes()
    cmap = plt.get_cmap("Dark2")
    for i,v in enumerate(disks):
        c = patches.Circle(xy=(v[1], v[2]), radius=v[0], fc=cmap(int(i%10)),alpha=0.4)
        ax.add_patch(c)
        ax.text(v[1], v[2], i, size = 20, color = cmap(int(i%10)))
        c = patches.Circle(xy=(v[1], v[2]), radius=v[0], ec='black', fill=False)
        ax.add_patch(c)
    plt.axis('scaled')
    ax.set_aspect('equal')
    plt.savefig(fname)
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', help='Path to coordinates file',default=None)
    parser.add_argument('--dim', '-d', type=int, default=2,
                        help='Embedding dimension')
    parser.add_argument('--num', '-n', type=int, default=10,
                        help='number of disks')
    parser.add_argument('--outdir', '-o', help='Path to output',default="out")
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if args.input:
        dat = np.loadtxt(args.input,delimiter=",")
    else:
    # random generation
        dat = np.random.rand(args.num,args.dim+1)
        dat[:,0] / 4 # small radius to avoid too much overlap
        np.savetxt(os.path.join(args.outdir,"gen_coords.csv"),dat,delimiter=",")

    e = []
    for i in range(len(dat)-1):
        for j in range(i+1,len(dat)):
            d = np.sqrt(np.sum( (dat[i,1:]-dat[j,1:])**2) )
            if d + dat[i,0] < dat[j,0]:
                e.append((j,i))
            elif d + dat[j,0] < dat[i,0]:
                e.append((i,j))
    np.savetxt(os.path.join(args.outdir,"gen_dag.csv"),np.array(e, dtype=np.int),delimiter=",", fmt="%i")
    plot_all(dat,os.path.join(args.outdir,"plot.png"))

    