#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
the source code of Diagram-ANChor Embedding

2019-12-07 update negative loss
"""

from __future__ import print_function
import numpy as np
import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training,datasets,iterators,Variable
from chainer.training import extensions
from chainer.dataset import dataset_mixin, convert, concat_examples
import pandas as pd
from chainerui.utils import save_args

import os
from datetime import datetime as dt
from consts import optim,dtypes
import networkx as nx
import shutil

from itertools import product, combinations
## dataset preparation
class Dataset_edge(dataset_mixin.DatasetMixin):
    def __init__(self, dat):
        self.edges = np.array(dat, dtype=np.int32)

    def __len__(self):
        return len(self.edges)

    def get_example(self, i):
        return self.edges[i,0],self.edges[i,1]

class Dataset_vert(dataset_mixin.DatasetMixin):
    def __init__(self, dat):
        self.vertices = np.array(dat, dtype=np.int32)

    def __len__(self):
        return len(self.vertices)

    def get_example(self, i):
        return self.vertices[i]

## updater 
class Updater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.coords = kwargs.pop('models')
        params = kwargs.pop('params')
        super(Updater, self).__init__(*args, **kwargs)
        self.args = params['args']
    def update_core(self):
        epsilon = 1e-10
        opt = self.get_optimizer('main')
        nDim = self.args.dim
        margin = self.args.margin

        # r = F.relu(self.coords.W[:,0] - 0.1) + 0.1
        r = F.relu(self.coords.W[:,0]) + 0.1
        c = self.coords.W[:,1:nDim+1]
        x = self.coords.W[:,nDim+1:]

        # positive loss
        batch = self.get_iterator('main').next()
        a,b = self.converter(batch) #edge a->b
        distance = F.sqrt(F.sum((c[a]-x[b])**2,axis=1)+epsilon)
        loss_pos = F.average(F.relu(distance - r[a] + margin))
        chainer.report({'loss_pos': loss_pos}, self.coords)
        loss = loss_pos

        # anchor loss
        batch = self.get_iterator('vertex').next()
        v = self.converter(batch) #vertex a
        distance2 = F.sum((c[v]-x[v])**2,axis=1) + epsilon
        loss_vert = F.average(F.relu(F.sqrt(distance2) - r[v] + margin))
        chainer.report({'loss_vert': loss_vert}, self.coords)
        loss += loss_vert * self.args.lambda_vert

        #anchor should be near center
        # if self.args.lambda_spring > 0:

        distance2 = F.sum((c[v]-x[v])**2,axis=1) + epsilon
        loss_spring = F.average(F.sqrt(distance2))
        chainer.report({'loss_spring': loss_spring}, self.coords)
        loss += loss_spring * self.args.lambda_spring

        # radius should be similar
        if self.args.lambda_uniform_radius>0:
            loss_uniform_radius = F.average( (F.max(r)-F.min(r))**2 )
            # loss_uniform_radius = F.average(F.relu(r[b] - r[a] + margin))
            chainer.report({'loss_uniform_radius': loss_uniform_radius}, self.coords)
            loss += self.args.lambda_uniform_radius * loss_uniform_radius

        # negative sample
        if self.args.lambda_neg>0:
            batch = self.get_iterator('negative').next()
            a,b = self.converter(batch) #there is no edge a->b
            distance = F.sqrt(F.sum((c[a]-x[b])**2,axis=1) + epsilon)
            loss_neg = F.average(F.relu(r[a] - distance + margin))
            # distance = F.sqrt(F.sum((c[a]-c[b])**2,axis=1) + epsilon)
            # loss_neg = F.average(F.relu(r[a] + r[b] - distance + margin))
            chainer.report({'loss_neg': loss_neg}, self.coords)
            loss += self.args.lambda_neg * loss_neg

        # coulomb force between points
        if self.args.lambda_coulomb>0:
            #(1-1)点の並び替え
            # p = np.random.permutation(len(r))
            # a,b = p[1:], p[:-1]

            #(1-2)batch数分の頂点の全組み合わせ
            # p = np.array(list(combinations(set(v),2)))
            # a,b = p[:,0], p[:,1]

            #(1-3)全頂点の全組み合わせ
            # p = np.array(list(combinations(range(len(r)),2)))
            # a,b = p[:,0], p[:,1]

            # TODO: 別のbatchを作成する
            #(1-4)別のiteratorを用意して(1-2)と同じことを行う
            batch = self.get_iterator('coulomb').next()
            v = self.converter(batch) #vertex a
            p = np.array(list(combinations(set(v),2)))
            a,b = p[:,0], p[:,1]

            #(2-1)positiveと逆向きのLossを加える場合
            # distance_ab = F.sqrt(F.sum((c[a]-x[b])**2,axis=1) + epsilon)
            # distance_ba = F.sqrt(F.sum((c[b]-x[a])**2,axis=1) + epsilon)
            # loss_coulomb = F.average(F.relu(r[a] - distance_ab + margin)) \
            #          + F.average(F.relu(r[b] - distance_ba + margin))

            #(2-2)円盤が重ならないようにする場合
            distance = F.sqrt(F.sum((c[a]-c[b])**2,axis=1)+epsilon)
            loss_coulomb = F.average(F.relu(r[a] + r[b] - distance + margin))
            chainer.report({'loss_coulomb': loss_coulomb}, self.coords)
            loss += self.args.lambda_coulomb * loss_coulomb

        # update
        self.coords.cleargrads()
        loss.backward()
        opt.update(loss=loss)

# evaluator
class Evaluator(extensions.Evaluator):
    name = "myval"
    def __init__(self, *args, **kwargs):
        params = kwargs.pop('params')
        super(Evaluator, self).__init__(*args, **kwargs)
        self.args = params['args']
        self.count = 0
    def evaluate(self):
        coords = self.get_target('main')
        if self.eval_hook:
            self.eval_hook(self)
        if(self.args.gpu>-1):
            dat = coords.xp.asnumpy(coords.W.data).copy()
        else:
            dat = coords.W.data.copy()
        dat[:,0] = np.maximum(dat[:,0],0)+0.1
        np.savetxt(os.path.join(self.args.outdir,"out{:0>4}.csv".format(self.count)), dat, fmt='%1.5f', delimiter=",")
        if self.args.visualize:
            plot_all(dat,os.path.join(self.args.outdir,"plot{:0>4}.png".format(self.count)))
        reconstruct_graph(dat,os.path.join(self.args.outdir,"reconstruct{:0>4}.csv".format(self.count)))
        self.count += 1
        return {"myval/none":0}

def reconstruct_graph(disks, fname):
    edges = set()
    nDim = (len(disks[0]) - 1) // 2
    r = disks[:,0]
    c = disks[:,1:nDim+1]
    x = disks[:,nDim+1:]
    nVert = len(r)
    for a,b in combinations(range(nVert),2):
        # judge whether there is an edge u->v
        d_a2b = np.sqrt(np.sum((c[a]-x[b])**2))
        d_b2a = np.sqrt(np.sum((c[b]-x[a])**2))

        if d_a2b <= r[a]:
            edges.add((a,b))
        if d_b2a <= r[b]:
            edges.add((b,a))

    f = open(fname, "w")
    for i,j in sorted(edges):
        print(f"{i},{j}",file=f)

# # plot results
# def plot_all(disks,fname):
#     fig = plt.figure()
#     fig.xlim(-5,5)
#     fig.ylim(-5,5)
#     ax = plt.axes()
#     cmap = plt.get_cmap("Dark2")
#     for i,v in enumerate(disks):
#         c = patches.Circle(xy=(v[1], v[2]), radius=v[0], fc=cmap(int(i%10)),alpha=0.4)
#         ax.add_patch(c)
#         ax.text(v[1], v[2], i, size = 20, color = cmap(int(i%10)))
#         c = patches.Circle(xy=(v[1], v[2]), radius=v[0], ec='black', fill=False)
#         ax.add_patch(c)
#     plt.axis('scaled')
#     ax.set_aspect('equal')

#     plt.savefig(fname)
#     plt.close()

# read graph from csv
def read_graph(fname, use_negative_sample):
    vert = set()
    edge = set()
    with open(fname) as infh:
        for line in infh:
            l = line.strip().split(',')
            vert.add(l[0])
            for i in range(len(l)-1):
                edge.add((l[i],l[i+1]))
                vert.add(l[i+1])
    print("#edges {}, #vertices {}".format(len(edge),len(vert)))
    if use_negative_sample:
        neg_edge = set(product(vert,vert)) - edge
    else:
        neg_edge = set()
    return len(vert), list(vert), list(edge), list(neg_edge)

def plot_all(disks, fname):
    fig = plt.figure()
    ax = plt.axes()
    cmap = plt.get_cmap("Dark2")
    for i,v in enumerate(disks):
        radius = v[0]
        center = (v[1], v[2])
        anchor = (v[3], v[4])
        c = patches.Circle(xy=center, radius=radius, fc=cmap(int(i%10)),alpha=0.4)
        ax.add_patch(c)
        c = patches.Circle(xy=anchor, radius=0.05, fc=cmap(int(i%10)))
        ax.add_patch(c)
        # ax.text(v[1], v[2], i, size = 20, color = cmap(int(i%10)))
        c = patches.Circle(xy=center, radius=v[0], ec='black', fill=False)
        ax.add_patch(c)
    plt.axis('scaled')
    ax.set_aspect('equal')
    plt.savefig(fname)
    plt.close()

## main
def main():
    # command line argument parsing
    parser = argparse.ArgumentParser(description='Disk Embedding')
    parser.add_argument('input', help='Path to DAG description file',default="split.csv")
    parser.add_argument('--batchsize', '-bs', type=int, default=1,
                        help='Number of samples in each mini-batch')
    parser.add_argument('--batchsize_coulomb', '-bc', type=int, default=None,
                        help='Number of samples in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--dim', '-d', type=int, default=2,
                        help='Embedding dimension')
    parser.add_argument('--margin', '-m', type=float, default=0.3,
                        help='margin for the metric boundary')
    parser.add_argument('--weight_decay', '-wd', type=float, default=0,
                        help='weight decay for regularization on coordinates')
    parser.add_argument('--wd_norm', '-wn', choices=['l1','l2'], default='l2',
                        help='norm of weight decay for regularization')
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-2,
                        help='learning rate')
    parser.add_argument('--learning_rate_drop', '-ld', type=int, default=5,
                        help='how many times to half learning rate')
    # parser.add_argument('--lambda_super_neg', '-lsn', type=float, default=1,
    #                     help='Super negative samples')
    parser.add_argument('--lambda_neg', '-ln', type=float, default=1,
                        help='negative samples')
    parser.add_argument('--lambda_coulomb', '-lc', type=float, default=0,
                        help='Coulomb force between disks')
    parser.add_argument('--lambda_uniform_radius', '-lur', type=float, default=0,
                        help='Radius should be similar')
    parser.add_argument('--lambda_vert', '-lv', type=float, default=0.5,
                        help='anchor should be in its corresponding disk')
    parser.add_argument('--lambda_spring', '-ls', type=float, default=0,
                        help='anchor should be close to its corresponding center')
    parser.add_argument('--outdir', '-o', default='../result',
                        help='Directory to output the result')
    parser.add_argument('--optimizer', '-op',choices=optim.keys(),default='Adam',
                        help='optimizer')
    parser.add_argument('--vis_freq', '-vf', type=int, default=200,
                        help='visualisation frequency in iteration')
    parser.add_argument('--visualize', '-vis', action='store_false',
                        help='visualization flag')
    parser.add_argument('--mpi', action='store_true',help='parallelise with MPI')
    args = parser.parse_args()

    args.outdir = os.path.join(args.outdir, dt.now().strftime('%m%d_%H%M'))
    save_args(args, args.outdir)

    chainer.config.autotune = True

    if args.batchsize_coulomb == None:
        args.batchsize_coulomb = args.batchsize

    ## ChainerMN
    if args.mpi:
        import chainermn
        if args.gpu >= 0:
            comm = chainermn.create_communicator('hierarchical')
            chainer.cuda.get_device(comm.intra_rank).use()
        else:
            comm = chainermn.create_communicator('naive')
        if comm.rank == 0:
            primary = True
            print(args)
            chainer.print_runtime_info()
        else:
            primary = False
        print("process {}".format(comm.rank))
    else:
        primary = True
        print(args)
        chainer.print_runtime_info()
        if args.gpu >= 0:
            chainer.cuda.get_device(args.gpu).use()

    use_negative_sample = (args.lambda_neg > 0)
    vnum, vert, pos_edge, neg_edge = read_graph(args.input, use_negative_sample)
    # print(pos_edge)
    vert_train_iter = iterators.SerialIterator(Dataset_vert(vert), args.batchsize, shuffle=True)
    train_iter = iterators.SerialIterator(Dataset_edge(pos_edge), args.batchsize, shuffle=True)
    if use_negative_sample:
        neg_train_iter = iterators.SerialIterator(Dataset_edge(neg_edge), args.batchsize, shuffle=True)
    else:
        neg_train_iter = iterators.SerialIterator(Dataset_edge(vert), args.batchsize, shuffle=True)
    coulomb_train_iter = iterators.SerialIterator(Dataset_vert(vert), args.batchsize_coulomb, shuffle=True)
    # super_neg_train_iter = iterators.SerialIterator(Dataset_edge(super_neg_edge), args.batchsize, shuffle=True)
    # initial embedding [1,2]^(dim+1),  the first coordinate corresponds to r (radius)
    coords = np.random.rand(vnum,args.dim*2+1)+1
    coords[:,0] = 0.1
    coords[:,args.dim+1:] = coords[:,1:args.dim+1] 
    coords = L.Parameter(coords)
    
    # Set up an optimizer
    def make_optimizer(model):
        if args.optimizer in ['SGD','Momentum','CMomentum','AdaGrad','RMSprop','NesterovAG']:
            optimizer = optim[args.optimizer](lr=args.learning_rate)
        elif args.optimizer in ['AdaDelta']:
            optimizer = optim[args.optimizer]()
        elif args.optimizer in ['Adam','AdaBound','Eve']:
            optimizer = optim[args.optimizer](alpha=args.learning_rate, weight_decay_rate=args.weight_decay)
        if args.mpi:
            optimizer = chainermn.create_multi_node_optimizer(optimizer, comm)
        optimizer.setup(model)
        return optimizer

    opt = make_optimizer(coords)
    if args.weight_decay>0 and (not args.optimizer in ['Adam','AdaBound','Eve']):
        if args.wd_norm =='l2':
            opt.add_hook(chainer.optimizer_hooks.WeightDecay(args.weight_decay))
        else:
            opt.add_hook(chainer.optimizer_hooks.Lasso(args.weight_decay))

    if args.gpu >= 0:
        coords.to_gpu() 

    updater = Updater(
        models=coords,
        iterator={'vertex':vert_train_iter, 'main': train_iter, 'negative': neg_train_iter, 'coulomb':coulomb_train_iter},
        optimizer={'main': opt},
        device=args.gpu,
#        converter=convert.ConcatWithAsyncTransfer(),
        params={'args': args}
        )
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.outdir)

    if primary:
        log_interval = 100, 'iteration'
        log_keys = ['iteration','lr','elapsed_time','main/loss_pos', 'main/loss_neg', 'main/loss_coulomb', 'main/loss_vert', 'main/loss_spring']
        trainer.extend(extensions.observe_lr('main'), trigger=log_interval)
        trainer.extend(extensions.LogReport(keys=log_keys, trigger=log_interval))
        trainer.extend(extensions.PrintReport(log_keys), trigger=log_interval)
        if extensions.PlotReport.available():
            trainer.extend(extensions.PlotReport(log_keys[3:], 'epoch', file_name='loss.png'))
        trainer.extend(extensions.ProgressBar(update_interval=10))
        trainer.extend(extensions.LogReport(trigger=log_interval))
        trainer.extend(Evaluator(train_iter, coords, params={'args': args}, device=args.gpu),trigger=(args.vis_freq, 'iteration'))
        # trainer.extend(Evaluator(train_iter, coords, params={'args': args}, device=args.gpu),trigger=(args.epoch, 'epoch'))
        # copy input DAG data file
        shutil.copyfile(args.input,os.path.join(args.outdir,os.path.basename(args.input)))
        # ChainerUI
        save_args(args, args.outdir)

    if args.optimizer in ['Momentum','CMomentum','AdaGrad','RMSprop','NesterovAG']:
        trainer.extend(extensions.ExponentialShift('lr', 0.5, optimizer=opt), trigger=(args.epoch/args.learning_rate_drop, 'epoch'))
    elif args.optimizer in ['Adam','AdaBound','Eve']:
        trainer.extend(extensions.ExponentialShift("alpha", 0.5, optimizer=opt), trigger=(args.epoch/args.learning_rate_drop, 'epoch'))

    trainer.run()



if __name__ == '__main__':
    main()
