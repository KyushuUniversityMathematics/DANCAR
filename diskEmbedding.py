#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
## dataset preparation
class Dataset(dataset_mixin.DatasetMixin):
    def __init__(self, dat):
#        rawdat = pd.read_csv(fname, header=None)
#        self.edges = rawdat.iloc[:,[0,1]].values.astype(np.int32)
#        self.vertices = pd.concat([rawdat[0],rawdat[1]]).unique()
        self.edges = np.array(dat, dtype=np.int32)

    def __len__(self):
        return len(self.edges)

    def get_example(self, i):
        return self.edges[i,0],self.edges[i,1]

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

        # positive sample
        batch = self.get_iterator('main').next()
        a,b = self.converter(batch)
        d = F.sqrt(F.sum((self.coords.W[a,1:]-self.coords.W[b,1:])**2,axis=1)+epsilon)
        loss_pos = F.average(F.relu(d + self.args.margin + F.exp(self.coords.W[b,0]) - F.exp(self.coords.W[a,0])))
        chainer.report({'loss_pos': loss_pos}, self.coords)
#        print(d,loss_pos)
        loss = loss_pos

        # negative sample
        if self.args.lambda_neg>0:
            batch = self.get_iterator('negative').next()
            a,b = self.converter(batch)
            d = F.sqrt(F.sum((self.coords.W[a,1:]-self.coords.W[b,1:])**2,axis=1)+epsilon)
            loss_neg = F.average(F.relu(-d + self.args.margin - F.exp(self.coords.W[b,0]) + F.exp(self.coords.W[a,0])))
            chainer.report({'loss_neg': loss_neg}, self.coords)
            loss += self.args.lambda_neg * loss_neg

        # super negative sample
        if self.args.lambda_super_neg>0:
            batch = self.get_iterator('super_neg').next()
            a,b = self.converter(batch)
            d = F.sqrt(F.sum((self.coords.W[a,1:]-self.coords.W[b,1:])**2,axis=1)+epsilon)
            loss_super_neg = F.average(F.relu(-d + self.args.margin + F.exp(self.coords.W[b,0]) + F.exp(self.coords.W[a,0])))
            chainer.report({'loss_super_neg': loss_super_neg}, self.coords)
            loss += self.args.lambda_super_neg * loss_super_neg

        # coulomb force between points
        if self.args.lambda_coulomb>0:
            p = np.random.permutation(len(self.coords.W))
            d = F.sqrt(F.sum((self.coords.W[p[1:],1:]-self.coords.W[p[:-1],1:])**2,axis=1)+epsilon)
            loss_coulomb = F.average(F.relu(-d + self.args.margin + F.exp(self.coords.W[p[1:],0]) + F.exp(self.coords.W[p[:-1],0])))
#            loss_coulomb = -F.average((self.coords.W[p[1:],1:]-self.coords.W[p[:-1],1:])**2)
            chainer.report({'loss_coulomb': loss_coulomb}, self.coords)
            loss += self.args.lambda_coulomb * loss_coulomb
        
        # radius should be similar
        if self.args.lambda_uniform_radius>0:            
            loss_uniform_radius = F.average( (F.max(self.coords.W[:,0])-F.min(self.coords.W[:,0]))**2 )
            chainer.report({'loss_uniform_radius': loss_uniform_radius}, self.coords)
            loss += self.args.lambda_uniform_radius * loss_uniform_radius

        # update
        self.coords.cleargrads()
        loss.backward()
        opt.update(loss=loss)

# evaluator
class Evaluator(extensions.Evaluator):
    name = "myval"
    def __init__(self, *args, **kwargs):
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
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
        dat[:,0] = np.exp(dat[:,0])
        np.savetxt(os.path.join(self.args.outdir,"out{:0>4}.csv".format(self.count)), dat, fmt='%1.5f', delimiter=",")
        plot_all(dat,os.path.join(self.args.outdir,"plot{:0>4}.png".format(self.count)))
        self.count += 1
        return {"myval/none":0}


# plot results
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

# read graph from csv
def read_graph(fname):
    g = nx.DiGraph()
    g_trans = set()
    with open(fname) as infh:
        for line in infh:
            l = line.strip().split(',')
            g.add_edges_from([(l[i],l[i+1]) for i in range(len(l)-1)])
    pos_edge = []
    reachable = {}
    for v in g.nodes():
        reachable[v] = nx.descendants(g,v)
        for w in reachable[v]:
            pos_edge.append((v,w))
            g_trans.add((v,w))
        reachable[v].add(v)
        g_trans.add((v,v))
    neg_edge = []
    super_neg_edge = []
    for v in g.nodes():
        for w in g.nodes():
            if (v,w) not in g_trans:
                neg_edge.append((v,w))
                # pair of nodes with no common descendant
                if not (reachable[v] & reachable[w]):
                    super_neg_edge.append((v,w))
    print("#edges {}, #vertices {}".format(len(pos_edge),len(g.nodes())))
    return (len(g.nodes()),pos_edge,neg_edge,super_neg_edge)

## main
def main():
    # command line argument parsing
    parser = argparse.ArgumentParser(description='Disk Embedding')
    parser.add_argument('input', help='Path to DAG description file',default="split.csv")
    parser.add_argument('--batchsize', '-bs', type=int, default=1,
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
    parser.add_argument('--lambda_super_neg', '-lsn', type=float, default=1,
                        help='Super negative samples')
    parser.add_argument('--lambda_neg', '-ln', type=float, default=1,
                        help='negative samples')
    parser.add_argument('--lambda_coulomb', '-lc', type=float, default=0,
                        help='Coulomb force between points')
    parser.add_argument('--lambda_uniform_radius', '-lur', type=float, default=0,
                        help='Radius should be similar')
    parser.add_argument('--outdir', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--optimizer', '-op',choices=optim.keys(),default='Adam',
                        help='optimizer')
    parser.add_argument('--vis_freq', '-vf', type=int, default=2000,
                        help='visualisation frequency in iteration')
    parser.add_argument('--mpi', action='store_true',help='parallelise with MPI')
    args = parser.parse_args()

    args.outdir = os.path.join(args.outdir, dt.now().strftime('%m%d_%H%M'))
    save_args(args, args.outdir)

    chainer.config.autotune = True

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
    
    vnum,pos_edge,neg_edge,super_neg_edge = read_graph(args.input)
    train_iter = iterators.SerialIterator(Dataset(pos_edge), args.batchsize, shuffle=True)
    neg_train_iter = iterators.SerialIterator(Dataset(neg_edge), args.batchsize, shuffle=True)
    super_neg_train_iter = iterators.SerialIterator(Dataset(super_neg_edge), args.batchsize, shuffle=True)
    # initial embedding [1,2]^(dim+1),  the first coordinate corresponds to r (radius)
    coords = np.random.rand(vnum,args.dim+1)+1
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
        iterator={'main': train_iter, 'negative': neg_train_iter, 'super_neg': super_neg_train_iter},
        optimizer={'main': opt},
        device=args.gpu,
#        converter=convert.ConcatWithAsyncTransfer(),
        params={'args': args}
        )
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.outdir)

    if primary:
        log_interval = 100, 'iteration'
        log_keys = ['iteration','lr','elapsed_time','main/loss_pos', 'main/loss_neg', 'main/loss_super_neg', 'main/loss_coulomb','main/loss_uniform_radius']
        trainer.extend(extensions.observe_lr('main'), trigger=log_interval)
        trainer.extend(extensions.LogReport(keys=log_keys, trigger=log_interval))
        trainer.extend(extensions.PrintReport(log_keys), trigger=log_interval)
        if extensions.PlotReport.available():
            trainer.extend(extensions.PlotReport(log_keys[3:], 'epoch', file_name='loss.png'))
        trainer.extend(extensions.ProgressBar(update_interval=10))
        trainer.extend(extensions.LogReport(trigger=log_interval))
        trainer.extend(Evaluator(train_iter, coords, params={'args': args}, device=args.gpu),trigger=(args.vis_freq, 'iteration'))
        trainer.extend(Evaluator(train_iter, coords, params={'args': args}, device=args.gpu),trigger=(args.epoch, 'epoch'))
        trainer.extend(extensions.ParameterStatistics(coords))
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
