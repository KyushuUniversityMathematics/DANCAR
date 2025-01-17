#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import argparse
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training,datasets,iterators,Variable
from chainer.training import extensions
from chainer.dataset import dataset_mixin, convert, concat_examples, tabular
import pandas as pd
from chainerui.utils import save_args

import os,sys
from datetime import datetime as dt
from consts import optim,dtypes
import networkx as nx
import shutil
import random

from graphUtil import *

## updater 
class Updater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.coords = kwargs.pop('models')
        params = kwargs.pop('params')
        super(Updater, self).__init__(*args, **kwargs)
        self.args = params['args']
        self.graph = params['graph']
        
    def update_core(self):
        # TODO: 
        # log parametrisation of radius?

        epsilon = 1e-10
        opt = self.get_optimizer('main')
        r = F.relu(self.coords.W[:,0]) # radius
        x = self.coords.W[:,1:(self.args.dim+1)] # anchor
        c = self.coords.W[:,(self.args.dim+1):] # sphere centre
        a,b = self.converter(self.get_iterator('main').next()) # edge
        loss = 0

        # anchor loss
        if self.args.lambda_anchor > 0: # DANCAR
            v, = self.converter(self.get_iterator('anchor').next())
            loss_anc = F.average(F.relu(F.sqrt(F.sum((c[v]-x[v])**2,axis=1) + epsilon) - r[v] + self.args.margin))
            chainer.report({'loss_anc': loss_anc}, self.coords)
            loss += self.args.lambda_anchor * loss_anc
        else:
            x = c
            
        # positive sample: a contains b
        if self.args.lambda_pos > 0:
            d = F.sqrt(F.sum((c[a]-x[b])**2,axis=1)+epsilon)
            loss_pos = F.average(F.relu(self.args.margin + d + self.args.dag * r[b] - r[a]))
            chainer.report({'loss_pos': loss_pos}, self.coords)
            loss += self.args.lambda_pos*loss_pos

        # negative sample
        if self.args.lambda_neg>0:
            v, = self.converter(self.get_iterator('vertex').next())
            if self.args.batchsize_negative>0:
                na, nb = [],[]
                for u in v: # sample non-edges. accurate but slow
                    nnbor = set(nx.non_neighbors(self.graph, u))
                    for q in random.sample(nnbor, min(self.args.batchsize_negative,len(list(nnbor)))):
                        na.append(u)
                        nb.append(q)
                na = np.array(na)
                nb = np.array(nb)
            else:  # random vertex pairs
                na = v
                nb = np.roll(v,1)
            d = F.sqrt(F.sum((c[na]-x[nb])**2,axis=1)+epsilon)
            loss_neg = F.average(F.relu(self.args.margin - (d - r[na])))
#            loss_neg = F.average(F.relu(self.args.margin - (d + self.args.dag * r[nb] - r[na])))
            chainer.report({'loss_neg': loss_neg}, self.coords)
            loss += self.args.lambda_neg * loss_neg
        
        # radius should be similar
        if self.args.lambda_uniform_radius>0:            
            loss_uniform_radius = (F.max(r)-F.min(r))**2
            chainer.report({'loss_rad': loss_uniform_radius}, self.coords)
            loss += self.args.lambda_uniform_radius * loss_uniform_radius

        # update the coordinates
        self.coords.cleargrads()
        loss.backward()
        opt.update(loss=loss)
        self.coords.W.array[:,0] = self.coords.xp.clip(self.coords.W.array[:,0],a_min=0.01,a_max=None)

# evaluator
class Evaluator(extensions.Evaluator):
    name = "myval"
    def __init__(self, *args, **kwargs):
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        params = kwargs.pop('params')
        super(Evaluator, self).__init__(*args, **kwargs)
        self.args = params['args']
        self.graph = params['graph']
        self.count = 0
    def evaluate(self):
        coords = self.get_target('main')
        if self.eval_hook:
            self.eval_hook(self)
        if self.args.lambda_anchor == 0: # anchor = centre
            coords.W.array[:,1:(self.args.dim+1)] = coords.W.array[:,(self.args.dim+1):]
        if(self.args.gpu>-1):
            dat = coords.xp.asnumpy(coords.W.data).copy()
        else:
            dat = coords.W.data.copy()

        np.savetxt(os.path.join(self.args.outdir,"coords{:0>4}.csv".format(self.count)), dat, fmt='%1.5f', delimiter=",")
        redge = reconstruct(dat,dag=self.args.dag)
        if self.args.reconstruct:
            np.savetxt(os.path.join(self.args.outdir,"reconstructed{:0>4}.csv".format(self.count)),redge,fmt='%i',delimiter=",")
        if self.args.plot:
            plot_disks(dat,os.path.join(self.args.outdir,"plot{:0>4}.png".format(self.count)))
        self.count += 1

        # loss eval
        if self.args.validation:
            f1,precision,recall,accuracy = compare_graph(self.graph,nx.from_edgelist(redge,nx.DiGraph()),output=False)
            if self.args.lambda_anchor>0:
                anchor_violation, num_vert = check_anchor_containment(dat)
                anchor_ratio = anchor_violation/num_vert
            else:
                anchor_ratio = 0
            return {"myval/rec":recall,"myval/f1":f1, "myval/prc":precision, "myval/anc":anchor_ratio}
        else:
            return {"myval/none": 0}

def plot_log(f,a,summary):
    a.set_yscale('log')

## main
def main():
    # command line argument parsing
    parser = argparse.ArgumentParser(description='Digraph Embedding')
    parser.add_argument('input', help='Path to the digraph description file')
    parser.add_argument('--validation', '-val', default=None, help='Path to the digraph description file for validation')
    parser.add_argument('--coordinates', '-c', help='Path to the coordinate file for initialization')
    parser.add_argument('--batchsize_edge', '-be', type=int, default=100,
                        help='Number of samples in each edge mini-batch')
    parser.add_argument('--batchsize_anchor', '-ba', type=int, default=-1,
                        help='Number of samples in each anchor mini-batch')
    parser.add_argument('--batchsize_vert', '-bv', type=int, default=-1,
                        help='Number of samples in each vertex mini-batch (used for sampling negative edges)')
    parser.add_argument('--batchsize_negative', '-bn', type=int, default=0,
                        help='Number of negative edges sampled for each vertex mini-batch (positive: exact negative edge sampling, negative: random sampling to approximate negative edges)')
    parser.add_argument('--vertex_offset', type=int, default=0, help='the smallest index of vertices')
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--dim', '-d', type=int, default=2,
                        help='Embedding dimension')
    parser.add_argument('--dag', type=float, default=0, help='0:non-acyclic, 1:acyclic')
    parser.add_argument('--margin', '-m', type=float, default=0.01,
                        help='margin for the metric boundary')
    parser.add_argument('--weight_decay', '-wd', type=float, default=0,
                        help='weight decay for regularization on coordinates')
    parser.add_argument('--wd_norm', '-wn', choices=['l1','l2'], default='l2',
                        help='norm of weight decay for regularization')
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-2,
                        help='learning rate')
    parser.add_argument('--learning_rate_drop', '-ld', type=int, default=5,
                        help='how many times to half learning rate')
#    parser.add_argument('--lambda_super_neg', '-lsn', type=float, default=0,
#                        help='Super negative samples')
    parser.add_argument('--lambda_pos', '-lp', type=float, default=1,
                        help='weight for loss for positive edges')
    parser.add_argument('--lambda_neg', '-ln', type=float, default=1,
                        help='weight for loss for negative edges')
    parser.add_argument('--lambda_anchor', '-la', type=float, default=1,
                        help='anchor should reside in the disk. if set to 0, anchors are fixed to the centre of the spheres')
    parser.add_argument('--lambda_uniform_radius', '-lur', type=float, default=0,
                        help='all radiuses should be similar')
    parser.add_argument('--outdir', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--optimizer', '-op',choices=optim.keys(),default='Adam',
                        help='optimizer')
    parser.add_argument('--vis_freq', '-vf', type=int, default=-1,
                        help='evaluation frequency in iteration')
    parser.add_argument('--mpi', action='store_true',help='parallelise with MPI')
    parser.add_argument('--reconstruct', '-r', action='store_true',help='reconstruct graph during evaluation')
    parser.add_argument('--plot', '-p', action='store_true',help='plot result (dim=2 only)')
#    parser.add_argument('--training', '-t', action='store_false',help='reconstruct graph')
    args = parser.parse_args()

    # default batchsize
    if args.batchsize_anchor < 0:
        args.batchsize_anchor = 10*args.batchsize_edge
    if args.batchsize_vert < 0:
        if args.batchsize_negative == 0:
            args.batchsize_vert = 10*args.batchsize_edge
        else:
            args.batchsize_vert = args.batchsize_edge

    args.outdir = os.path.join(args.outdir, dt.now().strftime('%m%d_%H%M'))
    save_args(args, args.outdir)
    chainer.config.autotune = True

    vert,pos_edge=read_graph(args.input,args.vertex_offset)
    vnum = np.max(vert)+1

    ## ChainerMN
    if args.mpi:
        import chainermn
        if args.gpu >= 0:
            comm = chainermn.create_communicator()
            chainer.cuda.get_device(comm.intra_rank).use()
        else:
            comm = chainermn.create_communicator('naive')
        if comm.rank == 0:
            primary = True
            print(args)
            chainer.print_runtime_info()
            print("#edges {}, #vertices {}".format(len(pos_edge),len(vert)))
        else:
            primary = False
        print("process {}".format(comm.rank))
    else:
        primary = True
        print(args)
        chainer.print_runtime_info()
        print("#edges {}, #vertices {}".format(len(pos_edge),len(vert)))
        if args.gpu >= 0:
            chainer.cuda.get_device(args.gpu).use()

    # data
    edge_iter = iterators.SerialIterator(datasets.TupleDataset(pos_edge[:,0],pos_edge[:,1]), args.batchsize_edge, shuffle=True)
    vert_iter = iterators.SerialIterator(datasets.TupleDataset(vert), args.batchsize_vert, shuffle=True)
    anchor_iter = iterators.SerialIterator(datasets.TupleDataset(vert), args.batchsize_anchor, shuffle=True)
    graph = nx.from_edgelist(pos_edge,nx.DiGraph())
    if args.validation and primary:
        val_vert,val_edge=read_graph(args.validation,args.vertex_offset)
        val_graph = nx.from_edgelist(val_edge,nx.DiGraph())
        print("validation #edges {}, #vertices {}".format(len(val_edge),len(val_vert)))
    else:
        val_graph = graph

    if args.vis_freq < 0:
        args.vis_freq = int(len(pos_edge)*args.epoch/10)

    # initial embedding
    if args.coordinates:
        coords = np.loadtxt(args.coordinates,delimiter=",")
    else:
        coords = np.zeros( (vnum,1+2*args.dim) )
        # anchor = centre
        X = 2*np.random.rand(vnum,args.dim)-1
        coords[:,1:args.dim+1] = X
        coords[:,args.dim+1:] = X
        # the first coordinate corresponds to the radius r=0.1
        coords[:,0] = 0.1
    coords = L.Parameter(coords)
    
    # set up an optimizer
    def make_optimizer(model):
        if args.optimizer in ['SGD','Momentum','CMomentum','AdaGrad','RMSprop','NesterovAG','LBFGS']:
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
        iterator={'main': edge_iter, 'vertex': vert_iter, 'anchor': anchor_iter},  
        optimizer={'main': opt},
        device=args.gpu,
#        converter=convert.ConcatWithAsyncTransfer(),
        params={'args': args, 'graph': graph}
        )
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.outdir)

    if primary:
        log_interval = 20, 'iteration'
        log_keys = ['iteration','lr','elapsed_time','main/loss_pos', 'main/loss_neg','main/loss_anc']
        if args.validation:
            log_keys.extend(['myval/prc','myval/rec','myval/f1','myval/anc'])
        if args.lambda_uniform_radius>0:
            log_keys.append('main/loss_rad')
        trainer.extend(extensions.observe_lr('main'), trigger=log_interval)
        trainer.extend(extensions.LogReport(keys=log_keys, trigger=log_interval))
#        trainer.extend(extensions.LogReport(keys=log_keys, trigger=log_interval))
        trainer.extend(extensions.PrintReport(log_keys), trigger=log_interval)
#        trainer.extend(extensions.PrintReport(log_keys), trigger=(1, 'iteration'))
        if extensions.PlotReport.available():
            trainer.extend(extensions.PlotReport(log_keys[3:], 'epoch', file_name='loss.png',postprocess=plot_log))
        trainer.extend(extensions.ProgressBar(update_interval=10))
        trainer.extend(extensions.LogReport(trigger=log_interval))
        trainer.extend(extensions.snapshot_object(opt, 'opt{.updater.epoch}.npz'), trigger=(args.epoch, 'epoch'))
        if args.vis_freq>0:
            trainer.extend(Evaluator({'main': edge_iter}, coords, params={'args': args,'graph': val_graph}, device=args.gpu),trigger=(args.vis_freq, 'iteration'))
#        trainer.extend(extensions.ParameterStatistics(coords))

        # ChainerUI
        save_args(args, args.outdir)

    if args.optimizer in ['Momentum','CMomentum','AdaGrad','RMSprop','NesterovAG']:
        trainer.extend(extensions.ExponentialShift('lr', 0.5, optimizer=opt), trigger=(args.epoch/args.learning_rate_drop, 'epoch'))
    elif args.optimizer in ['Adam','AdaBound','Eve']:
        trainer.extend(extensions.ExponentialShift("alpha", 0.5, optimizer=opt), trigger=(args.epoch/args.learning_rate_drop, 'epoch'))

#    if args.training:
    trainer.run()

    # result
    if primary:
        # save DAG data file
        if(args.gpu>-1):
            dat = coords.xp.asnumpy(coords.W.data)
        else:
            dat = coords.W.data
        if args.lambda_anchor == 0: # anchor = centre
            dat[:,1:(args.dim+1)] = dat[:,(args.dim+1):]
        redge = reconstruct(dat,dag=args.dag)
        np.savetxt(os.path.join(args.outdir,"original.csv"),pos_edge,fmt='%i',delimiter=",")
        np.savetxt(os.path.join(args.outdir,"reconstructed.csv"),redge,fmt='%i',delimiter=",")
        np.savetxt(os.path.join(args.outdir,"coords.csv"), dat, fmt='%1.5f', delimiter=",")
        f1,prc,rec,acc = compare_graph(val_graph,nx.from_edgelist(redge,nx.DiGraph()))
        if args.plot:
            plot_digraph(pos_edge,os.path.join(args.outdir,"original.png"))
            plot_digraph(redge,os.path.join(args.outdir,"reconstructed.png"))
            plot_disks(dat,os.path.join(args.outdir,"plot.png"))
        with open(os.path.join(args.outdir,"args.txt"), 'w') as fh:
            fh.write(" ".join(sys.argv))
            fh.write(f"f1: {f1}, precision: {prc}, recall: {rec}, accuracy: {acc}")

if __name__ == '__main__':
    main()
