Disk Embedding of DAG
=============

## Requirements
- Python 3: [Anaconda](https://www.anaconda.com/download/) is recommended
- Python libraries: Chainer, chainerui:  `pip install chainer chainerui`
- for parallel learning: mpi4py: `conda install mpi4py` or `pip install mpi4py` 

# How to use
- data file specification
e.g.
```
0,1,2,3,4
3,5,6
```
A line corresponds to a chain denoted by a sequence of vertex indices.
Vertex ID should be integer from 0 to n-1

- command line options
```
python diskEmbedding.py -h
```

- Parallel learning
```
    mpiexec -n 4 python diskEmbedding.py dag.csv --mpi
```

- Random sample generation
```
python disk2dag.py -o outdir
```
