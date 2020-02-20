Embedding of directed graphs
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
Each line corresponds to a chain denoted by a sequence of vertex indices.
Vertex ID should be integer from 0 to n-1

- to see command line options
```
python dgEmbedding.py -h
```

- toy examples

The following creates the "result" directory.

Disk embedding
```
    python dgEmbedding.py example/UK.csv -e 100 -be 3 -bv 3 -la 0 -ln 1 --dag 1
```

DANCAR embedding
```
    python dgEmbedding.py example/circle.csv -e 100 -be 3 -bv 3 -la 1 -ln 3 --dag 0
```


- Parallel learning using MPI
```
    mpiexec -n 4 python dgEmbedding.py example/circle.csv --mpi
```

- Random sample generation
```
python disk2dag.py -o outdir
```
