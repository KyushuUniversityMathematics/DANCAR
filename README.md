Embedding of directed graphs
=============
Written by Shizuo KAJI

This code computes an embedding of a directed graph 
(that is, it associates vectors for each vertex) so that the existence of an edge
is modeled by the geometry.

More precisely, each vertex is represented by a ball and a point (called the anchor) contained in the ball.
That is, each vertex is identified by
- the coordinates of the centre in R^n
- the radius
- the coordinates of the anchor in R^n
where R^n is the n-dimensional Euclidean space.
A directed edge (u,v) is modeled by the relation that the anchor of v is contained in the ball of u.

For details, look at 
- N. Hata, S. Kaji, A. Yoshida, and K. Fujisawa, Nested Subspace Arrangement for Representation of Relational Data, ICML2020.

## License
MIT License

## Requirements
- Python 3: [Anaconda](https://www.anaconda.com/download/) is recommended
- Python libraries: Chainer, chainerui:  `pip install chainer chainerui`
- for parallel learning: mpi4py: `conda install mpi4py` or `pip install mpi4py` 

# How to use
- The input text file describing a directed graph should look as follows:
```
0,1,2,3,4
3,5,6
```
Each line corresponds to a chain of directed edges denoted by a sequence of vertex indices.
Vertex ID should be integer from 0 to n-1

- to see command line options
```
python dgEmbedding.py -h
```

- Toy examples

The following creates the "result" directory, where outputs are stored.

DANCAR embedding
```
    python dgEmbedding.py example/UK.csv -e 100 -be 3 -bv 3 -bn 1 -ln 1 -la 1 --dag 0 -p
```

Disk embedding
```
    python dgEmbedding.py example/UK.csv -e 100 -be 3 -bv 3 -bn 1 -ln 1 -lur 0.5 -la 0 --dag 1 -p
```
For this simple acyclic example, Disk embedding produces visually more pleasing results.

- Parallel learning using MPI
To reproduce the result with WordNet described in the paper:
```
    mpiexec -n 10 python dgEmbedding.py example/wordnet_sorted.csv --mpi -be 10000 -ln 2000 -d 10 -m 0.01 -ld 10 --epoch 1000 -val example/wordnet_sorted.csv
```

- Result
The coordinates are saved to the file "coords.csv". 
Each line represents a vertex. The first entry is the radius. 
If the embedding dimension is d,
next d entries are the coordinates of the anchor, and the last d entries are those of the centre.

Reconstructed graph edges are stored in the file "reconstructed.csv".

We can also reconstruct edges from coordinates saved in "coords.csv" and store them in "reconstruct.csv" by
```
    python graphUtil.py -c coords.csv -r reconstructed.csv
```

To compare the original and the reconstructed graphs,
```
    python graphUtil.py -i original.csv -r reconstructed.csv
```
