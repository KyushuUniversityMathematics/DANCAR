from random import random, sample, randint
import argparse
def main():

    V = 30
    parser = argparse.ArgumentParser(description='Disk Embedding')
    parser.add_argument('-n', help='#nodes',type=int, default=V)
    parser.add_argument('-o', '--output', help='output filename',default="random_graph.csv")
    parser.add_argument('-p', '--probability', help='',type=float, default=0.3)

    args = parser.parse_args()

    prob = args.probability
    V    = args.n
    filename = args.output

    edges = {i:set() for i in range(V)}
    preds = {i:set() for i in range(V)}
    for i in range(V):
        candidates_i = set(range(i+1,V))
        for j in candidates_i:
            if random() < prob:
                edges[i].add(j)
                preds[j].add(i)
        if edges[i] == set() == preds[i]:
            j = sample(set(range(V)) - {i}, 1)
            i, j = sorted([i,j])
            edges[i].add(j)
            preds[j].add(i)

    f = open(filename, "w")
    for i in edges:
        for j in edges[i]:
            print(f"{i},{j}", file=f)
    f.close()

if __name__ == "__main__":
    main()