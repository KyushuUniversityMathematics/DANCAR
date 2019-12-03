before  = sorted({tuple(map(int,s.strip().split(","))) for s in open("wordnet.csv")})
nodes = {i[0] for i in before} | {i[1] for i in before}
indices = {x:i for i,x in sorted(nodes)}
after = [(indices[x],indices[y]) for x,y in before]
f = open("wordnet_sorted.csv","w")
g = open("wordnet_index.csv","w")

for i,j in after:
    print(f"{i},{j}",file=f)

for i,x in indices.items():
    print(f"{i},{x}",file=g)

f.close()
g.close()