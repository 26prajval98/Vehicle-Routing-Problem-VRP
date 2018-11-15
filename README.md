# Vehicle Routing Problem (VRP)
Vehicle Routing Problem in CUDA-C++ (METHOD OF CLARKE AND WRIGHT)


Algorithm :

1. Starting solution: each of the n vehicles serves one customer.
2. For all pairs of nodes i,j, i…j, calculate the savings for joining the cycles using egde [i,j]:
s = c + c - c . ij 0i 0j ij
3. Sort the savings in decreasing order.
4. Take edge [i,j] from the top of the savings list. Join two separate cycles with edge [i,j], if
 (i) the nodes belong to separate cycles
 (ii) the maximum capacity of the vehicle is not exceeded
 (iii) i and j are first or last customer on their cycles.
5. Repeat 4 until the savings list is handled or the capacities don't allow more merging.
The cycles if i and j are NOT united in sep 4, if the nodes belong to the same cycle OR the
capacity is exceeded OR either node is an interior node of the cycle.
Improvement: Optimize the tour of each vehicle with a TSP heuristic

Time:

| Nodes | Serial | Parallel |
|-------|--------|----------|
| 5     | 42     | 318      |
| 10    | 345    | 343      |
| 50    | 3945   | 387      |
| 100   | 12686  | 412      |
|       |        |          |

Check http://www.mafy.lut.fi/study/DiscreteOpt/CH6.pdf
