"""
This is an implementation of kNN using heap, it has a computational complexity of O(nlogk).
"""

import numpy as np 
import heapq

def distance(x_test, x, p):
    return sum((p1 - p2)**p for p1, p2 in zip(x_test, x))**(1/p)

def kNN(query_point, dataset, k, p):
    """
    query_point : centre which used to find the distance between this point and all other points.
    dataset : set of points.
    k : k from k-NN.
    p : p from Minkowski distance.
    """
    heap = []
    for point in dataset:
        dist = distance(query_point, point, p)
        #Calculate the distance of query_point to another point in the dataset.
        if len(heap) < k:
            #if the heap has less than k elements, add points from dataset into the heap, using negative distance to ensure max-heap property beacuse python default uses min-heap.
            heapq.heappush(heap, (-dist, point))
        else:
            #Root node has the largest distance, if any -dist larger than root node, pop the current root node and replace with new -dist.
            #Push the tuple (-dist, point) from bottom to the root node and sort the heap automatically the ensure max-heap property.
            if -dist > heap[0][0]:
                heapq.heappop(heap)
                heapq.heappush(heap, (-dist, point))
    #Extract neighbours once all points are added and sorted.
    neighbors = [heapq.heappop(heap)[1] for _ in range(len(heap))]
    return neighbors[::-1]
    
dataset = [(1,1), (1,2), (3,4), (5,6), (7,8)]
y = [1,1,1,-1,-1]
query_point = (2,3)
k=2
n = kNN(query_point, dataset, k, 2)
c = [y[dataset.index(x)] for x in n]
true_c = max(set(c), key=c.count)
print(f"Class of query point : {true_c}")
