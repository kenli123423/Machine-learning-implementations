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
        if len(heap) < k:
            heapq.heappush(heap, (-dist, point))
        else:
            if -dist > heap[0][0]:
                heapq.heappop(heap)
                heapq.heappush(heap, (-dist, point))
    neighbors = [heapq.heappop(heap)[1] for _ in range(len(heap))]
    return neighbors[::-1]
dataset = [(1,2), (3,4), (5,6), (7,8)]
query_point = (2,3)
k=2
n = kNN(query_point, dataset, k, 2)
print(n)
