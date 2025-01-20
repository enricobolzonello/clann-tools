from typing import NamedTuple, Callable
import numpy as np
from math import floor

def norm(a):
    return float(np.sum(a**2) ** 0.5)  

def euclidean_distance(x, y):
    return floor(np.linalg.norm(x - y))

def jaccard_distance(a: list[int], b: list[int]) -> float:
    if len(a) == 0 or len(b) == 0:
        return 0
    intersect = len(set(a) & set(b))
    return intersect / (float)(len(a) + len(b) - intersect)

def angular_distance(a, b):
    a = np.array(a) 
    b = np.array(b)  
    return float(1 - np.dot(a, b) / (norm(a) * norm(b)))

class Distance(NamedTuple):
    distance: Callable[[np.ndarray, np.ndarray], float]

metric_distances = {
    'euclidean' : Distance(
        distance=lambda a,b: euclidean_distance(a,b)
    ),
    'jaccard': Distance(
        distance=lambda a,b: jaccard_distance(a,b)
    ),
    'angular': Distance(
        distance=lambda a,b: angular_distance(a,b)
    )
}

def compute_distance(distance_name: str, a,b) -> float:
    if distance_name not in metric_distances:
        raise KeyError(f"Metric distance {distance_name} not defined. Known distances: {list(metric_distances.keys())}")
    return metric_distances[distance_name].distance(a,b)