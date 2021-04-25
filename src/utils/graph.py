from typing import Tuple, Dict, Set, TypeVar, Generator, List, Generic, Mapping, Union
from collections import defaultdict

T = TypeVar("T")

# T_GRAPH[T] = Tuple[Set[T], Dict[T, List[T]]]

class Graph(Generic[T]):
    __slots__ = ('vertices', 'edges')

    def __init__(self):
        self.vertices: Set[T] = set()
        self.edges: Dict[T, Set[T]] = defaultdict(set)

    def add_v(self, v: T) -> None:
        self.vertices.add(v)

    def add_e(self, v1: T, v2: T) -> None:
        self.edges[v1].add(v2)

# walk along the graph
def dfs_walk(graph: Graph[T], v_start: T, visited: Set[T] = None) -> Generator[T, None, None]:
    visited = visited or set()
    visited.add(v_start)
    edges = graph.edges
    for neighbor in filter(lambda v: v not in visited, edges[v_start]):
        yield neighbor
        yield from dfs_walk(graph, neighbor, visited)
