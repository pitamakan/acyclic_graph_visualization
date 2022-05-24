import argparse
import networkx
from matplotlib import image
from matplotlib import pyplot as plt
from scipy.optimize import linprog

class Graph:
    def __init__(self, input_file):
        input = networkx.read_graphml(input_file)

        name_to_ind = {name: ind for ind, name in enumerate(input.nodes())}
        self.size = len(name_to_ind)

        self.incoming_edges = {name_to_ind[k]: [] for k in input.nodes()}
        self.outgoing_edges = {name_to_ind[k]: [] for k in input.nodes()}
        self.dummy = set()
        self.layers = []

        for source, adjacencies in input.adjacency():
            for target in adjacencies:
                self.incoming_edges[name_to_ind[target]].append(name_to_ind[source])
                self.outgoing_edges[name_to_ind[source]].append(name_to_ind[target])

    def get_size(self):
        return self.size

    def is_dummy(self, node):
        return node in self.dummy

    def get_index(self, node):
        return self.indexes(node)

    def get_outgoing_edges(self, node):
        return self.outgoing_edges[node]

    def get_incoming_edges(self, node):
        return self.incoming_edges[node]

    def split_edge(self, source, target):
        index = self.size
        self.size += 1
        self.incoming_edges[index] = [source]
        self.outgoing_edges[index] = [target]
        self.dummy.add(index)

        self.outgoing_edges[source][self.outgoing_edges[source].index(target)] = index
        self.incoming_edges[target][self.incoming_edges[target].index(source)] = index
        return index

    def print_graph(self, layers, output_file, show=False):
        coords = {}
        max_layer = max([len(layer) for layer in layers])
        for (i, layer) in enumerate(layers):
            if i > 0:
                median_position = [(u, sum([coords[v][0] for v in self.get_incoming_edges(u)])) for u in layer]
                layer = [x[0] for x in sorted(median_position, key=lambda tup: tup[1])]

            for (num, u) in enumerate(layer):
                coords[u] = (num + (max_layer - len(layer)) / 2, i)


        for source in range(self.get_size()):
            for target in self.get_outgoing_edges(source):
                plt.plot(
                    (-coords[source][0], -coords[target][0]),
                    (-coords[source][1], -coords[target][1]),
                    color='blue', linewidth=3)
        for (node, (x, y)) in coords.items():
            if not self.is_dummy(node):
                plt.plot(-x, -y, marker="o", color="r", markersize=12, markeredgecolor="black")

        plt.axis('off')
        if output_file is not None:
            plt.savefig(output_file, dpi=100)
        if show:
            plt.show()

def simple_algo(graph):
    size = graph.get_size()
    A = []
    for source in range(size):
        for target in graph.get_outgoing_edges(source):
            row = [0] * size
            row[source] = 1
            row[target] = -1
            A.append(row)
    b = [-1] * len(A)
    c = [len(graph.get_incoming_edges(v)) - len(graph.get_outgoing_edges(v)) for v in range(size)]

    answer = linprog(c, A, b, bounds=(0, size - 1), method="revised simplex")
    layers = []
    for (v, layer) in enumerate(list(map(int, answer.x))):
        while len(layers) <= layer:
            layers.append([])
        layers[layer].append(v)
    return layers

def less(first, second):
    for u, v in zip(sorted(first, reverse=True), sorted(second, reverse=True)):
        if u < v:
            return True
        if v < u:
            return False
    if len(first) < len(second):
        return True
    return False

def initialize_pi(graph):
    size = graph.get_size()
    pi = [0] * size
    for pi_value in range(1, size + 1):
        min_index, min_set = None, None
        for target in range(size):
            if pi[target] > 0:
                continue
            target_set = set()
            for source in graph.get_incoming_edges(target):
                if pi[source] != 0:
                    target_set.add(source)
            if min_set is None or less(target_set, min_set):
                min_index, min_set = target, min_set
        pi[min_index] = pi_value
    return pi

def coffman_graham(graph, w):
    size = graph.get_size()
    pi = initialize_pi(graph)
    U = set()
    layers = [[]]
    while len(U) != size:
        best_u = None
        for u in range(size):
            if u in U:
                continue
            if len([v for v in graph.get_outgoing_edges(u) if v not in U]) != 0:
                continue
            if best_u is None or pi[u] > pi[best_u]:
                best_u = u
        if len(layers[-1]) >= w or len([v for v in graph.get_outgoing_edges(best_u) if v in layers[-1]]) != 0:
            layers.append([])
        layers[-1].append(best_u)
        U.add(best_u)
    layers.reverse()
    return layers

def add_dummies(layers, graph):
    size = graph.get_size()
    layer_num = [0] * size
    for (num, layer) in enumerate(layers):
        for v in layer:
            layer_num[v] = num
    for u in range(size):
        for v in graph.get_outgoing_edges(u):
            if (graph.is_dummy(v)):
                continue
            if layer_num[u] + 1 != layer_num[v]:
                start = u
                for layer in range(layer_num[u] + 1, layer_num[v]):
                    dummy = graph.split_edge(start, v)
                    layers[layer].append(dummy)
                    start = dummy
    return layers, graph

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, type=str, help='Input graphml file')
    parser.add_argument('-o', '--output', type=str, help='Output .png, .jpg or .jpeg file')
    parser.add_argument('-w', '--width', type=int, help='Max layer width')
    parser.add_argument('-s', '--show', action=argparse.BooleanOptionalAction, help='Show with plt')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    graph = Graph(args.input)

    layers = None
    if args.width is None:
        layers = simple_algo(graph)
    else:
        layers = coffman_graham(graph, args.width)
    layers, graph = add_dummies(layers, graph)
    graph.print_graph(layers, args.output, args.show)
