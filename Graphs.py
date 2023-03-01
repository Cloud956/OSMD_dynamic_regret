import networkx as nx
import matplotlib.pyplot as plt

class Graph:

    def __init__(self):
        self.graph = {}
        self.edge_id = 0  # initialize the edge id counter

    def add_edge(self, u, v, weight):
        if u not in self.graph:
            self.graph[u] = {}
        if v not in self.graph:
            self.graph[v] = {}

        # assign a unique id to the edge
        if self.get_edge_id(u, v) is None:
            edge_id = self.edge_id
            self.edge_id += 1
        else:
            edge_id = self.get_edge_id(u, v)

        self.graph[u][v] = {'weight': weight, 'id': edge_id}
        self.graph[v][u] = {'weight': weight, 'id': edge_id}

    def remove_edge(self, u, v):
        if u in self.graph and v in self.graph[u]:
            del self.graph[u][v]
        if v in self.graph and u in self.graph[v]:
            del self.graph[v][u]

    def get_edge_weight(self, u, v):
        if u in self.graph and v in self.graph[u]:
            return self.graph[u][v].get('weight')
        return None
    
    def get_edge_id(self, u, v):
        if u in self.graph and v in self.graph[u]:
            return self.graph[u][v].get('id')
        return None

    def get_neighbors(self, u):
        if u in self.graph:
            return list(self.graph[u].keys())
        return []

    def num_vertices(self):
        return len(self.graph)
    
    def num_edges(self):
        num_edges = sum(len(self.graph[u]) for u in self.graph)
        return num_edges // 2

    def __str__(self):
        result = ""
        for u in self.graph:
            result += f"{u}: {self.graph[u]}\n"
        return result

class GridGraph(Graph):
    def __init__(self, n):
        super().__init__()
        self.n = n

        # create the vertices and edges
        for i in range(n):
            for j in range(n):
                vertex = i * n + j
                if j > 0:
                    self.add_edge(vertex, (vertex - 1), 1)
                if j < n-1:
                    self.add_edge(vertex, (vertex + 1), 1)
                if i > 0:
                    self.add_edge(vertex, (vertex - n), 1)
                if i < n-1:
                    self.add_edge(vertex, (vertex + n), 1)

    def __str__(self):
        return super().__str__()

class GraphVisualizer:
    def __init__(self, graph):
        self.graph = graph
        
    def visualize(self):
        # Create a new NetworkX graph
        nx_g = nx.Graph()

        # Add the edges and their weights to the NetworkX graph
        for u in self.graph.graph:
            for v in self.graph.graph[u]:
                w = self.graph.get_edge_weight(u, v)
                if w is not None:
                    nx_g.add_edge(u, v, weight=w)

        # Get the positions of the nodes for visualization
        pos = nx.spring_layout(nx_g)

        # Draw the nodes and edges of the NetworkX graph
        nx.draw_networkx_nodes(nx_g, pos)
        nx.draw_networkx_edges(nx_g, pos)
        nx.draw_networkx_edge_labels(nx_g, pos, edge_labels={(u, v): self.graph.get_edge_id(u, v) for u, v in nx_g.edges}) # change .get_id to .get_edge_weight
        nx.draw_networkx_labels(nx_g, pos, labels={node: str(node) for node in nx_g.nodes()})


        # Show the graph
        plt.axis('off')
        plt.show()

class GridGraphVisualizer:

    def __init__(self, graph):
        self.graph = graph

    def visualize(self):
        G = nx.Graph()
        # Add nodes with positions
        for i in range(self.graph.n):
            for j in range(self.graph.n):
                vertex = i * self.graph.n + j
                G.add_node(vertex, pos=(j, self.graph.n-i-1))  # Set position of node

        # Add edges
        for i in range(self.graph.n):
            for j in range(self.graph.n):
                vertex = i * self.graph.n + j
                if j > 0:
                    G.add_edge(vertex, vertex - 1, weight=(self.graph.get_edge_id(vertex, vertex-1), self.graph.get_edge_weight(vertex, vertex-1)))
                if j < self.graph.n-1:
                    G.add_edge(vertex, vertex + 1, weight=(self.graph.get_edge_id(vertex, vertex+1), self.graph.get_edge_weight(vertex, vertex+1)))
                if i > 0:
                    G.add_edge(vertex, vertex - self.graph.n, weight=(self.graph.get_edge_id(vertex, vertex-self.graph.n), self.graph.get_edge_weight(vertex, vertex-self.graph.n)))
                if i < self.graph.n-1:
                    G.add_edge(vertex, vertex + self.graph.n, weight=(self.graph.get_edge_id(vertex, vertex+self.graph.n), self.graph.get_edge_weight(vertex, vertex+self.graph.n)))

        # Draw the graph
        plt.figure()
        pos = nx.get_node_attributes(G, 'pos')  # Get node positions
        nx.draw(G, pos, with_labels=True, font_weight='bold')
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        plt.show()
