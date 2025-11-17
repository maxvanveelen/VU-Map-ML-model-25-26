import time
import math
import numpy as np
import networkx as nx

class Trajectory:
    def __init__(self, path_graph, real_node_ids):
        self.history = []
        self.current_state = None
        self.path_graph = path_graph
        self.traj_graph = nx.Graph()
        self.real_node_ids = real_node_ids

    def update(self, x, y, z, timestamp):
        if timestamp is None:
            timestamp = time.time()

        if self.current_state is not None:
            self.history = self.history + [self.current_state]
        
        self.current_state = {
            "timestamp": timestamp,
            "raw": {
                "x": x,
                "y": y,
                "z": z
            },
            "matched": {
                "x": None,
                "y": None,
                "z": None,
                "segment": None,
                "distance": None
            }
        }

        self.match()

        # Add info to traj_graph
        node_id = self.real_node_ids[len(self.history)]
        pos = (self.current_state["raw"]["x"], self.current_state["raw"]["z"])
        self.traj_graph.add_node(node_id, pos=(pos), color="pink")
        if len(self.history) > 0:
            self.traj_graph.add_edge(self.real_node_ids[len(self.history) - 1], node_id, color="pink")

        pos = (self.current_state["matched"]["x"], self.current_state["matched"]["z"])
        self.traj_graph.add_node(f"M{node_id}", pos=(pos), color="yellow")
        self.traj_graph.add_edge(node_id, f"M{node_id}", color="yellow")

    def __str__(self):
        result = f"Trajectory(current_state={self.current_state},\n history=\n"

        for n in self.history:
            result += str(n)
            result += "\n"

        result += ")"

        return result
    
    def match(self):
        best_projection = None
        best_segment = None
        best_distance = math.inf

        for n1, n2 in self.path_graph.edges():
            d, proj = distance_point_to_segment(
                self.current_state["raw"]["x"], self.current_state["raw"]["z"],
                self.path_graph.nodes[n1]["pos"][0], self.path_graph.nodes[n1]["pos"][1],
                self.path_graph.nodes[n2]["pos"][0], self.path_graph.nodes[n2]["pos"][1],
            )

            if d < best_distance:
                best_projection = proj
                best_segment = (n1, n2)
                best_distance = d

        self.current_state["matched"]["x"] = best_projection[0]
        self.current_state["matched"]["y"] = None # TODO: y
        self.current_state["matched"]["z"] = best_projection[1]
        self.current_state["matched"]["segment"] = best_segment
        self.current_state["matched"]["distance"] = best_distance

    # Draw the graph on top of the image (traj_graph)
    def draw(self, ax):
        pos = nx.get_node_attributes(self.traj_graph, "pos")
        scale = 5538.46153846 # TODO: this number is just an estimate obtained by dividing the desired x value in the image by the actual x in the dataframe
        pos = {node: (x * scale, z * scale) for node, (x, z) in pos.items()}
        node_colors = [self.traj_graph.nodes[n]["color"] for n in self.traj_graph.nodes()]
        edge_colors = [self.traj_graph[u][v]["color"] for u, v in self.traj_graph.edges()]
        nx.draw(self.traj_graph, pos, with_labels=True, node_color=node_colors, edge_color=edge_colors, ax=ax)

# TODO: This function is from chatGPT
def distance_point_to_segment(px, py, x1, y1, x2, y2):
    # Convert to numpy arrays
    P = np.array([px, py])
    A = np.array([x1, y1])
    B = np.array([x2, y2])
    
    AB = B - A
    AP = P - A
    
    # Handle degenerate case where A and B are the same point
    if np.allclose(A, B):
        return np.linalg.norm(P - A)
    
    # Compute projection scalar t of AP onto AB, normalized to [0,1]
    t = np.dot(AP, AB) / np.dot(AB, AB)
    t = max(0, min(1, t))  # clamp t to segment
    
    # Closest point on the segment
    closest = A + t * AB
    
    # Distance from P to closest point, as well as the closest point itself
    return np.linalg.norm(P - closest), closest
