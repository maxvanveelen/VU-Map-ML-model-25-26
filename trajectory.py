import time
import math
import numpy as np
import networkx as nx
from typing import List
from rtree import index
from shapely.geometry import Point, LineString
from scipy.spatial import KDTree

SIGMA = 0.06                # TODO should be the std dev of the error of the rssi localization, this is an estimate of that.
BETA = 0.5                  # TODO: TUNE THIS CORRECTLY
RADIUS = 0.1                # TODO: How far to look for road segments for candidates.
SIZE_SLIDING_WINDOW = 5     # TODO: How many previous observations to consider when calculating transition probability.

class Candidate:
    # edge_id: TODO
    # u: TODO
    # v: TODO
    # projection: TODO
    # distance: TODO
    # geom: TODO
    # log_emission: TODO
    # rssi_point: TODO
    # is_best: TODO

    def __init__(self):
        edge_id = None
        u = None
        v = None
        projection = None
        distance = None
        geom = None
        log_emission = None
        rssi_point = None
        is_best = False

    def __str__(self):
        result = [
            f"Candidate(\n",
            f"          edge_id      = {self.edge_id}\n",
            f"          u            = {self.u}\n",
            f"          v            = {self.v}\n",
            f"          projection   = {self.projection}\n",
            f"          distance     = {self.distance}\n",
            f"          geom         = {self.geom}\n",
            f"          log_emission = {self.log_emission}\n",
            f"          rssi_point   = {self.rssi_point}\n",
            f"          is_best      = {self.is_best}\n",
            f"         )\n"
        ]
        return "".join(result)

class Trajectory:
    history: List[dict]
    current_state: dict
    path_graph: nx.Graph
    traj_graph: nx.Graph
    real_node_ids: List[int]    # used for drawing visual

    edge_spatial_index: index.Index            # used to find candidate edges
    edge_geoms : dict

    kdtree : KDTree
    node_ids: list[int]         # used to search KD-Tree

    def __init__(self, path_graph, real_node_ids):
        self.history = []
        self.current_state = None
        self.path_graph = path_graph
        self.traj_graph = nx.Graph()
        self.real_node_ids = real_node_ids

        self.edge_spatial_index = None
        self.edge_geoms = None
        self.init_spatial_index()

        self.kdtree, self.node_ids = self.init_KD_Tree()

    def __str__(self):
        result = f"Trajectory(current_state={self.current_state},\n history=\n"

        for n in self.history:
            result += str(n)
            result += "\n"

        result += ")"

        return result
    

    def init_spatial_index(self):
        """creates the spatial index (edge_spatial_index) and lookup dict (edge_geoms) for the edges of the path graph"""

        edge_spatial_index = index.Index()
        edge_geoms = {}
        edge_id = 0

        for u, v, data in self.path_graph.edges(data=True):
            xu, yu = self.path_graph.nodes[u]["pos"]
            xv, yv = self.path_graph.nodes[v]["pos"]

            geom = LineString([(xu, yu), (xv, yv)])
            edge_geoms[edge_id] = (u, v, geom)
            edge_spatial_index.insert(edge_id, geom.bounds)

            edge_id += 1

        self.edge_spatial_index = edge_spatial_index
        self.edge_geoms = edge_geoms

    def init_KD_Tree(self):
        """
        Stores node coordinates in a KD-tree, to efficiently search for the nearest node to a position.
        Returns both the KD-tree and the list of node ids for searching the KD-tree.
        """

        node_ids = []
        node_coords = []

        for n, data in self.path_graph.nodes(data=True):
            x, y = data["pos"]
            node_ids.append(n)
            node_coords.append((x, y))

        node_coords = np.array(node_coords)
        return KDTree(node_coords), node_ids

    def update(self, x, y, z, timestamp):
        if timestamp is None:
            timestamp = time.time()

        # if self.current_state is not None:
            # self.history = self.history + [self.current_state]
        
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
            },
            "candidates": None
        }

        #TODO history now includes the current state, this may cause bugs in old code
        self.history = self.history + [self.current_state]

        # self.match()
        self.match_HMM_Viterbi()

        # Add info to traj_graph
        node_id = self.real_node_ids[len(self.history) - 1]
        pos = (self.current_state["raw"]["x"], self.current_state["raw"]["z"])
        self.traj_graph.add_node(node_id, pos=(pos), color="pink")
        if len(self.history) > 1:
            self.traj_graph.add_edge(self.real_node_ids[len(self.history) - 2], node_id, color="pink")

        for i, c in enumerate(self.current_state["candidates"]):

            print(c)

            pos = (c.projection.x, c.projection.y)
            if c.is_best:
                self.traj_graph.add_node(f"C{node_id}_{i}", pos=(pos), color="green")
                self.traj_graph.add_edge(node_id, f"C{node_id}_{i}", color="green")
            else:
                self.traj_graph.add_node(f"C{node_id}_{i}", pos=(pos), color="yellow")
                self.traj_graph.add_edge(node_id, f"C{node_id}_{i}", color="yellow")
    
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

    def match_HMM_Viterbi(self):
        """
        Does the HMM_Viterbi map matching of the current state. Adds the matched properties to the current state and updates the current state in the history to include these.
        Also returns the final path through the considered observations (sliding window) TODO: maybe useful later for navigation?
        """

        self.generate_candidates(self.current_state, RADIUS)

        observations = self.history[-SIZE_SLIDING_WINDOW:]

        T = len(observations)
        
        # V[t][c] = best log-probability ending at candidate c at time t
        # BP[t][c] = best previous candidate leading to c
        V = [{} for _ in range(T)]
        BP = [{} for _ in range(T)]
        
        for c in observations[0]["candidates"]:
            V[0][c] = c.log_emission
            BP[0][c] = None

        # Viterbi recursion
        for t in range(1, T):

            for c in observations[t]["candidates"]:
                best_score = -math.inf
                best_prev = None

                log_emission = c.log_emission

                for p in observations[t - 1]["candidates"]:
                    prev_score = V[t - 1].get(p, -math.inf)
                    if prev_score == -math.inf:
                        continue

                    log_transition = self.calc_log_transition_probability(p, c, BETA)
                
                    score = prev_score + log_emission + log_transition

                    if score > best_score:
                        best_score = score
                        best_prev = p

                if best_prev is not None:
                    V[t][c] = best_score
                    BP[t][c] = best_prev
                else:
                    V[t][c] = -math.inf
                    BP[t][c] = None

        # save best final candidate for current state
        last_t = T - 1
        best_final_candidate = max(V[last_t], key=V[last_t].get)

        # Backtracking: reconstruct most likely path through considered observations (sliding window).
        path = [None] * T
        path[last_t] = best_final_candidate

        for t in range(last_t, 0, -1):
            path[t - 1] = BP[t][path[t]]

        best_final_candidate.is_best = True # TODO: may conflict with returned path occasionally, but fine for now

        self.current_state["matched"]["x"] = best_final_candidate.projection.x
        self.current_state["matched"]["y"] = None # TODO: y?
        self.current_state["matched"]["z"] = best_final_candidate.projection.y
        self.current_state["matched"]["segment"] = (best_final_candidate.u, best_final_candidate.v)
        self.current_state["matched"]["distance"] = best_final_candidate.distance

        # set last item in history to the current state, now including the matched properties
        self.history[-1] = self.current_state

        return path

    # Draw the graph on top of the image (traj_graph)
    def draw_traj(self, ax):
        pos = nx.get_node_attributes(self.traj_graph, "pos")
        scale = 5538.46153846 # TODO: this number is just an estimate obtained by dividing the desired x value in the image by the actual x in the dataframe
        pos = {node: (x * scale, z * scale) for node, (x, z) in pos.items()}
        node_colors = [self.traj_graph.nodes[n]["color"] for n in self.traj_graph.nodes()]
        edge_colors = [self.traj_graph[u][v]["color"] for u, v in self.traj_graph.edges()]
        nx.draw(self.traj_graph, pos, with_labels=True, node_color=node_colors, edge_color=edge_colors, ax=ax)

    def generate_candidates(self, observation, radius):
        """Generates candidates given an observation (current_state) and a radius"""

        rssi_point = Point(observation["raw"]["x"], observation["raw"]["z"])

        # create circular search area
        search_area = rssi_point.buffer(radius)

        # first: get candidate edges via bbox
        candidate_edge_ids = list(self.edge_spatial_index.intersection(search_area.bounds))

        # second: filter for actual distance (guaranteed within radius)
        guaranteed_edge_ids = []
        for edge_id in candidate_edge_ids:
            geom = self.edge_geoms[edge_id][2]  # extract LineString
            if geom.intersects(search_area):
                guaranteed_edge_ids.append(edge_id)

        candidates = []

        for eid in guaranteed_edge_ids:
            u, v, geom = self.edge_geoms[eid]

            proj_dist = geom.project(rssi_point)
            proj_point = geom.interpolate(proj_dist)

            dist = rssi_point.distance(proj_point)

            c = Candidate()
            c.edge_id = eid
            c.u = u
            c.v = v
            c.projection = proj_point
            c.distance = dist 
            c.geom = geom
            c.log_emission = self.calc_log_emission_probability(c.distance, SIGMA)
            c.rssi_point = rssi_point
            c.is_best = False

            candidates.append(c)

        observation["candidates"] = candidates
        
    def calc_log_emission_probability(self, distance, sigma):
        """Calculates log emission probability given a distance and a sigma (constant)"""
        return -(distance*distance) / (2 * sigma * sigma)

    def calc_log_transition_probability(self, cand1, cand2, beta):
        """Calculates log transition probability given two candidates and a beta (constant)"""
        straight_line_distance = cand1.rssi_point.distance(cand2.rssi_point)

        cand1_nearest_node, cand1_dist_to_nearest_node = nearest_node(cand1.projection, self.kdtree, self.node_ids)
        cand2_nearest_node, cand2_dist_to_nearest_node = nearest_node(cand2.projection, self.kdtree, self.node_ids)

        try:
            shortest_path = nx.shortest_path_length(self.path_graph, source=cand1_nearest_node, target=cand2_nearest_node, weight="distance")
        except nx.NetworkXNoPath:
            return -1e9  # impossible transition
        
        route_distance = cand1_dist_to_nearest_node + shortest_path + cand2_dist_to_nearest_node

        difference = route_distance - straight_line_distance
        return -(difference * difference) / (2 * beta * beta)

# NOTE: 
# This function was originally written by chatGPT.
# Comments have been altered for my own understanding, but the code remains the same.
def distance_point_to_segment(px, py, xa, ya, xb, yb):
    """
    Returns the distance between a point and the closest point on a line segment,
    and the coordinates of that closest point itself.
    
    :param px: x coordinate of the point
    :param py: y coordinate of the point
    :param xa: x coordinate of the first end of the segment
    :param ya: y coordinate of the first end of the segment
    :param xb: x coordinate of the second end of the segment
    :param yb: y coordinate of the second end of the segment
    """

    # create vectors AB and AP
    P = np.array([px, py])
    A = np.array([xa, ya])
    B = np.array([xb, yb])
    AB = B - A
    AP = P - A
    
    # handle case where A and B are the same
    if np.allclose(A, B):
        return np.linalg.norm(P - A), A
    
    # compute projection scalar t of AP onto AB
    t = np.dot(AP, AB) / np.dot(AB, AB)

    # clamp t to segment AB, t = 0 is A, t = 1 is B.
    t = max(0, min(1, t))
    
    # closest point on the segment
    closest = A + t * AB
    
    return np.linalg.norm(P - closest), closest

def nearest_node(proj_point, kdtree, node_ids):
    """
    Given: 
    proj_point: A candidates projected point, consisting of (x, y/z).
    kdtree:     The KD-tree containing the node coordinates.
    node_ids:   The list of node ids used to search the KD-tree
    
    Returns:
    1. The id of the node that is nearest to the projection.
    2. The distance between that node and the projection.
    """

    distance, index_nearest = kdtree.query((proj_point.x, proj_point.y))
    return node_ids[index_nearest], distance
