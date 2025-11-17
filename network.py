#TODO fix imports 

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import math
from trajectory import Trajectory
import pickle

from sklearn.discriminant_analysis import StandardScaler
from util import X_MAX, Y_MAX, Z_MAX, get_ap_locations_names, least_squares_trilaterate, load_files, filter_columns, evaluate_model, split_data, split_data_parts, trilaterate, unscale_xyz
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.pipeline import FunctionTransformer, Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV, ParameterGrid

from pipeline import SplitPipeline

def find_node_by_position(G, pos):
    for n, data in G.nodes(data=True):
        if data.get("pos") == pos:
            return n
    return None

def add_weighted_edge(G, node1, node2):
    pos = nx.get_node_attributes(G, "pos")
    node1_pos = pos[node1]
    node2_pos = pos[node2]

    node1_x = node1_pos[0]
    node1_z = node1_pos[1]

    node2_x = node2_pos[0]
    node2_z = node2_pos[1]

    distance = math.sqrt((node2_x - node1_x)**2 + (node2_z - node1_z)**2)
    
    G.add_edge(node1, node2, weight=distance)


def create_traj(G, df, requested_nodes):
    G_path = nx.Graph()
    traj = Trajectory(G, requested_nodes)

    #1. get random readings for requested nodes
    requested_node_positions = []

    for n in requested_nodes:
        requested_node_positions = requested_node_positions + [G.nodes[n].get("pos")]
        print(n, " = ", G.nodes[n].get("pos"))

    requested_node_random_picks = pd.DataFrame()

    for n in requested_node_positions:
        matched_rows = df[(df["x"] == n[0]) & (df["z"] == n[1])]
        random_row = matched_rows.sample(n=1, random_state=None)  # optional random_state for reproducibility
        requested_node_random_picks = pd.concat([requested_node_random_picks, random_row], ignore_index=True)

    print(requested_node_random_picks)

    #2. get predicted locations from model
    #TODO: model might not always have this name
    with open("sklearn_models/multilayer/Direct location prediction-11-12--11-03-11.pkl", "rb") as file:
        model = pickle.load(file)

    y_pred = model.predict(requested_node_random_picks)
    print("Prediction:", y_pred)

    #3. Build traj using Trajectory.update()
    for i, n in enumerate(y_pred):
        traj.update(y_pred[i][0], y_pred[i][1], y_pred[i][2], None)

    # Return the traj
    print(traj)
    return traj

def main():
    G = nx.Graph()

    df = load_files(["samples F5 everything.csv"])
    # df = load_files(["samples F6 everything.csv"])

    print(df)

    # Add nodes to network G
    for i, (x, y, z) in enumerate(df[["x", "y", "z"]].values):
        if find_node_by_position(G, (x, z)) is None:
            print(i, x, y, z)
            G.add_node(G.number_of_nodes(), pos=(x, z))

    # Add edges in between nodes (manual)
    add_weighted_edge(G, 0, 1)
    add_weighted_edge(G, 1, 2)
    add_weighted_edge(G, 2, 3)
    add_weighted_edge(G, 3, 4)
    add_weighted_edge(G, 4, 5)
    add_weighted_edge(G, 3, 5)
    add_weighted_edge(G, 4, 32)
    add_weighted_edge(G, 32, 33)
    add_weighted_edge(G, 33, 34)
    add_weighted_edge(G, 34, 35)
    add_weighted_edge(G, 35, 36)
    add_weighted_edge(G, 36, 37)
    add_weighted_edge(G, 37, 6)
    add_weighted_edge(G, 5, 6)
    add_weighted_edge(G, 6, 7)
    add_weighted_edge(G, 7, 8)
    add_weighted_edge(G, 8, 9)
    add_weighted_edge(G, 9, 10)
    add_weighted_edge(G, 10, 11)
    add_weighted_edge(G, 11, 12)
    add_weighted_edge(G, 12, 13)
    add_weighted_edge(G, 13, 14)
    add_weighted_edge(G, 14, 15)
    add_weighted_edge(G, 15, 16)
    add_weighted_edge(G, 16, 17)
    add_weighted_edge(G, 17, 18)
    add_weighted_edge(G, 18, 19)
    add_weighted_edge(G, 13, 20)
    add_weighted_edge(G, 20, 21)
    add_weighted_edge(G, 21, 22)
    add_weighted_edge(G, 22, 23)
    add_weighted_edge(G, 23, 24)
    add_weighted_edge(G, 24, 38)
    add_weighted_edge(G, 38, 39)
    add_weighted_edge(G, 39, 40)
    add_weighted_edge(G, 40, 41)
    add_weighted_edge(G, 24, 25)
    add_weighted_edge(G, 25, 26)
    add_weighted_edge(G, 26, 27)
    add_weighted_edge(G, 27, 28)
    add_weighted_edge(G, 28, 29)
    add_weighted_edge(G, 29, 30)
    add_weighted_edge(G, 30, 31)
    add_weighted_edge(G, 31, 0)

    # path = [0, 31, 30, 29, 28, 27, 26, 25, 24]
    path = [0, 1, 2, 3, 4, 32, 33, 34, 35, 36, 37, 6, 7, 8, 9, 10, 11, 12]
    traj = create_traj(G, df, path)

    # Prepare window with image, flip y-axis of figure and image.
    fig, ax = plt.subplots()
    img = mpimg.imread("floorplans/floor-06-cropped.png")
    img_flipped = np.flipud(img)
    ax.imshow(img_flipped, alpha=0.5)
    ax.invert_yaxis()

    # Draw the graph on top of the image
    pos = nx.get_node_attributes(G, "pos")
    scale = 5538.46153846 # TODO: this number is just an estimate obtained by dividing the desired x value in the image by the actual x in the dataframe
    pos = {node: (x * scale, z * scale) for node, (x, z) in pos.items()}
    nx.draw(G, pos, with_labels=True, node_color="cyan", edge_color="cyan", ax=ax)

    # Add weight labels to edges
    # edge_labels = nx.get_edge_attributes(G, "weight")
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    traj.draw(ax)

    # Show plot
    plt.show()

if __name__ == "__main__":
    main()
