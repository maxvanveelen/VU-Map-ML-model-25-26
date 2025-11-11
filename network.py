import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from util import X_MAX, Y_MAX, Z_MAX, get_ap_locations_names, least_squares_trilaterate, load_files, filter_columns, evaluate_model, split_data, split_data_parts, trilaterate, unscale_xyz
import math

def find_node_by_position(G, pos):
    for n, data in G.nodes(data=True):
        if data.get("pos") == pos:
            return n
    return None

def add_weighted_edge(G, node1, node2):
    pos = nx.get_node_attributes(G, 'pos')
    node1_pos = pos[node1]
    node2_pos = pos[node2]

    node1_x = node1_pos[0]
    node1_z = node1_pos[1]

    node2_x = node2_pos[0]
    node2_z = node2_pos[1]

    distance = math.sqrt((node2_x - node1_x)**2 + (node2_z - node1_z)**2)
    
    G.add_edge(node1, node2, weight=distance)

def main():
    G = nx.Graph()

    df = load_files(["samples F5 everything.csv"])
    # df = load_files(["samples F6 everything.csv"])

    print(df)

    # Add nodes to network G
    for i, (x, y, z) in enumerate(df[['x', 'y', 'z']].values):
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
    add_weighted_edge(G, 30, 31)
    add_weighted_edge(G, 31, 0)

    # Prepare window with image, flip y-axis of figure and image.
    fig, ax = plt.subplots()
    img = mpimg.imread("floorplans/floor-06-cropped.png")
    img_flipped = np.flipud(img)
    ax.imshow(img_flipped, alpha=0.5)
    ax.invert_yaxis()

    # Draw the graph on top of the image
    pos = nx.get_node_attributes(G, 'pos')
    scale = 5538.46153846 # TODO: this number is just an estimate obtained by dividing the desired x value in the image by the actual x in the dataframe
    pos = {node: (x * scale, y * scale) for node, (x, y) in pos.items()}
    nx.draw(G, pos, with_labels=True, node_color="cyan", edge_color="black", ax=ax)

    # Add weight labels to edges
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # Show plot
    plt.show()

if __name__ == '__main__':
    main()
