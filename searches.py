import osmnx as ox
from IPython.display import IFrame
import networkx as nx
import folium
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
import heapq
import math
from collections import deque
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


# Get the road network data from OpenStreetMap
def get_road_network(location, distance):
    G = ox.graph_from_address(location, network_type="walk", dist=distance)
    return G

def save_map(map_obj, filename):
    map_obj.save(filename)


def visualize_road_network(G, location, distance, filename="road_network.html"):
    lat, lng = ox.geocode(location)
    map_center = [lat, lng]
    map_osm = folium.Map(location=map_center, zoom_start=12)
    ox.plot_graph_folium(
        G, graph_map=map_osm, popup_attribute="name", node_labels=True, edge_width=2
    )

    for node, data in G.nodes(data=True):
        folium.Marker(location=[data["y"], data["x"]], popup=f"Node: {node}").add_to(
            map_osm
        )

    save_map(map_osm, filename)


def visualize_path_folium(
    G,
    shortest_path,
    location,
    source_node,
    target_nodes,
    distance,
    filename="shortest_path.html",
):
    lat, lng = ox.geocode(location)
    map_center = [lat, lng]
    map_osm = folium.Map(location=map_center, zoom_start=12)
    ox.plot_graph_folium(G, graph_map=map_osm, node_labels=True, edge_width=2)

    folium.Marker(
        location=(G.nodes[source_node]["y"], G.nodes[source_node]["x"]),
        icon=folium.Icon(color="green"),
        popup=f"Source<br>Distance: {distance:.2f} meters",
    ).add_to(map_osm)

    for target_node in target_nodes:
        folium.Marker(
            location=(G.nodes[target_node]["y"], G.nodes[target_node]["x"]),
            icon=folium.Icon(color="red"),
            popup="Destination",
        ).add_to(map_osm)

    gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)
    shortest_path_coords = []
    for i in range(len(shortest_path) - 1):
        edge = (shortest_path[i], shortest_path[i + 1], 0)
        edge_coords = gdf_edges.loc[edge]["geometry"]
        shortest_path_coords.extend(
            [(point[1], point[0]) for point in edge_coords.coords]
        )

    folium.PolyLine(locations=shortest_path_coords, color="blue", weight=5).add_to(
        map_osm
    )
    save_map(map_osm, filename)


# ## Heuristic Functions
loc = "LUMS Lahore, Pakistan"
dist = 500

G = get_road_network(loc, dist)

# # Print nodes information
# for node, data in G.nodes(data=True):
#     print(f"Node {node}: Latitude - {data['y']}, Longitude - {data['x']}")

# # Print edges information
# for u, v, data in G.edges(data=True):
#     print(f"Edge ({u}, {v}): Length - {data['length']}")


def euclidean_heuristic(node1, node2):
    """
    node1: The ID of the first node
    node2: The ID of the second node
    Returns the Euclidean distance between the two nodes
    """
    x1, y1 = G.nodes[node1]["x"], G.nodes[node1]["y"]
    x2, y2 = G.nodes[node2]["x"], G.nodes[node2]["y"]
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def manhattan_heuristic(node1, node2):
    """
    node1: The ID of the first node
    node2: The ID of the second node
    Returns the Manhattan distance between the two nodes
    """
    x1, y1 = G.nodes[node1]["x"], G.nodes[node1]["y"]
    x2, y2 = G.nodes[node2]["x"], G.nodes[node2]["y"]
    return abs(x2 - x1) + abs(y2 - y1)


def haversine_heuristic(node1, node2):
    """
    node1: The ID of the first node
    node2: The ID of the second node
    Returns the Haversine distance between the two nodes
    """
    lat1, lon1 = math.radians(G.nodes[node1]["y"]), math.radians(G.nodes[node1]["x"])
    lat2, lon2 = math.radians(G.nodes[node2]["y"]), math.radians(G.nodes[node2]["x"])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    R = 6371.0  # Radius of the Earth in km
    return R * c


# def test_heuristics():
#     node1 = list(G.nodes())[0]
#     node2 = list(G.nodes())[1]

#     print(
#         f"Euclidean distance between node {node1} and node {node2}: {euclidean_heuristic(node1, node2)}"
#     )

#     print(
#         f"Manhattan distance between node {node1} and node {node2}: {manhattan_heuristic(node1, node2)}"
#     )

#     print(
#         f"Haversine distance between node {node1} and node {node2}: {haversine_heuristic(node1, node2)}"
#     )


# test_heuristics()


def a_star(graph, start, goal, heuristic_func):
    """
    graph: The graph object representing the road network.
    start: The starting node ID.
    goal: The destination node ID.
    heuristic_func: The heuristic function to be used to calculate the cost.
    """
    queue = []
    heapq.heappush(queue, (0, start))

    path = {start: None}
    cost = {start: 0}

    while queue:
        _, current_node = heapq.heappop(queue)

        if current_node == goal:
            break

        for neighbor in graph.neighbors(current_node):
            new_cost = (
                cost[current_node] + graph.edges[current_node, neighbor, 0]["length"]
            )

            if neighbor not in cost or new_cost < cost[neighbor]:
                cost[neighbor] = new_cost
                priority = new_cost + heuristic_func(current_node, neighbor)
                # print(new_cost)
                heapq.heappush(queue, (priority, neighbor))

                path[neighbor] = current_node

    shortest_path = []
    while current_node is not None:
        shortest_path.append(current_node)
        current_node = path[current_node]
    shortest_path.reverse()

    return shortest_path, cost[goal]


# def test_a_star():
#     source_node = list(G.nodes())[0]
#     target_node = list(G.nodes())[1]

#     shortest_path, total_cost = a_star(G, source_node, target_node, euclidean_heuristic)

#     print(
#         f"Shortest path from node {source_node} to node {target_node} using A* algorithm: {shortest_path}"
#     )
#     print(f"Total cost: {total_cost}")


# test_a_star()


def best_first_search(graph, start, goal, heuristic_func):
    """
    graph: The graph object representing the road network.
    start: The starting node ID.
    goal: The destination node ID.
    heuristic_func: Euclidean heuristic ONLY
    """
    queue = []
    heapq.heappush(queue, (0, start))

    path = {start: None}
    cost = {start: 0}

    while queue:
        _, current_node = heapq.heappop(queue)

        if current_node == goal:
            break

        for neighbor in graph.neighbors(current_node):
            new_cost = (
                cost[current_node] + graph.edges[current_node, neighbor, 0]["length"]
            )

            if neighbor not in cost or new_cost < cost[neighbor]:

                cost[neighbor] = new_cost
                priority = heuristic_func(current_node, neighbor)

                heapq.heappush(queue, (priority, neighbor))

                path[neighbor] = current_node

    shortest_path = []
    while current_node is not None:
        shortest_path.append(current_node)
        current_node = path[current_node]
    shortest_path.reverse()

    return shortest_path, cost[goal]


# def test_best_first_search():
#     source_node = list(G.nodes())[0]
#     target_node = list(G.nodes())[1]

#     shortest_path, total_cost = best_first_search(
#         G, source_node, target_node, euclidean_heuristic
#     )

#     print(
#         f"Shortest path from node {source_node} to node {target_node} using Best-First Search algorithm: {shortest_path}"
#     )
#     print(f"Total cost: {total_cost}")


# test_best_first_search()


def bfs(graph, start, goal, heuristic_func):
    """
    graph: The graph object representing the road network.
    start: The starting node ID.
    goal: The destination node ID.
    heuristic_func: Euclidean heuristic ONLY
    """
    queue = [(0, start)]

    path = {start: None}
    cost = {start: 0}

    while queue:
        _, current_node = heapq.heappop(queue)

        if current_node == goal:
            break

        for neighbor in graph.neighbors(current_node):
            new_cost = (
                cost[current_node] + graph.edges[current_node, neighbor, 0]["length"]
            )

            if neighbor not in cost or new_cost < cost[neighbor]:

                cost[neighbor] = new_cost

                priority = heuristic_func(current_node, neighbor)

                heapq.heappush(queue, (priority, neighbor))

                path[neighbor] = current_node

    shortest_path = []
    while current_node is not None:
        shortest_path.append(current_node)
        current_node = path[current_node]
    shortest_path.reverse()

    return shortest_path, cost[goal]


# def test_bfs():
#     source_node = list(G.nodes())[0]
#     target_node = list(G.nodes())[1]

#     shortest_path, total_cost = bfs(G, source_node, target_node, euclidean_heuristic)

#     print(f"Shortest path from node {source_node} to node {target_node} using informed Breadth-First Search algorithm: {shortest_path}")
#     print(f"Total cost: {total_cost}")

# test_bfs()


def astar_networkx_path(G, source, target):
    """
    G: Graph object representing the road network
    source:  ID of the source node
    target: ID of the target node
    """
    return nx.astar_path(G, source=source, target=target, weight="length")


def a_star_multiple(graph, start, goals, heuristic_func):
    """
    graph: The graph object representing the road network.
    start: The starting node ID.
    goals: The list of destination node IDs.
    heuristic_func: The heuristic function to be used to calculate the cost.
    """
    complete_path = [start]

    total_cost = 0

    for i in range(len(goals)):
        current_start = complete_path[-1]

        current_goal = goals[i]

        shortest_path, cost = a_star(graph, current_start, current_goal, heuristic_func)

        complete_path += shortest_path[1:]

        total_cost += cost

    return complete_path, total_cost


def main():
    print("AI Search Algorithms")
    location = "LUMS Lahore, Pakistan"
    distance = 500
    source = 810005319  # SSE
    destination = 11337034500  # SDSB

    G = get_road_network(location, distance)
    visualize_road_network(G, location, distance, "road_network.html")
    print("Road network visualized in 'road_network.html'")

    shortest_path, total_cost = a_star(G, source, destination, euclidean_heuristic)
    visualize_path_folium(
        G,
        shortest_path,
        location,
        source,
        [destination],
        total_cost,
        "a_star_euclidean.html",
    )
    print("A* with Euclidean heuristic visualized in 'a_star_euclidean.html'")

    shortest_path, total_cost = a_star(G, source, destination, manhattan_heuristic)
    visualize_path_folium(
        G,
        shortest_path,
        location,
        source,
        [destination],
        total_cost,
        "a_star_manhattan.html",
    )
    print("A* with Manhattan heuristic visualized in 'a_star_manhattan.html'")

    shortest_path, total_cost = a_star(G, source, destination, haversine_heuristic)
    visualize_path_folium(
        G,
        shortest_path,
        location,
        source,
        [destination],
        total_cost,
        "a_star_haversine.html",
    )
    print("A* with Haversine heuristic visualized in 'a_star_haversine.html'")

    shortest_path, total_cost = best_first_search(
        G, source, destination, euclidean_heuristic
    )
    visualize_path_folium(
        G,
        shortest_path,
        location,
        source,
        [destination],
        total_cost,
        "best_first_search.html",
    )
    print("Best First Search visualized in 'best_first_search.html'")

    shortest_path, total_cost = bfs(G, source, destination, euclidean_heuristic)
    visualize_path_folium(
        G, shortest_path, location, source, [destination], total_cost, "bfs.html"
    )
    print("Informed BFS visualized in 'bfs.html'")

    dest1 = 11336997534  # Cricket Ground node ID
    dest2 = 809970907  # Mosque
    optimal_route, total_cost = a_star_multiple(
        G, source, [dest1, dest2], euclidean_heuristic
    )
    visualize_path_folium(
        G,
        optimal_route,
        location,
        source,
        [dest1, dest2],
        total_cost,
        "a_star_multiple.html",
    )
    print("A* multiple goals visualized in 'a_star_multiple.html'")

    print("Code Terminated")


if __name__ == "__main__":
    main()


# # Analysis
#Best Algo?-> A* Search. This is because in our case we had access to many heuritic functions which could be used to implement the A* search algorithm and given that one is able to find a suitable heuritic function, A* algorithm in this case becomes the most optimal form of search. Best first only takes into account the heuristic which is not enough in the case of mutiple nodes/destinations, as shown in task 4 where we had to have multiple destinations, previous costs were considered as well hence A* was the most optimal choice. Similarly, A* was also better than the informed BFS model since informed BFS does not guarantee to find the shortest path whereas A* is optimized to do so.
# Note: It should be noted that for some of the routes, all the algorithms ended up giving the same path. This might be becasue that was the only optimal path according to the model's prediction.
