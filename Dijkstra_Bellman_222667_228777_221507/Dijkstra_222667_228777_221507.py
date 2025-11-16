import heapq
import time
import random
import os
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

class Graph:
    def __init__(self):
        self.nodes = set()
        self.edges = defaultdict(list)
        self.weights = {}
        
    def add_edge(self, from_node, to_node, weight):
        self.nodes.add(from_node)
        self.nodes.add(to_node)
        self.edges[from_node].append(to_node)
        self.edges[to_node].append(from_node)  # Since the graph is undirected
        self.weights[(from_node, to_node)] = weight
        self.weights[(to_node, from_node)] = weight  # Since the graph is undirected
        
    def get_nodes(self):
        return sorted(list(self.nodes))
    
    def get_neighbors(self, node):
        return self.edges[node]
    
    def get_weight(self, from_node, to_node):
        return self.weights.get((from_node, to_node), float('inf'))

def load_graph(filename, use_nx=False):
    """
    Load the graph from the provided file and assign random weights
    """
    if use_nx:
        G = nx.Graph()
        with open(filename, 'r') as file:
            for line in file:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) == 2:
                    source = int(parts[0])
                    target = int(parts[1])
                    weight = random.randint(1, 9)
                    G.add_edge(source, target, weight=weight)
        return G
    else:
        graph = Graph()
        with open(filename, 'r') as file:
            for line in file:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) == 2:
                    from_node = int(parts[0])
                    to_node = int(parts[1])
                    weight = random.randint(1, 9)
                    graph.add_edge(from_node, to_node, weight)
        return graph

def dijkstra(graph, start, trace_file=None):
    """
    Implements Dijkstra's algorithm for shortest path
    with detailed tracing of queue operations
    """
    trace_output = []
    
    # Initialize distances with infinity for all nodes except start
    distances = {node: float('inf') for node in graph.nodes}
    distances[start] = 0
    
    # Initialize predecessors for path reconstruction
    predecessors = {node: None for node in graph.nodes}
    
    # Priority queue (min-heap) for Dijkstra
    priority_queue = [(0, start)]
    heapq.heapify(priority_queue)
    
    trace_output.append(f"Initial queue: {priority_queue}")
    
    # Set to keep track of processed nodes
    processed = set()
    
    while priority_queue:
        # Extract node with minimum distance
        current_distance, current_node = heapq.heappop(priority_queue)
        trace_output.append(f"Extracted: ({current_distance}, {current_node})")
        
        # Skip if we've already processed this node
        if current_node in processed:
            trace_output.append(f"  Node {current_node} already processed, skipping")
            continue
        
        # Mark node as processed
        processed.add(current_node)
        trace_output.append(f"  Processed nodes: {sorted(list(processed))}")
        
        # If current distance is worse than what we already know, skip
        if current_distance > distances[current_node]:
            trace_output.append(f"  Current distance {current_distance} > known best {distances[current_node]}, skipping")
            continue
        
        # Check all neighbors
        neighbors = graph.get_neighbors(current_node)
        trace_output.append(f"  Checking neighbors of {current_node}: {neighbors}")
        
        for neighbor in neighbors:
            # Skip if already processed
            if neighbor in processed:
                trace_output.append(f"    Neighbor {neighbor} already processed, skipping")
                continue
            
            # Calculate new distance
            weight = graph.get_weight(current_node, neighbor)
            new_distance = distances[current_node] + weight
            trace_output.append(f"    Edge ({current_node}, {neighbor}) weight: {weight}")
            trace_output.append(f"    New distance to {neighbor}: {new_distance}, old: {distances[neighbor]}")
            
            # Update distance if better path found
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                predecessors[neighbor] = current_node
                trace_output.append(f"    Updated distance to node {neighbor}: {new_distance}")
                
                # Add to priority queue
                heapq.heappush(priority_queue, (new_distance, neighbor))
                trace_output.append(f"    Added to queue: ({new_distance}, {neighbor})")
        
        trace_output.append(f"  Current queue: {priority_queue}")
    
    # Write trace to file if specified
    if trace_file:
        with open(trace_file, 'w') as f:
            for line in trace_output:
                f.write(line + '\n')
    
    return distances, predecessors

def bellman_ford(graph, start, trace_file=None):
    """
    Implements Bellman-Ford algorithm for shortest path
    with detailed tracing
    """
    trace_output = []
    
    # Initialize distances with infinity for all nodes except start
    distances = {node: float('inf') for node in graph.nodes}
    distances[start] = 0
    
    # Initialize predecessors for path reconstruction
    predecessors = {node: None for node in graph.nodes}
    
    trace_output.append(f"Initial distances: {distances}")
    
    # Process |V| - 1 times (where |V| is number of nodes)
    num_nodes = len(graph.nodes)
    
    for i in range(num_nodes - 1):
        trace_output.append(f"\nIteration {i+1}:")
        changes_made = False
        
        # Process all edges
        for u in graph.nodes:
            for v in graph.get_neighbors(u):
                weight = graph.get_weight(u, v)
                
                if distances[u] != float('inf') and distances[u] + weight < distances[v]:
                    trace_output.append(f"  Relaxing edge ({u}, {v}): {distances[v]} -> {distances[u] + weight}")
                    distances[v] = distances[u] + weight
                    predecessors[v] = u
                    changes_made = True
                    
        if not changes_made:
            trace_output.append("  No changes made in this iteration, algorithm can terminate early.")
            break
    
    # Check for negative cycles
    for u in graph.nodes:
        for v in graph.get_neighbors(u):
            weight = graph.get_weight(u, v)
            if distances[u] != float('inf') and distances[u] + weight < distances[v]:
                trace_output.append(f"Negative cycle detected! Edge ({u}, {v})")
                return None, None  # Negative cycle
    
    # Write trace to file if specified
    if trace_file:
        with open(trace_file, 'w') as f:
            for line in trace_output:
                f.write(line + '\n')
    
    return distances, predecessors

def find_graph_diameter(graph, graph_nx):
    """
    Find the diameter of the graph (the longest shortest path between any two nodes)
    with tracing, performance measurement, and visualization
    """
    # Create output directory if it doesn't exist
    os.makedirs("outputs/diameter", exist_ok=True)
    
    trace_file = "outputs/diameter/diameter_trace.txt"
    result_file = "outputs/diameter/diameter_result.txt"
    time_file = "outputs/diameter/diameter_time.txt"
    viz_file = "outputs/diameter/diameter_visualization.png"
    
    trace_output = []
    diameter = 0
    diameter_pair = (None, None)
    diameter_path = []
    execution_times = []
    
    # Get all nodes in the largest connected component
    largest_cc = max(nx.connected_components(graph_nx), key=len)
    nodes_to_process = list(largest_cc)
    total_nodes = len(nodes_to_process)
    
    trace_output.append(f"Calculating diameter for graph with {total_nodes} nodes in largest connected component")
    trace_output.append(f"Processing {total_nodes} nodes...")
    
    start_time = time.time()
    
    for i, node in enumerate(nodes_to_process):
        node_start_time = time.time()
        
        # Use NetworkX's optimized Dijkstra implementation for performance
        paths = nx.single_source_dijkstra_path(graph_nx, node, weight='weight')
        lengths = nx.single_source_dijkstra_path_length(graph_nx, node, weight='weight')
        
        # Find the maximum distance from this node
        max_dist_node = max(lengths.items(), key=lambda x: x[1] if x[1] != float('inf') else -1)[0]
        max_dist = lengths[max_dist_node]
        
        if max_dist > diameter:
            diameter = max_dist
            diameter_pair = (node, max_dist_node)
            diameter_path = paths[max_dist_node]
            trace_output.append(f"New diameter found: {diameter} between {node} and {max_dist_node}")
        
        node_time = time.time() - node_start_time
        execution_times.append(node_time)
        
        # Progress update every 10 nodes
        if (i + 1) % 10 == 0 or i == total_nodes - 1:
            trace_output.append(f"Processed {i + 1}/{total_nodes} nodes")
            trace_output.append(f"Current diameter: {diameter}")
            trace_output.append(f"Average time per node: {np.mean(execution_times):.4f}s")
    
    total_time = time.time() - start_time
    
    # Visualize the diameter path
    if diameter_path:
        visualize_diameter_path(graph_nx, diameter_path, diameter_pair, diameter, viz_file)
    
    # Write trace to file
    with open(trace_file, 'w') as f:
        f.write("Diameter Calculation Trace\n")
        f.write("="*50 + "\n")
        for line in trace_output:
            f.write(line + "\n")
    
    # Write result to file
    with open(result_file, 'w') as f:
        f.write("Graph Diameter Result\n")
        f.write("="*50 + "\n")
        f.write(f"Diameter: {diameter}\n")
        f.write(f"Node pair with this distance: {diameter_pair[0]} and {diameter_pair[1]}\n")
        f.write(f"Path length: {len(diameter_path)} nodes\n")
        f.write(f"Total nodes processed: {total_nodes}\n")
    
    # Write timing information to file
    with open(time_file, 'w') as f:
        f.write("Diameter Calculation Timing\n")
        f.write("="*50 + "\n")
        f.write(f"Total execution time: {total_time:.4f} seconds\n")
        f.write(f"Average time per node: {np.mean(execution_times):.4f} seconds\n")
        f.write(f"Minimum node time: {min(execution_times):.4f} seconds\n")
        f.write(f"Maximum node time: {max(execution_times):.4f} seconds\n")
    
    return diameter, diameter_pair, diameter_path, total_time

def visualize_diameter_path(G, path, endpoints, diameter, output_file):
    """
    Visualize the diameter path in the graph
    """
    # Extract the subgraph containing the path plus some neighbors
    path_set = set(path)
    extended_nodes = set(path)
    
    # Add some neighbors of the path nodes
    for node in path:
        neighbors = list(G.neighbors(node))
        sample_size = min(2, len(neighbors))  # Add up to 2 neighbors for each node
        extended_nodes.update(random.sample(neighbors, sample_size))
    
    subgraph = G.subgraph(extended_nodes)
    
    # Get positions for nodes (fixed layout for path nodes)
    pos = nx.spring_layout(subgraph, seed=42)
    
    plt.figure(figsize=(12, 8))
    
    # Draw all nodes and edges
    nx.draw_networkx_nodes(subgraph, pos, node_color='lightblue', node_size=300)
    nx.draw_networkx_edges(subgraph, pos, width=1.0, alpha=0.4)
    
    # Draw the diameter path edges
    path_edges = list(zip(path[:-1], path[1:]))
    nx.draw_networkx_edges(subgraph, pos, edgelist=path_edges, width=3.0, edge_color='red')
    
    # Draw the path nodes
    nx.draw_networkx_nodes(subgraph, pos, nodelist=path, node_color='yellow', node_size=400)
    
    # Highlight endpoint nodes
    nx.draw_networkx_nodes(subgraph, pos, nodelist=[endpoints[0]], node_color='green', node_size=500)
    nx.draw_networkx_nodes(subgraph, pos, nodelist=[endpoints[1]], node_color='blue', node_size=500)
    
    # Draw edge weights on the path
    edge_labels = {(u, v): G[u][v]['weight'] for u, v in path_edges}
    nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, font_size=10)
    
    # Draw node labels
    nx.draw_networkx_labels(subgraph, pos)
    
    plt.title(f"Graph Diameter Path (Distance: {diameter}) between {endpoints[0]} and {endpoints[1]}")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, dpi=500)
    plt.show()
    plt.close()

def reconstruct_paths(start, distances, predecessors, output_file=None):
    """
    Reconstructs and formats all shortest paths from the start node
    """
    output = []
    output.append(f"Shortest paths from node {start}:")
    
    for node in sorted(distances.keys()):
        if node == start:
            continue
            
        if distances[node] == float('inf'):
            output.append(f"  No path to node {node}")
            continue
            
        # Reconstruct path
        path = []
        current = node
        
        while current is not None:
            path.append(current)
            current = predecessors[current]
            
        path.reverse()
        
        # Format output
        path_str = " -> ".join(map(str, path))
        output.append(f"  To node {node}: Distance = {distances[node]}, Path = {path_str}")
    
    # Write to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            for line in output:
                f.write(line + '\n')
    
    return output

def visualize_subgraph(G, source_node, max_nodes=100):
    """
    Visualize a smaller portion of the graph centered around the source node
    """
    # Create a subgraph with BFS from source node
    nodes = list(nx.bfs_tree(G, source=source_node, depth_limit=3))[:max_nodes]
    subgraph = G.subgraph(nodes)
    
    # Get positions for nodes
    pos = nx.spring_layout(subgraph, seed=42)
    
    plt.figure(figsize=(12, 10))
    
    # Draw nodes
    nx.draw_networkx_nodes(subgraph, pos, node_size=300, node_color='lightblue')
    
    # Draw edges with weights as labels
    nx.draw_networkx_edges(subgraph, pos, width=1.0, alpha=0.7)
    edge_labels = {(u, v): d['weight'] for u, v, d in subgraph.edges(data=True)}
    nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, font_size=8)
    
    # Draw node labels
    nx.draw_networkx_labels(subgraph, pos, font_size=10)
    
    # Highlight source node
    nx.draw_networkx_nodes(subgraph, pos, nodelist=[source_node], 
                          node_color='red', node_size=500)
    
    plt.title(f"Subgraph visualization around node {source_node}")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("outputs/graph_visualization.png", dpi=500)
    plt.show()
    plt.close()
    
    return subgraph

def analyze_graph(G):
    """
    Analyze the graph and save basic metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics["Number of nodes"] = G.number_of_nodes()
    metrics["Number of edges"] = G.number_of_edges()
    metrics["Graph density"] = nx.density(G)
    
    # Get largest connected component
    largest_cc = max(nx.connected_components(G), key=len)
    largest_cc_graph = G.subgraph(largest_cc)
    metrics["Size of largest connected component"] = len(largest_cc)
    metrics["Percentage of nodes in largest component"] = len(largest_cc) / G.number_of_nodes() * 100
    
    # Calculate diameter of largest component (approximation if needed)
    if len(largest_cc) < 1000:  # Full calculation for smaller components
        metrics["Diameter of largest component"] = nx.diameter(largest_cc_graph)
    else:  # Approximation for larger components
        sample_nodes = random.sample(list(largest_cc), min(100, len(largest_cc)))
        max_path = 0
        for node in sample_nodes:
            path_lengths = nx.single_source_shortest_path_length(largest_cc_graph, node)
            if path_lengths:
                current_max = max(path_lengths.values())
                max_path = max(max_path, current_max)
        metrics["Approximate diameter of largest component"] = max_path
    
    # Write metrics to file
    with open("outputs/graph_metrics.txt", "w") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    
    return metrics

def visualize_shortest_path(G, source_node, target_node, path, distances):
    """
    Visualize the shortest path between source and target nodes
    """
    # Extract the subgraph containing the path plus some neighbors
    path_set = set(path)
    extended_nodes = set(path)
    
    # Add some neighbors of the path nodes
    for node in path:
        neighbors = list(G.neighbors(node))
        sample_size = min(2, len(neighbors))  # Add up to 2 neighbors for each node
        extended_nodes.update(random.sample(neighbors, sample_size))
    
    subgraph = G.subgraph(extended_nodes)
    
    # Get positions for nodes (fixed layout for path nodes)
    pos = nx.spring_layout(subgraph, seed=42)
    
    plt.figure(figsize=(12, 8))
    
    # Draw all nodes and edges
    nx.draw_networkx_nodes(subgraph, pos, node_color='lightblue', node_size=300)
    nx.draw_networkx_edges(subgraph, pos, width=1.0, alpha=0.4)
    
    # Draw the path edges
    path_edges = list(zip(path[:-1], path[1:]))
    nx.draw_networkx_edges(subgraph, pos, edgelist=path_edges, width=3.0, edge_color='blue')
    
    # Draw the path nodes
    nx.draw_networkx_nodes(subgraph, pos, nodelist=path, node_color='green', node_size=400)
    
    # Highlight source and target nodes
    nx.draw_networkx_nodes(subgraph, pos, nodelist=[source_node], node_color='red', node_size=500)
    nx.draw_networkx_nodes(subgraph, pos, nodelist=[target_node], node_color='purple', node_size=500)
    
    # Draw edge weights on the path
    edge_labels = {(u, v): G[u][v]['weight'] for u, v in path_edges}
    nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, font_size=10)
    
    # Draw node labels
    nx.draw_networkx_labels(subgraph, pos)
    
    plt.title(f"Shortest Path from {source_node} to {target_node} (Distance: {distances[target_node]})")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"outputs/shortest_path_{source_node}_to_{target_node}.png", dpi=500)
    plt.show()
    plt.close()

def create_algorithm_comparison(dijkstra_time, bellman_ford_time):
    """Create comparison chart of algorithm execution times"""
    plt.figure(figsize=(8, 6))
    algorithms = ['Dijkstra', 'Bellman-Ford']
    times = [dijkstra_time, bellman_ford_time]
    
    bars = plt.bar(algorithms, times, color=['blue', 'orange'])
    plt.ylabel('Execution Time (seconds)')
    plt.title('Algorithm Performance Comparison')
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig("outputs/algorithm_comparison.png", dpi=300)
    plt.show()
    plt.close()

def create_distance_distribution(distances):
    """Create histogram of shortest path distances"""
    # Filter out infinite distances
    finite_distances = [d for d in distances.values() if d != float('inf')]
    
    plt.figure(figsize=(10, 6))
    plt.hist(finite_distances, bins=20, color='green', edgecolor='black')
    plt.xlabel('Shortest Path Distance')
    plt.ylabel('Number of Nodes')
    plt.title('Distribution of Shortest Path Distances from Source Node')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("outputs/distance_distribution.png", dpi=300)
    plt.show()
    plt.close()

def create_reachability_pie(distances):
    """Create pie chart showing reachable vs unreachable nodes"""
    reachable = sum(1 for d in distances.values() if d != float('inf')) - 1  # exclude source node
    unreachable = sum(1 for d in distances.values() if d == float('inf'))
    
    plt.figure(figsize=(8, 8))
    labels = ['Reachable', 'Unreachable']
    sizes = [reachable, unreachable]
    colors = ['lightgreen', 'lightcoral']
    explode = (0.1, 0)  # explode the 1st slice
    
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
           autopct='%1.1f%%', shadow=True, startangle=140)
    plt.title('Node Reachability from Source Node')
    
    plt.tight_layout()
    plt.savefig("outputs/reachability_pie.png", dpi=300)
    plt.show()
    plt.close()

def display_menu():
    print("\n" + "="*50)
    print("GRAPH ANALYSIS AND VISUALIZATION TOOL")
    print("="*50)
    print("1. Run Dijkstra's Algorithm")
    print("2. Run Bellman-Ford Algorithm")
    print("3. Visualize Subgraph")
    print("4. Analyze Graph Metrics")
    print("5. Visualize Shortest Path")
    print("6. Find Graph Diameter")
    print("7. Run All Algorithms and Generate Visualizations")
    print("8. Exit")
    print("="*50)

def main():
    # Create output directory if it doesn't exist
    os.makedirs("outputs", exist_ok=True)
    
    # Load graph from file
    print("Loading graph from file...")
    graph_nx = load_graph("oregon1_010331.txt", use_nx=True)
    graph = load_graph("oregon1_010331.txt", use_nx=False)
    print(f"Graph loaded with {len(graph.nodes)} nodes")
    
    # Get source node from user
    while True:
        try:
            source_node = int(input("Enter source node (or press Enter for default node 4725): ") or "4725")
            if source_node in graph.nodes:
                break
            else:
                print(f"Node {source_node} not found in graph. Please enter a valid node.")
        except ValueError:
            print("Please enter a valid integer.")
    
    dijkstra_time = None
    bellman_ford_time = None
    distances = None
    
    while True:
        display_menu()
        choice = input("Enter your choice (1-8): ")
        
        if choice == '1':
            # Run Dijkstra's algorithm
            print("\nRunning Dijkstra's algorithm...")
            start_time = time.time()
            distances, predecessors = dijkstra(graph, source_node, trace_file="outputs/dijkstra_trace.txt")
            dijkstra_time = time.time() - start_time
            
            # Output results
            output = reconstruct_paths(source_node, distances, predecessors, 
                                    output_file="outputs/dijkstra_paths.txt")
            print("\nDijkstra's algorithm results:")
            for line in output[:10]:  # Display first 10 paths
                print(line)
            print(f"... (see full results in outputs/dijkstra_paths.txt)")
            print(f"Execution time: {dijkstra_time:.6f} seconds")
            
            # Generate distance distribution and reachability charts
            if distances:
                create_distance_distribution(distances)
                create_reachability_pie(distances)
            
        elif choice == '2':
            # Run Bellman-Ford algorithm
            print("\nRunning Bellman-Ford algorithm...")
            start_time = time.time()
            distances, predecessors = bellman_ford(graph, source_node, trace_file="outputs/bellman_ford_trace.txt")
            bellman_ford_time = time.time() - start_time
            
            if distances is not None:
                output = reconstruct_paths(source_node, distances, predecessors, 
                                        output_file="outputs/bellman_ford_paths.txt")
                print("\nBellman-Ford algorithm results:")
                for line in output[:10]:  # Display first 10 paths
                    print(line)
                print(f"... (see full results in outputs/bellman_ford_paths.txt)")
            else:
                print("\nBellman-Ford detected a negative cycle in the graph.")
            print(f"Execution time: {bellman_ford_time:.6f} seconds")
            
            # Generate distance distribution and reachability charts
            if distances:
                create_distance_distribution(distances)
                create_reachability_pie(distances)
            
        elif choice == '3':
            # Visualize subgraph
            print("\nVisualizing subgraph around source node...")
            visualize_subgraph(graph_nx, source_node)
            print("Subgraph visualization saved to outputs/graph_visualization.png")
            
        elif choice == '4':
            # Analyze graph metrics
            print("\nAnalyzing graph metrics...")
            metrics = analyze_graph(graph_nx)
            print("\nGraph Metrics:")
            for key, value in metrics.items():
                print(f"{key}: {value}")
            print("Full metrics saved to outputs/graph_metrics.txt")
            
        elif choice == '5':
            # Visualize shortest path
            if distances is None:
                print("\nPlease run Dijkstra or Bellman-Ford algorithm first to compute distances.")
                continue
                
            print("\nFinding shortest paths from source node...")
            paths = nx.single_source_dijkstra_path(graph_nx, source_node, weight='weight')
            
            # Select target nodes at different distances
            sorted_distances = sorted([(node, dist) for node, dist in distances.items()], key=lambda x: x[1])
            targets = []
            if len(sorted_distances) > 3:
                targets.append(sorted_distances[min(5, len(sorted_distances)-1)][0])
                targets.append(sorted_distances[len(sorted_distances)//2][0])
                targets.append(sorted_distances[-5][0])
            else:
                targets = [node for node, _ in sorted_distances if node != source_node][:3]
            
            for target in targets:
                if target != source_node:
                    print(f"Visualizing path to node {target}...")
                    visualize_shortest_path(graph_nx, source_node, target, paths[target], distances)
            
        elif choice == '6':
            # Find graph diameter
            print("\nFinding graph diameter (this may take a while for large graphs)...")
            diameter, diameter_pair, diameter_path, diameter_time = find_graph_diameter(graph, graph_nx)
            print(f"\nGraph diameter: {diameter}")
            print(f"Node pair with diameter distance: {diameter_pair[0]} and {diameter_pair[1]}")
            print(f"Path length: {len(diameter_path)} nodes")
            print(f"Execution time: {diameter_time:.2f} seconds")
            print("Visualization saved to outputs/diameter/diameter_visualization.png")
            print("Detailed results saved in outputs/diameter/ directory")
            
        elif choice == '7':
            # Run all algorithms and generate visualizations
            print("\nRunning all algorithms and generating visualizations...")
            
            # Dijkstra
            print("\n1. Running Dijkstra's algorithm...")
            start_time = time.time()
            distances, predecessors = dijkstra(graph, source_node, trace_file="outputs/dijkstra_trace.txt")
            dijkstra_time = time.time() - start_time
            print(f"Dijkstra completed in {dijkstra_time:.6f} seconds")
            
            # Bellman-Ford
            print("\n2. Running Bellman-Ford algorithm...")
            start_time = time.time()
            bf_distances, bf_predecessors = bellman_ford(graph, source_node, trace_file="outputs/bellman_ford_trace.txt")
            bellman_ford_time = time.time() - start_time
            if bf_distances is not None:
                print(f"Bellman-Ford completed in {bellman_ford_time:.6f} seconds")
            else:
                print("Bellman-Ford detected a negative cycle")
            
            # Create algorithm comparison chart if both ran successfully
            if dijkstra_time is not None and bellman_ford_time is not None:
                create_algorithm_comparison(dijkstra_time, bellman_ford_time)
            
            # Visualize subgraph
            print("\n3. Visualizing subgraph...")
            visualize_subgraph(graph_nx, source_node)
            
            # Analyze graph
            print("\n4. Analyzing graph metrics...")
            analyze_graph(graph_nx)
            
            # Find diameter
            print("\n5. Finding graph diameter...")
            diameter, diameter_pair, diameter_path, diameter_time = find_graph_diameter(graph, graph_nx)
            print(f"Diameter found: {diameter} between nodes {diameter_pair}")
            
            # Visualize shortest paths and generate charts
            if distances:
                print("\n6. Visualizing shortest paths and generating charts...")
                paths = nx.single_source_dijkstra_path(graph_nx, source_node, weight='weight')
                
                # Select target nodes at different distances
                sorted_distances = sorted([(node, dist) for node, dist in distances.items()], key=lambda x: x[1])
                targets = []
                if len(sorted_distances) > 3:
                    targets.append(sorted_distances[min(5, len(sorted_distances)-1)][0])
                    targets.append(sorted_distances[len(sorted_distances)//2][0])
                    targets.append(sorted_distances[-5][0])
                else:
                    targets = [node for node, _ in sorted_distances if node != source_node][:3]
                
                for target in targets:
                    if target != source_node:
                        visualize_shortest_path(graph_nx, source_node, target, paths[target], distances)
                
                # Generate distance distribution and reachability charts
                create_distance_distribution(distances)
                create_reachability_pie(distances)
            
            print("\nAll operations completed. Results saved in 'outputs' directory.")
            
        elif choice == '8':
            print("Exiting program...")
            break
            
        else:
            print("Invalid choice. Please enter a number between 1 and 8.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()