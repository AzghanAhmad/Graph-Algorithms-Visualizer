import time
import random
import heapq
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
import os

class Graph:
    def __init__(self):
        self.graph = defaultdict(list)  # Adjacency list
        self.edges = []  # List of all edges
        self.vertices = set()  # Set of all vertices
        self.weights = {}  # Dictionary to store weights
        self.output_dir = "output"
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def calculate_average_degree(self):
        """
        Calculate the average degree of the graph
        Returns both the average degree and a degree distribution dictionary
        """
        if not self.vertices:
            return 0.0, {}
        
        degree_dist = {}
        sum_degrees = 0
        
        for vertex in self.vertices:
            degree = len(self.graph[vertex])
            degree_dist[vertex] = degree
            sum_degrees += degree
        
        average_degree = sum_degrees / len(self.vertices)
        
        # Also verify using the edge count formula (2E/N for undirected graphs)
        edge_count_avg_degree = (2 * len(self.edges)) / len(self.vertices) if self.vertices else 0
        
        # They should be equal - we'll use this for verification
        assert abs(average_degree - edge_count_avg_degree) < 1e-9, "Degree calculation mismatch"
        
        return average_degree, degree_dist
    
    def analyze_degrees(self):
        """
        Analyze and visualize degree distribution
        """
        avg_degree, degree_dist = self.calculate_average_degree()
        
        # Save degree information to file
        degree_file = os.path.join(self.output_dir, "degree_analysis.txt")
        with open(degree_file, 'w') as f:
            f.write(f"Graph Degree Analysis\n")
            f.write(f"====================\n")
            f.write(f"Number of vertices: {len(self.vertices)}\n")
            f.write(f"Number of edges: {len(self.edges)}\n")
            f.write(f"Average degree: {avg_degree:.4f}\n\n")
            f.write("Degree distribution:\n")
            for vertex, degree in sorted(degree_dist.items(), key=lambda x: x[1], reverse=True):
                f.write(f"Vertex {vertex}: degree {degree}\n")
        
        # Plot degree distribution
        degrees = list(degree_dist.values())
        
        plt.figure(figsize=(12, 6))
        
        # Histogram
        plt.subplot(1, 2, 1)
        plt.hist(degrees, bins=range(min(degrees), max(degrees) + 1), 
                edgecolor='black', alpha=0.7)
        plt.axvline(avg_degree, color='r', linestyle='dashed', linewidth=1)
        plt.text(avg_degree + 0.5, plt.ylim()[1]*0.9, 
                f'Avg: {avg_degree:.2f}', color='red')
        plt.title('Degree Distribution Histogram')
        plt.xlabel('Degree')
        plt.ylabel('Number of vertices')
        
        # Log-log plot (for scale-free networks)
        plt.subplot(1, 2, 2)
        degree_counts = defaultdict(int)
        for d in degrees:
            degree_counts[d] += 1
        x = sorted(degree_counts.keys())
        y = [degree_counts[d] for d in x]
        plt.loglog(x, y, 'bo')
        plt.title('Degree Distribution (log-log)')
        plt.xlabel('Degree')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plot_file = os.path.join(self.output_dir, "degree_distribution.png")
        plt.savefig(plot_file, dpi=300)
        plt.close()
        
        print(f"Degree analysis saved to {degree_file}")
        print(f"Degree distribution plot saved to {plot_file}")
        
        return avg_degree, degree_dist
    
    def visualize_mst_first_100_nodes(self, mst_edges, algorithm_name):
        """Visualize the MST for first 100 nodes using networkx and matplotlib"""
        # Get the first 100 nodes from MST edges
        all_nodes = set()
        for u, v, _ in mst_edges:
            all_nodes.add(u)
            all_nodes.add(v)
            if len(all_nodes) >= 100:
                break
        
        # Create a subgraph with these nodes
        G = nx.Graph()
        included_edges = []
        
        for u, v, weight in mst_edges:
            if u in all_nodes and v in all_nodes:
                G.add_edge(u, v, weight=weight)
                included_edges.append((u, v, weight))
            if len(G.nodes) >= 100:
                break
        
        if not G.nodes:
            print("Not enough nodes in MST to visualize")
            return
        
        # Position nodes using spring layout
        pos = nx.spring_layout(G, k=0.15, iterations=50)  # Adjust k for better spacing
        
        plt.figure(figsize=(20, 12))
        
        # Draw the graph
        nx.draw_networkx_nodes(G, pos, node_size=200, node_color='lightblue')
        nx.draw_networkx_edges(G, pos, width=1.5, edge_color='gray')
        
        # Draw labels only for some nodes to reduce clutter
        labels = {}
        for i, node in enumerate(G.nodes()):
            if i % 10 == 0:  # Label every 10th node
                labels[node] = node
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        # Draw edge weights for some edges
        edge_labels = {}
        for i, (u, v, d) in enumerate(included_edges):
            if i % 20 == 0:  # Label every 20th edge
                edge_labels[(u, v)] = d['weight'] if isinstance(d, dict) else d
        
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        
        plt.title(f'MST (First 100 Nodes) using {algorithm_name} Algorithm')
        plt.axis('off')
        
        # Save the visualization
        filename = os.path.join(self.output_dir, f'mst_first_100_{algorithm_name.lower()}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"MST visualization (first 100 nodes) saved to {filename}")

    def load_from_file(self, file_path):
        """Load graph from file and assign random weights"""
        edges_set = set()  # To avoid duplicate edges
        
        with open(file_path, 'r') as file:
            for line in file:
                if line.startswith('#'):  # Skip comments
                    continue
                    
                parts = line.strip().split()
                if len(parts) == 2:
                    u, v = int(parts[0]), int(parts[1])
                    
                    # For undirected graph, ensure we don't add duplicates
                    edge = tuple(sorted([u, v]))
                    if edge not in edges_set:
                        # Assign random weight between 1-9
                        weight = random.randint(1, 9)
                        
                        # Add edge to the graph
                        self.graph[u].append((v, weight))
                        self.graph[v].append((u, weight))
                        
                        # Add to list of edges
                        self.edges.append((u, v, weight))
                        
                        # Store the weight
                        self.weights[(u, v)] = weight
                        self.weights[(v, u)] = weight
                        
                        # Add to set of edges
                        edges_set.add(edge)
                        
                        # Add vertices
                        self.vertices.add(u)
                        self.vertices.add(v)
    
    def is_connected(self):
        """Check if the graph is connected using BFS"""
        if not self.vertices:
            return False
        
        start_vertex = next(iter(self.vertices))
        visited = set([start_vertex])
        queue = [start_vertex]
        
        while queue:
            vertex = queue.pop(0)
            for neighbor, _ in self.graph[vertex]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return len(visited) == len(self.vertices)
    
    def prim_mst(self):
        """Implement Prim's algorithm for MST"""
        if not self.is_connected():
            print("Graph is not connected. Cannot find MST.")
            return [], 0
        
        start_time = time.time()
        
        # Start with any vertex
        start_vertex = next(iter(self.vertices))
        
        # Track vertices in MST
        mst_vertices = set([start_vertex])
        
        # Store MST edges
        mst_edges = []
        
        # Priority queue for edges
        pq = []
        
        # Add all edges from start vertex to priority queue
        for neighbor, weight in self.graph[start_vertex]:
            heapq.heappush(pq, (weight, start_vertex, neighbor))
        
        # While there are vertices to add to MST
        while pq and len(mst_vertices) < len(self.vertices):
            # Get the minimum weight edge
            weight, u, v = heapq.heappop(pq)
            
            # If v is already in MST, skip
            if v in mst_vertices:
                continue
            
            # Add v to MST
            mst_vertices.add(v)
            
            # Add edge to MST
            mst_edges.append((u, v, weight))
            
            # Add all edges from v to priority queue
            for neighbor, w in self.graph[v]:
                if neighbor not in mst_vertices:
                    heapq.heappush(pq, (w, v, neighbor))
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Check if we found an MST
        if len(mst_vertices) != len(self.vertices):
            print("Could not find a complete MST. Graph might not be connected.")
            return [], 0
        
        # Calculate total weight
        total_weight = sum(weight for _, _, weight in mst_edges)
        
        if mst_edges:
            self.visualize_mst(mst_edges, "Prim")
            self.visualize_mst_first_100_nodes(mst_edges, "Prim")

        return mst_edges, execution_time
    
    def kruskal_mst(self):
        """Implement Kruskal's algorithm for MST"""
        if not self.is_connected():
            print("Graph is not connected. Cannot find MST.")
            return [], 0
            
        start_time = time.time()
        
        # Sort all edges by weight
        edges = sorted(self.edges, key=lambda x: x[2])
        
        # Initialize parent and rank for disjoint set
        parent = {v: v for v in self.vertices}
        rank = {v: 0 for v in self.vertices}
        
        # Find operation with path compression
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])  # Path compression
            return parent[x]
        
        # Union operation with rank
        def union(x, y):
            root_x = find(x)
            root_y = find(y)
            
            if root_x == root_y:
                return
            
            # Attach smaller rank tree under root of higher rank tree
            if rank[root_x] < rank[root_y]:
                parent[root_x] = root_y
            elif rank[root_x] > rank[root_y]:
                parent[root_y] = root_x
            else:
                parent[root_y] = root_x
                rank[root_x] += 1
        
        # MST edges
        mst_edges = []
        
        # Process all edges
        for u, v, weight in edges:
            # If including this edge doesn't form a cycle, include it in MST
            if find(u) != find(v):
                union(u, v)
                mst_edges.append((u, v, weight))
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Calculate total weight
        total_weight = sum(weight for _, _, weight in mst_edges)
        
        
        if mst_edges:
            self.visualize_mst(mst_edges, "Kruskal")
            self.visualize_mst_first_100_nodes(mst_edges, "Kruskal")
        
        return mst_edges, execution_time
        
    def shortest_paths(self, source):
        """Find shortest paths from source to all vertices using Dijkstra's algorithm"""
        # Initialize distances and predecessors
        distances = {vertex: float('infinity') for vertex in self.vertices}
        predecessors = {vertex: None for vertex in self.vertices}
        distances[source] = 0
        
        # Priority queue
        pq = [(0, source)]
        
        while pq:
            # Get vertex with minimum distance
            current_distance, current_vertex = heapq.heappop(pq)
            
            # If we found a longer path, skip
            if current_distance > distances[current_vertex]:
                continue
            
            # Process all neighbors
            for neighbor, weight in self.graph[current_vertex]:
                distance = current_distance + weight
                
                # If we found a shorter path to neighbor
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    predecessors[neighbor] = current_vertex
                    heapq.heappush(pq, (distance, neighbor))
        
        return distances, predecessors
    
    def reconstruct_paths(self, source, distances, predecessors):
        """Reconstruct and format all shortest paths from the source node"""
        output = []
        output.append(f"Shortest paths from node {source}:")
        
        for node in sorted(distances.keys()):
            if node == source:
                continue
                
            if distances[node] == float('infinity'):
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
        
        return output
    
    def visualize_shortest_path(self, source, target, path, distances):
        """Visualize the shortest path between source and target nodes"""
        # Create a NetworkX graph for visualization
        G = nx.Graph()
        
        # Add all edges with weights
        for u, v, weight in self.edges:
            G.add_edge(u, v, weight=weight)
        
        # Extract the subgraph containing the path plus some neighbors
        path_set = set(path)
        extended_nodes = set(path)
        
        # Add some neighbors of the path nodes
        for node in path:
            neighbors = list(G.neighbors(node))
            sample_size = min(2, len(neighbors))  # Add up to 2 neighbors for each node
            extended_nodes.update(random.sample(neighbors, sample_size))
        
        subgraph = G.subgraph(extended_nodes)
        
        # Get positions for nodes
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
        nx.draw_networkx_nodes(subgraph, pos, nodelist=[source], node_color='red', node_size=500)
        nx.draw_networkx_nodes(subgraph, pos, nodelist=[target], node_color='purple', node_size=500)
        
        # Draw edge weights on the path
        edge_labels = {(u, v): subgraph[u][v]['weight'] for u, v in path_edges}
        nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, font_size=10)
        
        # Draw node labels
        nx.draw_networkx_labels(subgraph, pos)
        
        plt.title(f"Shortest Path from {source} to {target} (Distance: {distances[target]})")
        plt.axis('off')
        plt.tight_layout()
        
        # Save the visualization
        filename = os.path.join(self.output_dir, f'shortest_path_{source}_to_{target}.png')
        plt.savefig(filename, dpi=500)
        plt.close()
        
        print(f"Shortest path visualization saved to {filename}")
    
    def find_and_visualize_shortest_paths(self, source):
        """Find shortest paths and visualize sample paths"""
        print(f"\nFinding shortest paths from node {source}...")
        distances, predecessors = self.shortest_paths(source)
        
        # Reconstruct and save all paths
        paths_output = self.reconstruct_paths(source, distances, predecessors)
        paths_file = os.path.join(self.output_dir, f'shortest_paths_from_{source}.txt')
        with open(paths_file, 'w') as f:
            f.write("\n".join(paths_output))
        print(f"All shortest paths saved to {paths_file}")
        
        # Select target nodes at different distances
        sorted_distances = sorted([(node, dist) for node, dist in distances.items()], key=lambda x: x[1])
        targets = []
        
        if len(sorted_distances) > 3:
            # Add a close node
            targets.append(sorted_distances[min(5, len(sorted_distances)-1)][0])
            # Add a medium-distance node
            targets.append(sorted_distances[len(sorted_distances)//2][0])
            # Add a far node
            targets.append(sorted_distances[-5][0])
        else:
            targets = [node for node, _ in sorted_distances if node != source][:3]
        
        # Visualize paths to selected targets
        for target in targets:
            if target != source:
                print(f"Visualizing path to node {target}...")
                path = []
                current = target
                while current is not None:
                    path.append(current)
                    current = predecessors[current]
                path.reverse()
                self.visualize_shortest_path(source, target, path, distances)
    
    def visualize_mst(self, mst_edges, algorithm_name):
        """Visualize the MST using networkx and matplotlib"""
        G = nx.Graph()
        
        # Add edges to the graph
        for u, v, weight in mst_edges:
            G.add_edge(u, v, weight=weight)
        
        # Position nodes using spring layout
        pos = nx.spring_layout(G)
        
        # Draw the graph
        plt.figure(figsize=(12, 8))
        nx.draw_networkx_nodes(G, pos, node_size=200, node_color='lightblue')
        nx.draw_networkx_edges(G, pos, width=1.5, edge_color='gray')
        nx.draw_networkx_labels(G, pos, font_size=8)
        
        # Draw edge labels
        edge_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        
        plt.title(f'MST using {algorithm_name} Algorithm')
        plt.axis('off')
        
        # Save the visualization
        filename = os.path.join(self.output_dir, f'mst_{algorithm_name.lower()}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"MST visualization saved to {filename}")
    
    def analyze_shortest_paths(self, source):
        """Analyze shortest paths from the given source vertex"""
        if not self.is_connected():
            print("Graph is not connected. Cannot analyze shortest paths.")
            return
        
        print(f"Analyzing shortest paths from source node {source}...")
        distances, _ = self.shortest_paths(source)
        
        # Prepare data for visualization
        reachable_nodes = [v for v, d in distances.items() if d != float('infinity')]
        unreachable_nodes = [v for v, d in distances.items() if d == float('infinity')]
        
        reachable_distances = [d for d in distances.values() if d != float('infinity')]
        
        # Plot distance distribution
        plt.figure(figsize=(10, 6))
        plt.hist(reachable_distances, bins=20, edgecolor='black')
        plt.title(f'Distribution of Shortest Path Distances from Node {source}')
        plt.xlabel('Distance')
        plt.ylabel('Frequency')
        distance_plot = os.path.join(self.output_dir, f'distance_distribution_{source}.png')
        plt.savefig(distance_plot)
        plt.close()
        
        # Plot reachability pie chart
        labels = ['Reachable', 'Unreachable']
        sizes = [len(reachable_nodes), len(unreachable_nodes)]
        colors = ['#66b3ff', '#ff9999']
        
        plt.figure(figsize=(8, 8))
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title(f'Reachability from Node {source}')
        reachability_plot = os.path.join(self.output_dir, f'reachability_pie_{source}.png')
        plt.savefig(reachability_plot)
        plt.close()
        
        print(f"Distance distribution saved to {distance_plot}")
        print(f"Reachability chart saved to {reachability_plot}")
    
    def compare_algorithms(self):
        """Compare Prim's and Kruskal's algorithms"""
        print("Comparing Prim's and Kruskal's algorithms...")
        
        # Run Prim's algorithm
        prim_edges, prim_time = self.prim_mst()
        prim_weight = sum(weight for _, _, weight in prim_edges)
        
        # Run Kruskal's algorithm
        kruskal_edges, kruskal_time = self.kruskal_mst()
        kruskal_weight = sum(weight for _, _, weight in kruskal_edges)
        
        # Visualize both MSTs (full and first 100 nodes)
        if prim_edges:
            self.visualize_mst(prim_edges, "Prim")
            self.visualize_mst_first_100_nodes(prim_edges, "Prim")
        if kruskal_edges:
            self.visualize_mst(kruskal_edges, "Kruskal")
            self.visualize_mst_first_100_nodes(kruskal_edges, "Kruskal")
        
        # Write results to file
        with open(os.path.join(self.output_dir, "mst_comparison.txt"), "w") as file:
            file.write(f"Prim's Algorithm:\n")
            file.write(f"Execution time: {prim_time} seconds\n")
            file.write(f"MST weight: {prim_weight}\n\n")
            
            file.write(f"Kruskal's Algorithm:\n")
            file.write(f"Execution time: {kruskal_time} seconds\n")
            file.write(f"MST weight: {kruskal_weight}\n")
        
        # Plot execution times
        algorithms = ['Prim', 'Kruskal']
        times = [prim_time, kruskal_time]
        
        plt.figure(figsize=(10, 6))
        plt.bar(algorithms, times, color=['#66b3ff', '#ff9999'])
        plt.title('Execution Time Comparison')
        plt.xlabel('Algorithm')
        plt.ylabel('Time (seconds)')
        comparison_plot = os.path.join(self.output_dir, 'algorithm_comparison.png')
        plt.savefig(comparison_plot)
        plt.close()
        
        print(f"Comparison results saved to {os.path.join(self.output_dir, 'mst_comparison.txt')}")
        print(f"Comparison chart saved to {comparison_plot}")
        
        return prim_edges, kruskal_edges, prim_time, kruskal_time
    
    def run_all_analyses(self, source):
        """Run all analyses and save results"""
        print(f"Running all analyses with source node {source}...")

        # Run MST algorithms
        print("Running MST algorithms...")
        prim_edges, prim_time = self.prim_mst()
        kruskal_edges, kruskal_time = self.kruskal_mst()
        
        if prim_edges:
            prim_weight = sum(weight for _, _, weight in prim_edges)
            print(f"Prim's Algorithm: {prim_time:.6f} seconds, weight: {prim_weight}")
            
            with open(os.path.join(self.output_dir, "prim_mst.txt"), "w") as file:
                file.write(f"MST edges ({len(prim_edges)}):\n")
                for u, v, weight in prim_edges:
                    file.write(f"{u} -- {v} : {weight}\n")
                file.write(f"\nTotal weight: {prim_weight}\n")
                file.write(f"Execution time: {prim_time} seconds\n")
            
            self.visualize_mst(prim_edges, "Prim")
        
        if kruskal_edges:
            kruskal_weight = sum(weight for _, _, weight in kruskal_edges)
            print(f"Kruskal's Algorithm: {kruskal_time:.6f} seconds, weight: {kruskal_weight}")
            
            with open(os.path.join(self.output_dir, "kruskal_mst.txt"), "w") as file:
                file.write(f"MST edges ({len(kruskal_edges)}):\n")
                for u, v, weight in kruskal_edges:
                    file.write(f"{u} -- {v} : {weight}\n")
                file.write(f"\nTotal weight: {kruskal_weight}\n")
                file.write(f"Execution time: {kruskal_time} seconds\n")
            
            self.visualize_mst(kruskal_edges, "Kruskal")
        
        # Compare algorithms
        print("Comparing algorithms...")
        self.compare_algorithms()
        
        # Analyze shortest paths
        print(f"Analyzing shortest paths from source node {source}...")
        self.analyze_shortest_paths(source)
        
        # Find and visualize specific shortest paths
        self.find_and_visualize_shortest_paths(source)

         # First analyze graph properties
        print("\nAnalyzing graph properties...")
        avg_degree, _ = self.analyze_degrees()
        print(f"Average degree: {avg_degree:.2f}")
        
        print("All analyses completed. Results saved in the 'output' directory.")

def display_menu():
    print("\nSelect an option:")
    print("1. Run Prim's Algorithm")
    print("2. Run Kruskal's Algorithm")
    print("3. Compare MST Algorithms")
    print("4. Analyze Shortest Paths")
    print("5. Visualize Specific Shortest Paths")
    print("6. Analyze Degree Distribution")
    print("7. Run All Analyses")
    print("8. Exit")

def main():
    graph = Graph()
    
    print("Loading graph from file...")
    graph.load_from_file("oregon1_010331.txt")
    print(f"Loaded graph with {len(graph.vertices)} vertices and {len(graph.edges)} edges.")
    
    if not graph.is_connected():
        print("Warning: The graph is not connected. Some algorithms may not work correctly.")
    
    # Get source node from user
    while True:
        try:
            default_node = next(iter(graph.vertices))
            source_node = int(input(f"Enter source node (or press Enter for default node {default_node}): ") or str(default_node))
            if source_node in graph.vertices:
                break
            else:
                print(f"Node {source_node} not found in graph. Please enter a valid node.")
        except ValueError:
            print("Please enter a valid integer.")
    
    while True:
        display_menu()
        choice = input("Enter your choice (1-8): ")
        
        if choice == '1':
            print("Running Prim's Algorithm...")
            mst_edges, execution_time = graph.prim_mst()
            
            if mst_edges:
                total_weight = sum(weight for _, _, weight in mst_edges)
                print(f"MST found with {len(mst_edges)} edges and total weight {total_weight}")
                print(f"Execution time: {execution_time} seconds")
                print(f"MST edges saved to {os.path.join(graph.output_dir, 'prim_mst.txt')}")
                graph.visualize_mst(mst_edges, "Prim")
        
        elif choice == '2':
            print("Running Kruskal's Algorithm...")
            mst_edges, execution_time = graph.kruskal_mst()
            
            if mst_edges:
                total_weight = sum(weight for _, _, weight in mst_edges)
                print(f"MST found with {len(mst_edges)} edges and total weight {total_weight}")
                print(f"Execution time: {execution_time} seconds")
                print(f"MST edges saved to {os.path.join(graph.output_dir, 'kruskal_mst.txt')}")
                graph.visualize_mst(mst_edges, "Kruskal")
        
        elif choice == '3':
            print("Comparing MST Algorithms...")
            prim_edges, kruskal_edges, prim_time, kruskal_time = graph.compare_algorithms()
            
            if prim_edges and kruskal_edges:
                prim_weight = sum(weight for _, _, weight in prim_edges)
                kruskal_weight = sum(weight for _, _, weight in kruskal_edges)
                
                print(f"Prim's Algorithm: {prim_time:.6f} seconds, weight: {prim_weight}")
                print(f"Kruskal's Algorithm: {kruskal_time:.6f} seconds, weight: {kruskal_weight}")
        
        elif choice == '4':
            print(f"Analyzing Shortest Paths from node {source_node}...")
            graph.analyze_shortest_paths(source_node)
        
        elif choice == '5':
            print(f"Finding and visualizing shortest paths from node {source_node}...")
            graph.find_and_visualize_shortest_paths(source_node)
        
        elif choice == '6':
            print("Analyzing degree distribution...")
            avg_degree, _ = graph.analyze_degrees()
            print(f"Average degree: {avg_degree:.2f}")
        
        elif choice == '7':
            graph.run_all_analyses(source_node)
        
        elif choice == '8':
            print("Exiting program...")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()