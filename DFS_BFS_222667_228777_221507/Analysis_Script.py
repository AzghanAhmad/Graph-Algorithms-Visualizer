import time
import matplotlib.pyplot as plt
import os
import csv
import random
from collections import defaultdict, deque

class Graph:
    def __init__(self):
        self.graph = defaultdict(list)
        self.nodes = set()
        self.edges = 0
    
    def add_edge(self, u, v):
        # For undirected graph, add edge in both directions
        self.graph[u].append(v)
        self.graph[v].append(u)
        self.nodes.add(u)
        self.nodes.add(v)
        self.edges += 1
    
    def load_graph_from_file(self, filename):
        """Load graph from a file, supporting both plain text and gzipped files."""
        try:
            # Check if file is gzipped
            if filename.endswith('.gz'):
                import gzip
                with gzip.open(filename, 'rt') as f:  # 'rt' mode for text reading
                    for line in f:
                        # Skip comment lines
                        if line.startswith('#'):
                            continue
                        
                        # Parse the line to get the two nodes of an edge
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            u, v = int(parts[0]), int(parts[1])
                            self.add_edge(u, v)
            else:
                # Regular text file
                with open(filename, 'r') as f:
                    for line in f:
                        # Skip comment lines
                        if line.startswith('#'):
                            continue
                        
                        # Parse the line to get the two nodes of an edge
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            u, v = int(parts[0]), int(parts[1])
                            self.add_edge(u, v)
            
            print(f"Graph loaded successfully. Nodes: {len(self.nodes)}, Edges: {self.edges//2}")
            return True
        except Exception as e:
            print(f"Error loading graph: {e}")
            return False
    
    def get_subgraph(self, num_nodes):
        """Extract a subgraph with approximately num_nodes nodes."""
        if num_nodes >= len(self.nodes):
            return self
        
        subgraph = Graph()
        # Select random nodes for the subgraph
        selected_nodes = random.sample(list(self.nodes), num_nodes)
        
        # Add edges between selected nodes
        for u in selected_nodes:
            for v in self.graph[u]:
                if v in selected_nodes:
                    subgraph.add_edge(u, v)
        
        return subgraph
    
    def bfs(self, start_node):
        """Perform BFS traversal from a given starting node."""
        if start_node not in self.nodes:
            return []
        
        visited = {node: False for node in self.nodes}
        queue = deque([start_node])
        visited[start_node] = True
        bfs_result = []
        
        while queue:
            current = queue.popleft()
            bfs_result.append(current)
            
            for neighbor in self.graph[current]:
                if not visited[neighbor]:
                    queue.append(neighbor)
                    visited[neighbor] = True
        
        return bfs_result
    
    def bfs_with_list(self, start_node):
        """Perform BFS using a list instead of deque for analysis purposes."""
        if start_node not in self.nodes:
            return []
        
        visited = {node: False for node in self.nodes}
        queue = [start_node]  # Using a list instead of deque
        visited[start_node] = True
        bfs_result = []
        
        while queue:
            current = queue.pop(0)  # This is O(n) for lists, compared to O(1) for deque
            bfs_result.append(current)
            
            for neighbor in self.graph[current]:
                if not visited[neighbor]:
                    queue.append(neighbor)
                    visited[neighbor] = True
        
        return bfs_result
    
    def dfs(self, start_node):
        """Perform DFS traversal from a given starting node."""
        if start_node not in self.nodes:
            return []
        
        visited = {node: False for node in self.nodes}
        stack = [start_node]
        dfs_result = []
        
        while stack:
            current = stack.pop()
            
            if not visited[current]:
                visited[current] = True
                dfs_result.append(current)
                
                # Push neighbors to stack in reverse order
                neighbors = sorted(self.graph[current], reverse=True)
                for neighbor in neighbors:
                    if not visited[neighbor]:
                        stack.append(neighbor)
        
        return dfs_result
    
    def dfs_recursive(self, start_node):
        """Perform DFS using recursion instead of an explicit stack."""
        if start_node not in self.nodes:
            return []
        
        visited = {node: False for node in self.nodes}
        result = []
        
        def dfs_util(node):
            visited[node] = True
            result.append(node)
            
            for neighbor in self.graph[node]:
                if not visited[neighbor]:
                    dfs_util(neighbor)
        
        dfs_util(start_node)
        return result
    
    def detect_cycle(self):
        """Detect if there is a cycle in the undirected graph using DFS."""
        visited = {node: False for node in self.nodes}
        
        def dfs_cycle(node, parent):
            visited[node] = True
            
            for neighbor in self.graph[node]:
                # If neighbor is not visited, then check if subtree has a cycle
                if not visited[neighbor]:
                    if dfs_cycle(neighbor, node):
                        return True
                # If an adjacent vertex is visited and not the parent of current vertex,
                # then there is a cycle
                elif neighbor != parent:
                    return True
            
            return False
        
        # Check for cycles starting from each unvisited node
        for node in self.nodes:
            if not visited[node]:
                if dfs_cycle(node, -1):  # -1 indicates no parent
                    return True
        
        return False
    
    def detect_cycle_using_bfs(self):
        """Detect if there is a cycle in the undirected graph using BFS."""
        visited = {node: False for node in self.nodes}
        
        for start_node in self.nodes:
            if visited[start_node]:
                continue
                
            # Store node and its parent in queue
            queue = deque([(start_node, -1)])  # -1 indicates no parent
            visited[start_node] = True
            
            while queue:
                node, parent = queue.popleft()
                
                for neighbor in self.graph[node]:
                    # If neighbor is not visited, mark it visited and push to queue
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append((neighbor, node))
                    # If adjacent vertex is visited and not the parent of current vertex,
                    # then there is a cycle
                    elif neighbor != parent:
                        return True
        
        return False
    
    def calculate_diameter(self):
        """
        Calculate the diameter of the graph (longest shortest path between any two nodes).
        Returns the diameter value.
        
        For efficiency in the analysis script, we'll use a faster approach:
        1. Pick a random starting node
        2. Find the farthest node from it using BFS (call it node A)
        3. Find the farthest node from node A using BFS (call it node B)
        4. The distance between A and B is the diameter
        
        This is a more efficient approximation that works well for most graphs.
        """
        if not self.nodes:
            return 0
            
        # Choose a random node that exists in all subgraphs
        start_node = random.choice(list(self.nodes))
        
        # Find the farthest node from start_node (call it 'far_node')
        far_node, max_dist = self._get_farthest_node(start_node)
        
        # From far_node, find the farthest node again
        end_node, diameter = self._get_farthest_node(far_node)
        
        return diameter
    
    def _get_farthest_node(self, start_node):
        """Helper function to find the farthest node from a given start node using BFS."""
        if start_node not in self.nodes:
            return None, 0
            
        distances = {}  # Dictionary to store distances from start_node
        visited = {node: False for node in self.nodes}
        queue = deque([(start_node, 0)])  # Store (node, distance) pairs
        visited[start_node] = True
        
        while queue:
            current, dist = queue.popleft()
            distances[current] = dist
            
            for neighbor in self.graph[current]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append((neighbor, dist + 1))
        
        # Find the maximum distance and corresponding node
        if distances:
            max_dist = max(distances.values())
            max_dist_node = max(distances, key=distances.get)
            return max_dist_node, max_dist
        
        return None, 0
    
    def calculate_average_degree(self):
        """Calculate the average degree of nodes in the graph."""
        if not self.nodes:
            return 0
        
        total_degree = sum(len(self.graph[node]) for node in self.nodes)
        return total_degree / len(self.nodes)

def analyze_algorithms():
    """Analyze the performance of BFS, DFS, and cycle detection algorithms."""
    graph = Graph()
    
    # Use the specified dataset file directly
    dataset_file = 'oregon1_010331.txt.gz'
    print(f"Loading dataset from: {dataset_file}")
    if not graph.load_graph_from_file(dataset_file):
        print("Failed to load the graph. Exiting...")
        return
    
    # Create results directory if it doesn't exist
    if not os.path.exists("analysis_results"):
        os.makedirs("analysis_results")
    
    # Sizes of subgraphs to analyze
    sizes = [100, 200, 400, 600, 800, 1000]
    
    # Choose a random node that exists in all subgraphs
    start_node = random.choice(list(graph.nodes))
    
    # Lists to store execution times
    bfs_times = []
    bfs_list_times = []  # For BFS using list instead of deque
    dfs_times = []
    dfs_recursive_times = []  # For recursive DFS
    cycle_dfs_times = []
    cycle_bfs_times = []  # For cycle detection using BFS
    diameter_times = []
    avg_degrees = []
    diameters = []
    
    print(f"Starting analysis with source node: {start_node}")
    
    for size in sizes:
        print(f"Analyzing with {size} nodes...")
        
        # Get subgraph
        if size < len(graph.nodes):
            subgraph = graph.get_subgraph(size)
        else:
            subgraph = graph
        
        # Make sure the start node is in the subgraph
        if start_node not in subgraph.nodes and subgraph.nodes:
            start_node = next(iter(subgraph.nodes))
        
        # Calculate average degree
        avg_degree = subgraph.calculate_average_degree()
        avg_degrees.append(avg_degree)
        
        # Measure BFS time with deque
        start_time = time.time()
        subgraph.bfs(start_node)
        bfs_time = time.time() - start_time
        bfs_times.append(bfs_time)
        
        # Measure BFS time with list
        start_time = time.time()
        subgraph.bfs_with_list(start_node)
        bfs_list_time = time.time() - start_time
        bfs_list_times.append(bfs_list_time)
        
        # Measure DFS time with stack
        start_time = time.time()
        subgraph.dfs(start_node)
        dfs_time = time.time() - start_time
        dfs_times.append(dfs_time)
        
        # Measure DFS time with recursion
        start_time = time.time()
        subgraph.dfs_recursive(start_node)
        dfs_recursive_time = time.time() - start_time
        dfs_recursive_times.append(dfs_recursive_time)
        
        # Measure Cycle Detection time with DFS
        start_time = time.time()
        has_cycle_dfs = subgraph.detect_cycle()
        cycle_dfs_time = time.time() - start_time
        cycle_dfs_times.append(cycle_dfs_time)
        
        # Measure Cycle Detection time with BFS
        start_time = time.time()
        has_cycle_bfs = subgraph.detect_cycle_using_bfs()
        cycle_bfs_time = time.time() - start_time
        cycle_bfs_times.append(cycle_bfs_time)
        
        # Measure Diameter Calculation time
        start_time = time.time()
        diameter = subgraph.calculate_diameter()
        diameter_time = time.time() - start_time
        diameter_times.append(diameter_time)
        diameters.append(diameter)
        
        print(f"  Nodes: {len(subgraph.nodes)}, Edges: {subgraph.edges//2}")
        print(f"  Average Degree: {avg_degree:.2f}")
        print(f"  Graph Diameter: {diameter}")
        print(f"  BFS Time (deque): {bfs_time:.6f}s, BFS Time (list): {bfs_list_time:.6f}s")
        print(f"  DFS Time (stack): {dfs_time:.6f}s, DFS Time (recursive): {dfs_recursive_time:.6f}s")
        print(f"  Cycle Detection Time (DFS): {cycle_dfs_time:.6f}s, (BFS): {cycle_bfs_time:.6f}s")
        print(f"  Has Cycle: {has_cycle_dfs} (DFS), {has_cycle_bfs} (BFS)")
    
    # Create plots
    plt.figure(figsize=(20, 15))
    
    # Plot 1: BFS vs DFS comparison
    plt.subplot(2, 2, 1)
    plt.plot(sizes, bfs_times, 'o-', label='BFS (deque)')
    plt.plot(sizes, dfs_times, 's-', label='DFS (stack)')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Execution Time (seconds)')
    plt.title('BFS vs DFS: Execution Time Comparison')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Implementation Impact (Queue vs List for BFS, Stack vs Recursion for DFS)
    plt.subplot(2, 2, 2)
    plt.plot(sizes, bfs_times, 'o-', label='BFS (deque)')
    plt.plot(sizes, bfs_list_times, 'o--', label='BFS (list)')
    plt.plot(sizes, dfs_times, 's-', label='DFS (stack)')
    plt.plot(sizes, dfs_recursive_times, 's--', label='DFS (recursive)')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Impact of Data Structure Implementation')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Cycle Detection - DFS vs BFS
    plt.subplot(2, 2, 3)
    plt.plot(sizes, cycle_dfs_times, '^-', label='Cycle Detection (DFS)')
    plt.plot(sizes, cycle_bfs_times, '^--', label='Cycle Detection (BFS)')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Cycle Detection: DFS vs BFS')
    plt.legend()
    plt.grid(True)
    
    # Plot 4: All algorithms together
    plt.subplot(2, 2, 4)
    plt.plot(sizes, bfs_times, 'o-', label='BFS')
    plt.plot(sizes, dfs_times, 's-', label='DFS')
    plt.plot(sizes, cycle_dfs_times, '^-', label='Cycle Detection (DFS)')
    plt.plot(sizes, diameter_times, 'd-', label='Diameter Calculation')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Execution Time (seconds)')
    plt.title('All Algorithms: Execution Time Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('analysis_results/algorithm_comparison.png')
    
    # Additional plots for more detailed data structure analysis
    plt.figure(figsize=(15, 10))
    
    # Plot 5: BFS - Queue Performance Impact
    plt.subplot(2, 2, 1)
    plt.plot(sizes, bfs_times, 'o-', label='BFS (deque - O(1) operations)')
    plt.plot(sizes, bfs_list_times, 'o--', label='BFS (list - O(n) pop operations)')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Execution Time (seconds)')
    plt.title('BFS: Impact of Queue Implementation')
    plt.legend()
    plt.grid(True)
    
    # Plot 6: DFS - Stack Implementation Impact
    plt.subplot(2, 2, 2)
    plt.plot(sizes, dfs_times, 's-', label='DFS (explicit stack)')
    plt.plot(sizes, dfs_recursive_times, 's--', label='DFS (recursive - implicit stack)')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Execution Time (seconds)')
    plt.title('DFS: Stack vs Recursion')
    plt.legend()
    plt.grid(True)
    
    # Plot 7: Average Degree vs Execution Time
    plt.subplot(2, 2, 3)
    plt.scatter(avg_degrees, bfs_times, marker='o', label='BFS')
    plt.scatter(avg_degrees, dfs_times, marker='s', label='DFS')
    plt.scatter(avg_degrees, cycle_dfs_times, marker='^', label='Cycle Detection')
    plt.xlabel('Average Degree')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Algorithm Performance vs Graph Density')
    plt.legend()
    plt.grid(True)
    
    # Plot 8: Graph Diameter vs Node Count
    plt.subplot(2, 2, 4)
    plt.plot(sizes, diameters, 'o-', color='purple')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Diameter')
    plt.title('Graph Diameter vs. Number of Nodes')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('analysis_results/data_structure_analysis.png')
    
    # Save analysis data to CSV
    with open('analysis_results/performance_data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Nodes', 'Average Degree', 'Diameter', 
                         'BFS Time (deque)', 'BFS Time (list)',
                         'DFS Time (stack)', 'DFS Time (recursive)',
                         'Cycle Detection Time (DFS)', 'Cycle Detection Time (BFS)',
                         'Diameter Calculation Time'])
        for i in range(len(sizes)):
            writer.writerow([sizes[i], avg_degrees[i], diameters[i], 
                             bfs_times[i], bfs_list_times[i],
                             dfs_times[i], dfs_recursive_times[i],
                             cycle_dfs_times[i], cycle_bfs_times[i],
                             diameter_times[i]])
    
    print("Analysis completed. Results saved to 'analysis_results' directory.")

if __name__ == "__main__":
    analyze_algorithms()