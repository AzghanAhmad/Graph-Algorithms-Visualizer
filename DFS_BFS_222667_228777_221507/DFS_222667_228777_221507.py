import time
import os
import csv
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
    
    def bfs(self, start_node):
        """Perform BFS traversal from a given starting node."""
        if start_node not in self.nodes:
            print(f"Start node {start_node} not found in graph.")
            return []
        
        visited = {node: False for node in self.nodes}
        queue = deque([start_node])
        visited[start_node] = True
        bfs_result = []
        bfs_trace = []  # For storing the trace of queue operations
        
        while queue:
            queue_state = list(queue)  # Capture current queue state for trace
            current = queue.popleft()
            bfs_result.append(current)
            bfs_trace.append(f"Dequeued: {current}, Queue after: {list(queue)}")
            
            for neighbor in self.graph[current]:
                if not visited[neighbor]:
                    queue.append(neighbor)
                    visited[neighbor] = True
                    bfs_trace.append(f"Enqueued: {neighbor}, Queue after: {list(queue)}")
        
        return bfs_result, bfs_trace
    
    def dfs(self, start_node):
        """Perform DFS traversal from a given starting node."""
        if start_node not in self.nodes:
            print(f"Start node {start_node} not found in graph.")
            return []
        
        visited = {node: False for node in self.nodes}
        stack = [start_node]
        dfs_result = []
        dfs_trace = []  # For storing the trace of stack operations
        
        dfs_trace.append(f"Pushed: {start_node}, Stack: {stack}")
        
        while stack:
            stack_state = list(stack)  # Capture current stack state for trace
            current = stack.pop()
            dfs_trace.append(f"Popped: {current}, Stack after: {stack}")
            
            if not visited[current]:
                visited[current] = True
                dfs_result.append(current)
                
                # Push neighbors to stack in reverse order to process in original order
                neighbors = sorted(self.graph[current], reverse=True)
                for neighbor in neighbors:
                    if not visited[neighbor]:
                        stack.append(neighbor)
                        dfs_trace.append(f"Pushed: {neighbor}, Stack after: {stack}")
        
        return dfs_result, dfs_trace
    
    def detect_cycle(self):
        """Detect if there is a cycle in the undirected graph using DFS."""
        visited = {node: False for node in self.nodes}
        cycle_trace = []  # For storing the trace of DFS for cycle detection
        
        def dfs_cycle(node, parent):
            visited[node] = True
            cycle_trace.append(f"Visiting node {node}, Parent: {parent}")
            
            for neighbor in self.graph[node]:
                cycle_trace.append(f"Checking edge {node} -> {neighbor}")
                
                # If neighbor is not visited, then check if subtree has a cycle
                if not visited[neighbor]:
                    cycle_trace.append(f"Neighbor {neighbor} not visited yet, recursively checking")
                    if dfs_cycle(neighbor, node):
                        return True
                # If an adjacent vertex is visited and not the parent of current vertex,
                # then there is a cycle
                elif neighbor != parent:
                    cycle_trace.append(f"Found cycle! Node {neighbor} is already visited and not the parent of {node}")
                    return True
            
            return False
        
        # Check for cycles starting from each unvisited node
        for node in self.nodes:
            if not visited[node]:
                cycle_trace.append(f"Starting DFS from node {node}")
                if dfs_cycle(node, -1):  # -1 indicates no parent
                    return True, cycle_trace
        
        return False, cycle_trace
    
    def calculate_diameter(self):
        """
        Calculate the diameter of the graph (longest shortest path between any two nodes).
        Also returns the trace of the algorithm execution.
        """
        diameter = 0
        diameter_trace = []
        diameter_nodes = (None, None)  # Nodes with the maximum distance
        
        diameter_trace.append("Starting diameter calculation algorithm...")
        
        # For each node, perform BFS to find the farthest node
        for start_node in self.nodes:
            # Skip isolated nodes (nodes with no neighbors)
            if not self.graph[start_node]:
                continue
                
            diameter_trace.append(f"\nStarting BFS from node {start_node}")
            
            # Use BFS to find shortest paths from start_node to all other nodes
            distances = {}  # Dictionary to store distances from start_node
            visited = {node: False for node in self.nodes}
            queue = deque([(start_node, 0)])  # Store (node, distance) pairs
            visited[start_node] = True
            
            while queue:
                current, dist = queue.popleft()
                distances[current] = dist
                diameter_trace.append(f"  Visited node {current} at distance {dist}")
                
                for neighbor in self.graph[current]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append((neighbor, dist + 1))
                        diameter_trace.append(f"  Enqueued {neighbor} with distance {dist + 1}")
            
            # Find the maximum distance from this start_node
            if distances:
                max_dist = max(distances.values())
                max_dist_node = max(distances, key=distances.get)
                diameter_trace.append(f"Maximum distance from node {start_node} is {max_dist} to node {max_dist_node}")
                
                # Update diameter if this distance is greater
                if max_dist > diameter:
                    diameter = max_dist
                    diameter_nodes = (start_node, max_dist_node)
                    diameter_trace.append(f"New diameter found: {diameter} between nodes {start_node} and {max_dist_node}")
        
        diameter_trace.append(f"\nFinal diameter: {diameter} between nodes {diameter_nodes[0]} and {diameter_nodes[1]}")
        return diameter, diameter_nodes, diameter_trace
    
    def calculate_average_degree(self):
        """Calculate the average degree of nodes in the graph."""
        if not self.nodes:
            return 0
        
        total_degree = sum(len(self.graph[node]) for node in self.nodes)
        return total_degree / len(self.nodes)

def save_to_file(data, filename):
    """Save data to a file."""
    with open(filename, 'w') as f:
        f.write(data)
    print(f"Data saved to {filename}")

def main():
    graph = Graph()
    
    # Use the specified dataset file directly
    dataset_file = 'oregon1_010331.txt.gz'
    print(f"Loading dataset from: {dataset_file}")
    
    if not graph.load_graph_from_file(dataset_file):
        print("Failed to load the graph. Exiting...")
        return
    
    # Create results directory if it doesn't exist
    if not os.path.exists("results"):
        os.makedirs("results")
    
    # Ask user to input a source node
    print(f"Available nodes range from {min(graph.nodes)} to {max(graph.nodes)}")
    while True:
        try:
            source_node = int(input("Enter a source node for BFS and DFS traversals: "))
            if source_node in graph.nodes:
                break
            else:
                print(f"Node {source_node} not found in the graph. Please try again.")
        except ValueError:
            print("Please enter a valid integer.")
    
    print(f"Using source node: {source_node} for BFS and DFS traversals")
    
    # BFS
    start_time = time.time()
    bfs_result, bfs_trace = graph.bfs(source_node)
    bfs_time = time.time() - start_time
    
    # Save BFS result and execution time
    bfs_output = f"BFS Result: {bfs_result}\n\nBFS Execution Time: {bfs_time:.6f} seconds"
    save_to_file(bfs_output, "results/BFS_result.txt")
    
    # Save BFS trace separately
    bfs_trace_output = "\n".join(bfs_trace)
    save_to_file(bfs_trace_output, "results/BFS_trace.txt")
    
    # DFS
    start_time = time.time()
    dfs_result, dfs_trace = graph.dfs(source_node)
    dfs_time = time.time() - start_time
    
    # Save DFS result and execution time
    dfs_output = f"DFS Result: {dfs_result}\n\nDFS Execution Time: {dfs_time:.6f} seconds"
    save_to_file(dfs_output, "results/DFS_result.txt")
    
    # Save DFS trace separately
    dfs_trace_output = "\n".join(dfs_trace)
    save_to_file(dfs_trace_output, "results/DFS_trace.txt")
    
    # Cycle Detection
    start_time = time.time()
    has_cycle, cycle_trace = graph.detect_cycle()
    cycle_time = time.time() - start_time
    
    # Save cycle detection result and execution time
    cycle_output = f"Does the graph contain a cycle? {'Yes' if has_cycle else 'No'}\n\nCycle Detection Execution Time: {cycle_time:.6f} seconds"
    save_to_file(cycle_output, "results/Cycle_detection_result.txt")
    
    # Save cycle detection trace separately
    cycle_trace_output = "\n".join(cycle_trace)
    save_to_file(cycle_trace_output, "results/Cycle_trace.txt")
    
    # Calculate Diameter
    print("Calculating graph diameter...")
    start_time = time.time()
    diameter, diameter_nodes, diameter_trace = graph.calculate_diameter()
    diameter_time = time.time() - start_time
    
    # Save diameter result and execution time
    diameter_output = f"Graph Diameter: {diameter}\nDiameter Path: {diameter_nodes[0]} to {diameter_nodes[1]}\n\nDiameter Calculation Execution Time: {diameter_time:.6f} seconds"
    save_to_file(diameter_output, "results/Diameter_result.txt")
    
    # Save diameter trace separately
    diameter_trace_output = "\n".join(diameter_trace)
    save_to_file(diameter_trace_output, "results/Diameter_trace.txt")
    
    # Calculate Average Degree
    avg_degree = graph.calculate_average_degree()
    avg_degree_output = f"Average Degree: {avg_degree:.4f}"
    save_to_file(avg_degree_output, "results/Average_degree.txt")
    
    print("\nResults Summary:")
    print(f"BFS Time: {bfs_time:.6f} seconds")
    print(f"DFS Time: {dfs_time:.6f} seconds")
    print(f"Cycle Detection Time: {cycle_time:.6f} seconds")
    print(f"Diameter Calculation Time: {diameter_time:.6f} seconds")
    print(f"Cycle Detected: {'Yes' if has_cycle else 'No'}")
    print(f"Graph Diameter: {diameter} (between nodes {diameter_nodes[0]} and {diameter_nodes[1]})")
    print(f"Average Degree: {avg_degree:.4f}")
    
    print("\nFiles Generated:")
    print("1. results/BFS_result.txt - BFS traversal result and execution time")
    print("2. results/BFS_trace.txt - Complete trace of queue operations during BFS")
    print("3. results/DFS_result.txt - DFS traversal result and execution time")
    print("4. results/DFS_trace.txt - Complete trace of stack operations during DFS") 
    print("5. results/Cycle_detection_result.txt - Cycle detection result and execution time")
    print("6. results/Cycle_trace.txt - Complete trace of cycle detection algorithm")
    print("7. results/Diameter_result.txt - Graph diameter result and execution time")
    print("8. results/Diameter_trace.txt - Complete trace of diameter calculation algorithm")
    print("9. results/Average_degree.txt - Average degree of nodes in the graph")

if __name__ == "__main__":
    main()