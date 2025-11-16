# Graph Algorithms Visualizer & Analyzer 📊

**A comprehensive academic project** implementing and analyzing **six fundamental graph algorithms**:  
- **Single Source Shortest Path** (Dijkstra, Bellman-Ford)  
- **Minimum Spanning Tree** (Prim’s, Kruskal’s)  
- **Graph Traversal** (BFS, DFS)  

Includes **performance benchmarking**, **time complexity analysis**, **data structure impact evaluation**, and **visualization plots**.

> **Academic Project** — Submitted to: **Sir Irfan**  
> **Institution**: National University of Computer and Emerging Sciences (NUCES), Islamabad  
> **Date**: November 16, 2025

---

## Team Members

| Name             | Roll No.   |
|------------------|------------|
| **Azghan Ahmad** | 22i-2667   |
| Amna Asif        | 22i-8777   |
| Rabail           | 22i-1507   |

---

## How to Run (Each Algorithm Runs Independently)

Each algorithm is implemented in a **separate folder** and runs **independently**. Follow these steps:

```bash
# 1. Minimum Spanning Tree (Prim's & Kruskal's)
cd "Prims_Kruskal_222667_228777_221507"
python Prims_222667_228777_221507.py

# 2. Graph Traversal (BFS & DFS)
cd "../DFS_BFS_222667_228777_221507"
python DFS_222667_228777_221507.py

# 3. Shortest Path (Dijkstra & Bellman-Ford)
cd "../Dijkstra_Bellman_222667_228777_221507"
python Dijkstra_222667_228777_221507.py
```

> Note: All scripts use the same dataset: `oregon1_010331.txt` (included in each folder).

## Project Structure
```
textGraph_Algorithms_Project/
│
├── DFS_BFS_222667_228777_221507/
│   ├── DFS_222667_228777_221507.py
│   ├── oregon1_010331.txt
│   └── analysis_results/
│
├── Dijkstra_Bellman_222667_228777_221507/
│   ├── Dijkstra_222667_228777_221507.py
│   └── oregon1_010331.txt
│
├── Prims_Kruskal_222667_228777_221507/
│   ├── Prims_222667_228777_221507.py
│   ├── 222667_228777_221507.pdf        # Full Project Report
│   ├── 222667_228777_221507.docx
│   └── oregon1_010331.txt
│
├── graph_analysis_output/              # Generated plots
├── outputs/                            # Algorithm logs & results
│
├── .gitignore
└── README.md                           # This file
```

## Core Algorithms Implemented

| Category          | Algorithms           | Features                                     |
|------------------|-------------------|---------------------------------------------|
| Shortest Path     | Dijkstra, Bellman-Ford | Graph diameter, negative edge support     |
| MST               | Prim’s, Kruskal’s  | Union-Find, priority queue, degree distribution |
| Traversal         | BFS, DFS           | Cycle detection, diameter, traversal order  |

## Key Features & Analysis

| Feature                 | Description                                                         |
|-------------------------|---------------------------------------------------------------------|
| Time Complexity Analysis| Theoretical + empirical comparison                                   |
| Performance Plots       | Execution time vs. node count                                        |
| Data Structure Impact   | Queue (BFS), Stack (DFS), Priority Queue (Dijkstra/Prim)            |
| Graph Properties        | Diameter, cycle detection, degree distribution                       |
| Dataset                 | oregon1_010331.txt — Real-world AS-level Internet graph (10,670 nodes, 22,002 edges) |

## Setup & Requirements

1. Clone the Repository
```bash
git clone https://github.com/AzghanAhmad/graph-algorithms-visualizer.git
cd graph-algorithms-visualizer
```

2. Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows
```

3. Install Dependencies
```bash
pip install networkx matplotlib numpy pandas
```
Or use `requirements.txt`:
```bash
pip install -r requirements.txt
```

## Output Examples

**Sample Output (Dijkstra)**
```
Shortest path from node 0:
  To node 100: distance = 3
  To node 500: distance = 5
Graph Diameter: 9
```

**Sample Plot**
```
graph_analysis_output/execution_time_vs_nodes.png
graph_analysis_output/degree_distribution.png
```

## Report Summary (From PDF)

**Machine Specs:** Intel i5, 8GB RAM, Python 3.9  
**Time Complexities:**
- Dijkstra: O((V+E) log V)
- Bellman-Ford: O(VE)
- Prim’s: O(E log V)
- Kruskal’s: O(E log E)
- BFS/DFS: O(V+E)

**Key Findings:**
- Prim’s faster than Kruskal’s on dense graphs
- Bellman-Ford handles negative weights but slower
- BFS better for diameter than DFS

## Contributing

- Fork the repo
- Create your branch: `git checkout -b feature/new-algo`
- Commit: `git commit -m 'Add Floyd-Warshall'`
- Push & Open a PR

## License

MIT License – Free to use, modify, and distribute.  
See LICENSE for details.

## Author

Azghan Ahmad  
GitHub: @AzghanAhmad  
Email: azghan.ahmad@gmail.com (optional)  
NUCES FAST, Islamabad  
November 16, 2025

Design and Analysis of Algorithms – Final Project  
Not for production use. Educational implementation only.

