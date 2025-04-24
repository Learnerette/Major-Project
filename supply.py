import pandas as pd
import networkx as nx
import gurobipy as gp
from gurobipy import GRB
import time
import matplotlib.pyplot as plt
import sys
import os

def validate_data(df):
    """Validate the input DataFrame has required columns and valid data"""
    required_columns = ['Source', 'Target', 'Capacity', 'Budget']
    
    # Check for required columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check for numeric values in Capacity and Budget
    if not pd.api.types.is_numeric_dtype(df['Capacity']):
        raise ValueError("Capacity must be numeric")
    if not pd.api.types.is_numeric_dtype(df['Budget']):
        raise ValueError("Budget must be numeric")
    
    # Check for negative values
    if (df['Capacity'] < 0).any():
        raise ValueError("Capacity cannot be negative")
    if (df['Budget'] < 0).any():
        raise ValueError("Budget cannot be negative")

def load_and_validate_data(filepath):
    """Load and validate the CSV file"""
    try:
        # Check if file exists
        if not os.path.exists(filepath):
            # Try adding .csv extension if not present
            if not filepath.lower().endswith('.csv'):
                filepath += '.csv'
                if not os.path.exists(filepath):
                    raise FileNotFoundError(f"File not found: {filepath}")
        
        df = pd.read_csv(filepath)
        validate_data(df)
        return df
    except Exception as e:
        print(f"Error loading or validating data: {str(e)}")
        sys.exit(1)

def create_graph(df):
    """Create a directed graph from the DataFrame"""
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(row['Source'], row['Target'], 
                  capacity=row['Capacity'], 
                  budget=row['Budget'])
    return G

def max_flow_ford_fulkerson(graph, source, target):
    """Calculate maximum flow using Ford-Fulkerson"""
    return nx.maximum_flow_value(graph, source, target)

def get_flow_paths(graph, source, target):
    """Get all flow paths in the graph"""
    flow_value, flow_dict = nx.maximum_flow(graph, source, target)
    flow_paths = []
    for u in flow_dict:
        for v in flow_dict[u]:
            if flow_dict[u][v] > 0:
                flow_paths.append((u, v))
    return flow_paths

def get_flow_components(graph, flow_paths, source, target):
    """Get nodes and edges involved in the flow"""
    flow_edges = set()
    flow_nodes = set()
    
    for u, v in flow_paths:
        flow_edges.add((u, v))
        flow_nodes.add(u)
        flow_nodes.add(v)
    
    # Filter to only include edges that are part of the actual flow paths
    # (not just all edges with positive flow)
    filtered_edges = set()
    filtered_nodes = {source, target}
    
    # Perform BFS from source to find actual flow paths
    queue = [(source, [source])]
    while queue:
        node, path = queue.pop(0)
        for neighbor in graph.successors(node):
            if (node, neighbor) in flow_edges and neighbor not in path:
                if neighbor == target:
                    # Found a complete path
                    for i in range(len(path)):
                        filtered_nodes.add(path[i])
                        if i < len(path) - 1:
                            filtered_edges.add((path[i], path[i+1]))
                    filtered_edges.add((path[-1], target))
                    filtered_nodes.add(target)
                else:
                    queue.append((neighbor, path + [neighbor]))
    
    return filtered_edges, filtered_nodes

def create_layered_layout(graph, source, target, num_layers=5):
    """Create a layered layout for visualization"""
    pos = {}
    pos[source] = (0, 0.5)
    pos[target] = (num_layers + 1, 0.5)
    
    intermediate_nodes = [node for node in graph.nodes() if node != source and node != target]
    
    # Distribute nodes evenly across layers
    for i, node in enumerate(intermediate_nodes):
        layer = (i % num_layers) + 1  # Distribute across layers 1 to num_layers
        x_pos = layer
        y_pos = (i + 1) / (len(intermediate_nodes) + 1)  # Normalized vertical position
        pos[node] = (x_pos, y_pos)
    
    return pos

def visualize_graph(graph, flow_paths, title, source, target, num_layers=5):
    """Visualize the graph with highlighted flow paths"""
    try:
        pos = create_layered_layout(graph, source, target, num_layers)
        
        plt.figure(figsize=(20, 15))
        
        # Draw all nodes with different colors for source and target
        node_colors = []
        node_sizes = []
        for node in graph.nodes():
            if node == source:
                node_colors.append('green')
                node_sizes.append(800)
            elif node == target:
                node_colors.append('red')
                node_sizes.append(800)
            else:
                node_colors.append('skyblue')
                node_sizes.append(300)
        
        # Draw all edges first in light gray
        nx.draw_networkx_edges(graph, pos, edge_color='lightgray', 
                             width=1, arrowstyle='->', arrowsize=10, alpha=0.5)
        
        # Highlight flow paths if they exist
        if flow_paths:
            nx.draw_networkx_edges(graph, pos, edgelist=flow_paths,
                                 edge_color='red', width=2,
                                 arrowstyle='->', arrowsize=15)
        
        # Draw nodes
        nx.draw_networkx_nodes(graph, pos, node_size=node_sizes, 
                             node_color=node_colors, alpha=0.9)
        
        # Draw labels for important nodes only
        important_nodes = {source: source, target: target}
        nx.draw_networkx_labels(graph, pos, labels=important_nodes, 
                               font_size=12, font_weight='bold')
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Visualization warning: {str(e)}")
        print("Using fallback spring layout...")
        
        plt.figure(figsize=(20, 15))
        pos = nx.spring_layout(graph, k=0.15, iterations=50)
        
        # Draw with simpler styling for large graphs
        nx.draw_networkx(graph, pos, with_labels=False, node_size=50,
                       node_color='lightblue', edge_color='gray',
                       arrowsize=8, width=0.5)
        
        # Highlight source and target
        nx.draw_networkx_nodes(graph, pos, nodelist=[source], 
                             node_color='green', node_size=300)
        nx.draw_networkx_nodes(graph, pos, nodelist=[target], 
                             node_color='red', node_size=300)
        
        plt.title(title + " (Fallback Layout)", fontsize=16)
        plt.axis('off')
        plt.show()

def bilevel_optimization(graph, source, target, num_interdictions):
    """Perform bilevel optimization using Gurobi"""
    try:
        model = gp.Model("BilevelOptimization")
        interdiction_vars = {}
        
        for u, v, data in graph.edges(data=True):
            interdiction_vars[(u, v)] = model.addVar(vtype=GRB.BINARY, name=f"interdict_{u}_{v}")
        
        model.addConstr(
            gp.quicksum(interdiction_vars[(u, v)] for u, v in graph.edges()) <= num_interdictions,
            name="interdiction_constraint"
        )
        
        # Objective: Minimize the residual capacity after interdiction
        model.setObjective(
            gp.quicksum(data['capacity'] * (1 - interdiction_vars[(u, v)]) 
                       for u, v, data in graph.edges(data=True)),
            GRB.MINIMIZE
        )
        
        model.optimize()
        
        if model.status != GRB.OPTIMAL:
            raise RuntimeError("Optimization failed to find an optimal solution")
        
        interdicted_edges = [(u, v) for (u, v), var in interdiction_vars.items() if var.X > 0.5]
        total_budget = sum(data['budget'] for u, v, data in graph.edges(data=True) 
                          if (u, v) in interdicted_edges)
        
        interdicted_graph = graph.copy()
        interdicted_graph.remove_edges_from(interdicted_edges)
        
        max_flow_after = max_flow_ford_fulkerson(interdicted_graph, source, target)
        
        return interdicted_edges, total_budget, max_flow_after, interdicted_graph
    
    except gp.GurobiError as e:
        print(f"Gurobi error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"Error in optimization: {str(e)}")
        sys.exit(1)

def main():
    """Main function to run the analysis"""
    try:
        # Get file path from user
        while True:
            filepath = input("Enter the path to your CSV file: ").strip()
            if filepath:
                break
            print("Please enter a valid file path.")
        
        # Load and validate data
        df = load_and_validate_data(filepath)
        
        # Create graph
        G = create_graph(df)
        
        # Get source and target nodes
        sources = df['Source'].unique()
        targets = df['Target'].unique()
        source = sources[0]
        target = targets[-1]
        
        print(f"\nDetected source node: {source}")
        print(f"Detected target node: {target}")
        print(f"Total nodes: {len(G.nodes())}")
        print(f"Total edges: {len(G.edges())}\n")
        
        # Get number of interdictions from user
        while True:
            try:
                num_interdictions = int(input("Enter the number of edges to interdict (n): "))
                if 0 <= num_interdictions <= len(G.edges()):
                    break
                print(f"Please enter a number between 0 and {len(G.edges())}")
            except ValueError:
                print("Please enter a valid integer.")
        
        # Measure CPU time
        start_time = time.time()
        
        # Before interdiction analysis
        print("\nCalculating maximum flow before interdiction...")
        max_flow_before = max_flow_ford_fulkerson(G, source, target)
        flow_paths_before = get_flow_paths(G, source, target)
        flow_edges_before, flow_nodes_before = get_flow_components(G, flow_paths_before, source, target)
        num_edges_before = len(flow_edges_before)
        num_nodes_before = len(flow_nodes_before)
        
        # Visualization before interdiction
        print("Generating visualization before interdiction...")
        visualize_graph(G, flow_edges_before, 
                      f"Maximum Flow Paths Before Interdiction\nSource: {source}, Target: {target}", 
                      source, target)
        
        # Run optimization
        print("\nRunning bilevel optimization...")
        interdicted_edges, total_budget, max_flow_after, interdicted_graph = bilevel_optimization(
            G, source, target, num_interdictions)
        
        # After interdiction analysis
        print("Calculating maximum flow after interdiction...")
        flow_paths_after = get_flow_paths(interdicted_graph, source, target)
        flow_edges_after, flow_nodes_after = get_flow_components(interdicted_graph, flow_paths_after, source, target)
        num_edges_after = len(flow_edges_after)
        num_nodes_after = len(flow_nodes_after)
        
        # Visualization after interdiction
        print("Generating visualization after interdiction...")
        visualize_graph(interdicted_graph, flow_edges_after, 
                      f"Maximum Flow Paths After Interdiction\nSource: {source}, Target: {target}", 
                      source, target)
        
        # Calculate CPU time
        cpu_time = time.time() - start_time
        
        # Output results
        print("\n=== Results ===")
        print(f"Maximum Flow Before Interdiction: {max_flow_before}")
        print(f"Number of Edges in Maximum Flow Before Interdiction: {num_edges_before}")
        print(f"Number of Nodes Traversed in Maximum Flow Before Interdiction: {num_nodes_before}")
        print(f"\nMaximum Flow After Interdiction: {max_flow_after}")
        print(f"Number of Edges in Maximum Flow After Interdiction: {num_edges_after}")
        print(f"Number of Nodes Traversed in Maximum Flow After Interdiction: {num_nodes_after}")
        print(f"\nReduction in Flow: {max_flow_before - max_flow_after} ({((max_flow_before - max_flow_after)/max_flow_before*100):.2f}%)")
        print(f"\nInterdicted Edges ({len(interdicted_edges)}):")
        for edge in interdicted_edges[:10]:  # Show first 10 edges to avoid too much output
            print(f"  {edge[0]} -> {edge[1]}")
        if len(interdicted_edges) > 10:
            print(f"  ... and {len(interdicted_edges)-10} more edges")
        print(f"\nTotal Budget Used for Interdiction: {total_budget}")
        print(f"\nCPU Time: {cpu_time:.4f} seconds")
        
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        sys.exit(1)
if __name__ == "__main__":
    main()

