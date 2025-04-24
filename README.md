# Major-Project
Supply Chain Disruption - A Maximum flow network Interdiction Approach

Supply Chain Disruption: A Maximum Flow Network Interdiction Approach
---------------------------------------------------------------------

Overview
--------
This project implements a network analysis tool that models supply chain networks and simulates disruptions using maximum flow network interdiction. The tool helps identify critical paths and vulnerabilities in supply chain networks, allowing users to analyze the impact of potential disruptions.

Features
--------
Load supply chain network data from CSV files
Validate input data for consistency and completeness
Calculate maximum flow through the network using Ford-Fulkerson algorithm
Perform bilevel optimization to identify optimal interdiction strategies
Visualize network flows before and after interdiction
Quantify the impact of disruptions on supply chain capacity

Requirements
------------
Python 3.6+
Pandas
NetworkX
Gurobi Optimizer
Matplotlib
NumPy (dependency of other packages)

Installation
------------
1. Install Python dependencies
pip install pandas networkx matplotlib
2. Install Gurobi Optimizer
The project requires Gurobi, which needs a separate installation:

Download Gurobi from https://www.gurobi.com/downloads/
Follow the installation instructions for your operating system
Get and activate a Gurobi license (academic licenses are available for free)
Install the Python interface:

pip install gurobipy
Usage
Input Data Format
Prepare your network data in a CSV file with the following columns:

Source: Source node S1
Target: Target node T1
Capacity: Edge capacity (must be non-negative)
Budget: Cost to interdict the edge (must be non-negative)

Example CSV format:
Source,Target,Capacity,Budget
Supplier1,Warehouse1,100,500
Warehouse1,Distributor1,80,400
...
Running the Program
python supply_chain_disruption.py

The program will:
-----------------
Prompt for the path to your CSV file
Ask for the number of edges to interdict
Calculate and display the maximum flow before interdiction
Visualize the network before interdiction
Perform optimization to find optimal interdiction strategy
Calculate and display the maximum flow after interdiction
Visualize the network after interdiction
Display detailed results and statistics

Example Output:
---------------
The program will display:

Maximum flow values before and after interdiction
Number of edges and nodes in the flow paths
Percentage reduction in flow capacity
List of interdicted edges
Total budget used for interdiction
Computation time
