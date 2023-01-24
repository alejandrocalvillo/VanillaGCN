import networkx as nx
import random
import os
import yaml
from getpass import getpass

# Define destination for the generated samples
training_dataset_path = "Newtraining"
#paths relative to data folder
graphs_path = "graphs"
routings_path = "routings"
tm_path = "tm"
# Path to simulator file
simulation_file = os.path.join(training_dataset_path,"simulation.txt")
# Name of the dataset: Allows you to store several datasets in the same path
dataset_name = "dataset1"

if (os.path.isdir(training_dataset_path)):
    print ("Destination path already exists. Files within the directory may be overwritten.")
else:
    os.makedirs(os.path.join(training_dataset_path,graphs_path))
    os.mkdir(os.path.join(training_dataset_path,routings_path))
    os.mkdir(os.path.join(training_dataset_path,tm_path))

'''
Generate a graph topology file. The graphs generated have the following characteristics:
- All nodes have buffer sizes of 32000 bits and FIFO scheduling
- All links have bandwidths of 100000 bits per second
'''
def generate_topology(net_size, graph_file):
    G = nx.Graph()
    nodes = []
    node_degree = []
    for i in range(net_size):
        node_degree.append(random.choices([2,3,4,5,6],weights=[0.34,0.35,0.2,0.1,0.01])[0])
        
        nodes.append(i)
        G.add_node(i)
        # Assign to each node the scheduling Policy
        G.nodes[i]["schedulingPolicy"] = "FIFO"
        # Assign the buffer size of all the ports of the node
        G.nodes[i]["bufferSizes"] = 32000

    finish = False
    while (True):
        aux_nodes = list(nodes)
        n0 = random.choice(aux_nodes)
        aux_nodes.remove(n0)
        # Remove adjacents nodes (only one link between two nodes)
        for n1 in G[n0]:
            if (n1 in aux_nodes):
                aux_nodes.remove(n1)
        if (len(aux_nodes) == 0):
            # No more links can be added to this node - can not acomplish node_degree for this node
            nodes.remove(n0)
            if (len(nodes) == 1):
                break
            continue
        n1 = random.choice(aux_nodes)
        G.add_edge(n0, n1)
        # Assign the link capacity to the link
        G[n0][n1]["bandwidth"] = 100000
        
        for n in [n0,n1]:
            node_degree[n] -= 1
            if (node_degree[n] == 0):
                nodes.remove(n)
                if (len(nodes) == 1):
                    finish = True
                    break
        if (finish):
            break
    if (not nx.is_connected(G)):
        G = generate_topology(net_size, graph_file)
        return G
    
    nx.write_gml(G,graph_file)
    
    return (G)

'''
Generate a file with the shortest path routing of the topology G
'''
def generate_routing(G, routing_file):
    with open(routing_file,"w") as r_fd:
        lPaths = nx.shortest_path(G)
        for src in G:
            for dst in G:
                if (src == dst):
                    continue
                path =  ','.join(str(x) for x in lPaths[src][dst])
                r_fd.write(path+"\n")

'''
Generate a traffic matrix file. We consider flows between all nodes in the newtork, each with the following characterstics
- The average bandwidth ranges between 10 and max_avg_lbda
- We consider three time distributions (in case of the ON-OFF policy we have off periods of 10 and on periods of 5)
- We consider two packages distributions, chosen at random
- ToS is assigned randomly
'''
def generate_tm(G,max_avg_lbda, traffic_file):
    poisson = "0" 
    cbr = "1"
    on_off = "2,10,5" #time_distribution, avg off_time exp, avg on_time exp
    time_dist = [poisson,cbr,on_off]
    
    pkt_dist_1 = "0,300,0.5,1700,0.5" #genric pkt size dist, pkt_size 1, prob 1, pkt_size 2, prob 2
    pkt_dist_2 = "0,500,0.6,1000,0.2,1400,0.2" #genric pkt size dist, pkt_size 1, prob 1, 
                                               # pkt_size 2, prob 2, pkt_size 3, prob 3
    pkt_size_dist = [pkt_dist_1, pkt_dist_2]
    tos_lst = [0,1,2]
    
    with open(traffic_file,"w") as tm_fd:
        for src in G:
            for dst in G:
                avg_bw = random.randint(10,max_avg_lbda)
                td = random.choice(time_dist)
                sd = random.choice(pkt_size_dist)
                tos = random.choice(tos_lst)
                
                traffic_line = "{},{},{},{},{},{}".format(
                    src,dst,avg_bw,td,sd,tos)
                tm_fd.write(traffic_line+"\n")


"""
We generate the files using the previously defined functions. This code will produce 100 samples where:
- We generate 5 topologies, and then we generate 20 traffic matrices for each
- The topology sizes range from 6 to 10 nodes. Lo cambio a 9 
- We consider the maximum average bandwidth per flow as 1000
"""
max_avg_lbda = 1000
with open (simulation_file,"w") as fd:
    for net_size in range (9): #Antes [6,11] ahora 9
        #Generate graph
        graph_file = os.path.join(graphs_path,"graph_{}.txt".format(net_size))
        G = generate_topology(net_size, os.path.join(training_dataset_path,graph_file))
        # Generate routing
        routing_file = os.path.join(routings_path,"routing_{}.txt".format(net_size))
        generate_routing(G, os.path.join(training_dataset_path,routing_file))
        # Generate TM:
        for i in range (20):
            tm_file = os.path.join(tm_path,"tm_{}_{}.txt".format(net_size,i))
            generate_tm(G,max_avg_lbda, os.path.join(training_dataset_path,tm_file))
            sim_line = "{},{},{}\n".format(graph_file,routing_file,tm_file)   
            # If dataset has been generated in windows, convert paths into linux format
            fd.write(sim_line.replace("\\","/"))


# First we generate the configuration file
conf_file = os.path.join(training_dataset_path,"conf.yml")
conf_parameters = {
    "threads": 6,# Number of threads to use 
    "dataset_name": dataset_name, # Name of the dataset. It is created in <training_dataset_path>/results/<name>
    "samples_per_file": 10, # Number of samples per compressed file
    "rm_prev_results": "n", # If 'y' is selected and the results folder already exists, the folder is removed.
}

with open(conf_file, 'w') as fd:
    yaml.dump(conf_parameters, fd)


def docker_cmd(training_dataset_path):
    raw_cmd = f"docker run --rm --mount type=bind,src={os.path.join(os.getcwd(),training_dataset_path)},dst=/data bnnupc/netsim:v0.1"
    terminal_cmd = raw_cmd
    if os.name != 'nt': # Unix, requires sudo
        print("Superuser privileges are required to run docker. Introduce sudo password when prompted")
        terminal_cmd = f"echo {getpass()} | sudo -S " + raw_cmd
        raw_cmd = "sudo " + raw_cmd
    return raw_cmd, terminal_cmd


# Start the docker
raw_cmd, terminal_cmd = docker_cmd(training_dataset_path)
print("The next cell will launch docker from the notebook. Alternatively, run the following command from a terminal:")
print(raw_cmd)