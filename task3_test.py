import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class Node:

	def __init__(self, value, number, connections=None):

		self.index = number
		self.connections = connections
		self.value = value

class Network: 

	def __init__(self, nodes=None):

		if nodes is None:
			self.nodes = []
		else:
			self.nodes = nodes 

	def get_mean_degree(self):
		#Your code  for task 3 goes here
		if not self.nodes:
			return 0
			  
		degrees = [sum(1 for conn in node.connections if conn == 1) for node in self.nodes]
		mean_degree = np.mean(degrees)
		return mean_degree

	def get_mean_path_length(self):
		num_nodes = len(self.nodes)
		path_lengths = []
    
		for start in range(num_nodes):
			visited = [False] * num_nodes
			distances = [0] * num_nodes
			queue = [start]
        
			visited[start] = True
        
			while queue:
				current = queue.pop(0)
				for neighbour_index, connected in enumerate(self.nodes[current].connections):
					if connected and not visited[neighbour_index]:
						visited[neighbour_index] = True
						distances[neighbour_index] = distances[current] + 1
						queue.append(neighbour_index)
			path_lengths.extend([dist for dist in distances if dist > 0])
		if not path_lengths:
			return float('inf')
		mean_path_length = sum(path_lengths) / (num_nodes * (num_nodes - 1))
		return round(mean_path_length, 15)


	def get_mean_clustering_coefficient(self):
    # 遍历每个节点计算其聚类系数
		clustering_coeffs = []
		for node in self.nodes:
			neighbors = [i for i, connected in enumerate(node.connections) if connected]
			if len(neighbors) < 2:
				# 少于两个邻居，无法形成三角形
				clustering_coeffs.append(0)
				continue

			# 计算邻居之间可能的连接数（即可以形成的三角形数量）
			possible_triangles = len(neighbors) * (len(neighbors) - 1) / 2
			actual_triangles = 0
			# 遍历邻居之间的连接
			for i in range(len(neighbors)):
				for j in range(i + 1, len(neighbors)):
					if self.nodes[neighbors[i]].connections[neighbors[j]]:
						actual_triangles += 1

			# 计算并保存节点的聚类系数
			clustering_coeffs.append(actual_triangles / possible_triangles)

		# 计算所有节点聚类系数的平均值
		mean_clustering_coefficient = np.mean(clustering_coeffs)
		return mean_clustering_coefficient

	def make_random_network(self, N, connection_probability=0.5):
		'''
		This function makes a *random* network of size N.
		Each node is connected to each other node with probability p
		'''

		self.nodes = []
		for node_number in range(N):
			value = np.random.random()
			connections = [0 for _ in range(N)]
			self.nodes.append(Node(value, node_number, connections))

		for (index, node) in enumerate(self.nodes):
			for neighbour_index in range(index+1, N):
				if np.random.random() < connection_probability:
					node.connections[neighbour_index] = 1
					self.nodes[neighbour_index].connections[index] = 1

	#def make_ring_network(self, N, neighbour_range=1):
		#Your code  for task 4 goes here

	#def make_small_world_network(self, N, re_wire_prob=0.2):
		#Your code for task 4 goes here

	def plot(self):

		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.set_axis_off()

		num_nodes = len(self.nodes)
		network_radius = num_nodes * 10
		ax.set_xlim([-1.1*network_radius, 1.1*network_radius])
		ax.set_ylim([-1.1*network_radius, 1.1*network_radius])

		for (i, node) in enumerate(self.nodes):
			node_angle = i * 2 * np.pi / num_nodes
			node_x = network_radius * np.cos(node_angle)
			node_y = network_radius * np.sin(node_angle)

			circle = plt.Circle((node_x, node_y), 0.3*num_nodes, color=cm.hot(node.value))
			ax.add_patch(circle)

			for neighbour_index in range(i+1, num_nodes):
				if node.connections[neighbour_index]:
					neighbour_angle = neighbour_index * 2 * np.pi / num_nodes
					neighbour_x = network_radius * np.cos(neighbour_angle)
					neighbour_y = network_radius * np.sin(neighbour_angle)

					ax.plot((node_x, neighbour_x), (node_y, neighbour_y), color='black')

def test_networks():

	#Ring network
	nodes = []
	num_nodes = 10
	for node_number in range(num_nodes):
		connections = [0 for val in range(num_nodes)]
		connections[(node_number-1)%num_nodes] = 1
		connections[(node_number+1)%num_nodes] = 1
		new_node = Node(0, node_number, connections=connections)
		nodes.append(new_node)
	network = Network(nodes)

	print("Testing ring network")
	assert(network.get_mean_degree()==2), network.get_mean_degree()
	assert(network.get_mean_clustering_coefficient()==0), network.get_mean_clustering_coefficient()
	assert(network.get_mean_path_length()==2.777777777777778), network.get_mean_path_length()

	nodes = []
	num_nodes = 10
	for node_number in range(num_nodes):
		connections = [0 for val in range(num_nodes)]
		connections[(node_number+1)%num_nodes] = 1
		new_node = Node(0, node_number, connections=connections)
		nodes.append(new_node)
	network = Network(nodes)

	print("Testing one-sided network")
	assert(network.get_mean_degree()==1), network.get_mean_degree()
	assert(network.get_mean_clustering_coefficient()==0),  network.get_mean_clustering_coefficient()
	assert(network.get_mean_path_length()==5), network.get_mean_path_length()

	nodes = []
	num_nodes = 10
	for node_number in range(num_nodes):
		connections = [1 for val in range(num_nodes)]
		connections[node_number] = 0
		new_node = Node(0, node_number, connections=connections)
		nodes.append(new_node)
	network = Network(nodes)

	print("Testing fully connected network")
	assert(network.get_mean_degree()==num_nodes-1), network.get_mean_degree()
	assert(network.get_mean_clustering_coefficient()==1),  network.get_mean_clustering_coefficient()
	assert(network.get_mean_path_length()==1), network.get_mean_path_length()

	print("All tests passed")
test_networks()
