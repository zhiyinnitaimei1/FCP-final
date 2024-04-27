import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
matplotlib.use('TkAgg')

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

	def get_mean_clustering(self):
		#Your code for task 3 goes here

	def get_mean_path_length(self):
		#Your code for task 3 goes here

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

	def make_ring_network(self, N, neighbour_range=1):
		#Your code  for task 4 goes here

	def make_small_world_network(self, N, re_wire_prob=0.2):
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
	assert(network.get_clustering()==0), network.get_clustering()
	assert(network.get_path_length()==2.777777777777778), network.get_path_length()

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
	assert(network.get_clustering()==0),  network.get_clustering()
	assert(network.get_path_length()==5), network.get_path_length()

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
	assert(network.get_clustering()==1),  network.get_clustering()
	assert(network.get_path_length()==1), network.get_path_length()

	print("All tests passed")

'''
==============================================================================================================
This section contains code for the Ising Model - task 1 in the assignment
==============================================================================================================
'''
def calculate_agreement(population, row, col, external=0.0, alpha=1.0):
    #  Calculate the agreement at a given position.
    """
    population: the current state of the Ising model.
    row: the row index of the position.
    col: the column index of the position.
    external: the magnitude of any external "pull" on opinion.
    alpha: system's tolerance for disagreement.
    """

    n_rows, n_cols = population.shape
    sum_neighbors = 0

    # Neighbors' coordinates (Up, Right, Down, Left)
    neighbors = [(row - 1, col), (row, col + 1), (row + 1, col), (row, col - 1)]
    for x, y in neighbors:
        if 0 <= x < n_rows and 0 <= y < n_cols:
            sum_neighbors += population[x, y]
    # Agreement considers the external influence
    agreement = sum_neighbors * population[row, col] + external * population[row, col]

    return agreement
    # The agreement value at the given position


def ising_step(population, alpha=1.0, external=0.0):
    #  Single update of the Ising model.

    """
    This function will perform a single update of the Ising model.
    Inputs: population (numpy array)
            alpha (float) - system's tolerance for disagreement
            external (float) - the magnitude of any external "pull" on opinion
    """
    n_rows, n_cols = population.shape
    row = np.random.randint(0, n_rows)
    col = np.random.randint(0, n_cols)

    agreement = calculate_agreement(population, row, col, alpha, external)

    prob_flip = np.exp(-agreement) / alpha if agreement > 0 else 1

    if np.random.random() < prob_flip or agreement < 0:
        population[row, col] *= -1

    return population


def plot_ising(im, population):
    # Plot the Ising model.

    """
    im: matplotlib image object.
    population: the current state of the Ising model.
    """
    new_im = np.array([[255 if val == -1 else 1 for val in rows] for rows in population], dtype=np.int8)
    im.set_data(new_im)
    plt.pause(0.1)


def test_ising():
    # Test calculations.

    print("Testing ising model calculations")
    population = -np.ones((3, 3))
    assert(calculate_agreement(population,1,1)==4), "Test 1"

    population[1, 1] = 1.
    assert(calculate_agreement(population,1,1)==-4), "Test 2"

    population[0, 1] = 1.
    assert(calculate_agreement(population,1,1)==-2), "Test 3"

    population[1, 0] = 1.
    assert(calculate_agreement(population,1,1)==0), "Test 4"

    population[2, 1] = 1.
    assert(calculate_agreement(population,1,1)==2), "Test 5"

    population[1, 2] = 1.
    assert(calculate_agreement(population,1,1)==4), "Test 6"

    "Testing external pull"
    population = -np.ones((3, 3))
    assert(calculate_agreement(population,1,1,1)==3), "Test 7"
    assert(calculate_agreement(population,1,1,-1)==5), "Test 8"
    assert(calculate_agreement(population,1,1,10)==-6), "Test 9"
    assert(calculate_agreement(population,1,1, -10)==14), "Test 10"

    print("Tests passed")


def ising_main(population, alpha=None, external=0.0):
    # Main function for the Ising model.

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    im = ax.imshow(population, interpolation='none', cmap='RdPu_r')

    # Iterating an update 100 times
    for frame in range(100):
        # Iterating single steps 1000 times to form an update
        for step in range(1000):
            ising_step(population, external)
        print('Step:', frame, end='\r')
        plot_ising(im, population)



'''
==============================================================================================================
This section contains code for the Defuant Model - task 2 in the assignment
==============================================================================================================
'''
def defuant_model(opinions, threshold, beta,iterations):
    # Implementation of the Defuant model.
    """
    opinions: initial opinions.
    threshold: threshold for interaction.
    beta: updating rate.
    iterations: number of iterations.
    """
    opinions_over_time = [[] for i in range(len(opinions))]
    for t in range(iterations):
        i = np.random.randint(len(opinions))
        j = (i + 1) % len(opinions) if np.random.rand() > 0.5 else (i - 1) % len(opinions)

        if abs(opinions[i] - opinions[j]) < threshold:
            opinions[i] += beta * (opinions[j] - opinions[i])
            opinions[j] += beta * (opinions[i] - opinions[j])
        for i in range(len(opinions)):
            opinions_over_time[i].append(opinions[i])
    return opinions_over_time
    # list of lists, opinions over time

def run_defuant(beta, threshold, population_size, iterations, testing=False):
    """
    beta: updating rate.
    threshold: threshold for interaction.
    population_size: number of agents.
    iterations: number of iterations.
    testing: flag to plot opinions over time if True.
    """
    initial_population = np.random.rand(population_size)
    opinions_over_time = defuant_model(initial_population, threshold, beta,iterations)
    # Plot the final population distribution
    plt.figure(figsize=(14, 7))
    plt.subplot(1, 2, 1)
    opinions_transposed = list(zip(*opinions_over_time))
    plt.hist(opinions_transposed[-1], bins=np.linspace(0, 1, 20), color='blue', edgecolor='black')
    plt.title('Final Population Distribution')
    plt.xlabel('Opinion')
    plt.ylabel('Frequency')

    # If testing flag is set, plot the opinions over time
    # if testing:
    plt.subplot(1, 2, 2)
    plt.title('Opinions Over Time')
    plt.xlabel('Timestep')
    plt.ylabel('Opinion')
    for opinions in opinions_over_time:
        plt.plot(opinions, 'o', markersize=3, label=f'Person {opinions}')
    plt.tight_layout()
    plt.show()



'''
==============================================================================================================
This section contains code for the main function- you should write some code for handling flags here
==============================================================================================================
'''

def main():
    #You should write some code for handling flags here
    global testt
	
    #task 1:
    H = 0.0
    alpha = 1.0
    grid_size = 10
    if "-ising_model" in sys.argv:
        if "-external" in sys.argv:
            external_index = sys.argv.index("-external") + 1
            H = float(sys.argv[external_index])



        if "-alpha" in sys.argv:
            alpha_index = sys.argv.index("-alpha") + 1
            alpha = float(sys.argv[alpha_index])
        population = np.random.choice([-1, 1], size=(grid_size, grid_size))
        ising_main(population, alpha, H)
    elif "-test_ising" in sys.argv:
        test_ising()
    #task 2
    beta = 0.2
    threshold = 0.2
    testing = False
    testt=0
    if "-defuant" in sys.argv:
        testt = 1
        if "-beta" in sys.argv:
            beta_index = sys.argv.index("-beta") + 1
            beta = float(sys.argv[beta_index])
        if "-threshold" in sys.argv:
            threshold_index = sys.argv.index("-threshold") + 1
            threshold = float(sys.argv[threshold_index])

    elif "-test_defuant" in sys.argv:
        testing = 1
    if testing ==True or testt ==True:
        run_defuant(beta=beta, threshold=threshold, population_size=100, iterations=10000, testing=testing)
	    

if __name__=="__main__":
	main()
