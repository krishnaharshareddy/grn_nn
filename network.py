import csv
import numpy as np
import copy
def random_boolean_function(values, node_rands, tree_rands, and_probability = 0.5, activator_probability=0.5):
    val = values[0] if node_rands[0] < activator_probability else not values[0]
    for i,r in enumerate(tree_rands):
        if r<and_probability:
            val = val and (values[i+1] if node_rands[i+1] < activator_probability else not values[i+1])
        else:
            val = val or (values[i+1] if node_rands[i+1] < activator_probability else not values[i+1])
    return val

class GNWNetwork():
    def __init__(self, fname):
        self.and_probability = 0.5
        self.activator_probability = 0.5
        self.self_loop_probability = 0.5
        self.read_gnw(fname)
        self.reset_random_f(0)
    def read_gnw(self, fname):
        self.count = 0
        self.names_dict = {}
        self.in_edge_list = []
        self.out_edge_list = []
        self.tfs = []
        self.tf_in_edge_list = []
        self.tf_idx_to_tf = {}
        
        with open(fname,'rb') as f:
            reader = csv.reader(f, delimiter = '\t')
            for line in reader:
                if line[0] not in self.names_dict:
                    self.names_dict[line[0]] = self.count
                    self.in_edge_list.append([])
                    self.out_edge_list.append([])
                    self.count+=1
                if line[1] not in self.names_dict:
                    self.names_dict[line[1]] = self.count
                    self.in_edge_list.append([])
                    self.out_edge_list.append([])
                    self.count+=1
                self.out_edge_list[self.names_dict[line[0]]].append(self.names_dict[line[1]])
                self.in_edge_list[self.names_dict[line[1]]].append(self.names_dict[line[0]])
        min_in = 1000
        input_count = 0
        for i in range(self.count):
            if len(self.out_edge_list[i]) > 0:
                self.tfs.append(i)
                self.tf_idx_to_tf[i] = len(self.tfs)-1
                if len(self.in_edge_list[i]) == 0:
                    input_count+=1
                if len(self.in_edge_list[i]) < min_in:
                    self.lowest_in_node = i
                    min_in = len(self.in_edge_list[i])
        # Create the TF network here. 
        print("Number of TFs are:{}".format(len(self.tfs)))
        self.tf_num = len(self.tfs)
        print("Lowest incoming nodes to a TF are:{}".format((min_in)))
        print("Number of input are:{}".format((input_count)))
        for tf_idx, tf in enumerate(self.tfs):
            self.tf_in_edge_list.append([])
            if self.self_loop_probability < np.random.rand():  
                self.tf_in_edge_list[tf_idx].append(tf_idx)
            for edge in self.in_edge_list[tf]:
                self.tf_in_edge_list[tf_idx].append(self.tf_idx_to_tf[edge])
        return 
    def reset_random_f(self, seed=0, and_probability=0.5, activator_probability=0.5):
        self.and_probability = and_probability
        self.activator_probability = activator_probability
        np.random.seed(seed)
        self.tree_rands_list = []
        self.node_rands_list = []
        self.node_rands = []
        self.node_perm = []
        for tf_in_edges in self.tf_in_edge_list:
            self.tree_rands_list.append(np.random.rand(max(0,len(tf_in_edges)-1)))
            self.node_rands.append(np.random.rand(1))
            self.node_rands_list.append(np.random.rand(max(0,len(tf_in_edges))))
            self.node_perm.append(np.random.permutation(len(tf_in_edges)))
            
    def F(self, s):
        and_probability = self.and_probability
        activator_probability = self.activator_probability
        snew = copy.copy(s)
#         self.tree_rands_list = [[0],[1,0],[0,1],[0,1],[]]
#         self.node_rands_list = [[1,1],[1,1,0],[0,0,1],[0,0,1],[]]
#         self.node_perm = [[0,1],[0,1,2],[1,2,0],[0,2,1],[]]
        for idx,i in enumerate(s):
            values = [s[eidx] for eidx in self.tf_in_edge_list[idx]]
            permuted_values = [values[eidx] for eidx in self.node_perm[idx]]
            node_rands = [self.node_rands[eidx] for eidx in self.tf_in_edge_list[idx]]
            if len(values)>0:
                snew[idx] = random_boolean_function(permuted_values, 
                                                    self.node_rands_list[idx], 
                                                    self.tree_rands_list[idx], 
                                                    and_probability, 
                                                    activator_probability)
        del s
        return snew
        
def update(state, F, eta=0):
    state = F(state)
    # Randomness. Each gene has a probability to switch to on or off with probability eta
    rand = np.random.random(len(state))
    state = np.logical_xor((rand<eta), state)
    return state       

def num_to_state(num, length):
    return [i=='1' for i in list(format(num, '0{}b'.format(length)))]

def state_to_num(state):
    return int(''.join([str(x) for x in np.array(state)*1]), base=2)

def connected_components(sz, F):
    # Returns list of components and the components as well
    components = [-1 for i in range(2**sz)]
    num_components = -1
    current_component = []
    for i in range(len(components)):
        if components[i] > -1:
            continue
        current = i
        num_components +=1
        current_component.append([])
        while components[current] <0:
            components[current] = num_components
            current_component[num_components].append(current)
            current = state_to_num(update(num_to_state(current, sz), F))
        # If you come to a component already visited then 
        # 1. Extend the old component
        # 2. Reduce the number of components
        # 3. Rename all the current component to the already visited component
        # 4. Delete the current component
        if components[current]!=num_components:
            current_component[components[current]].extend(current_component[num_components])
            for i in current_component[num_components]:
                components[i] = components[current]
            num_components-=1
            current_component = current_component[:-1]
    return (current_component, components, num_components+1)
def basin_transition(i,j,F,trials=100,eta=0.01):
    count = 0
    for x in i:
        for _ in range(trials):
            count+=1.0
            current = x
            while(True):
                current = state_to_num(update(num_to_state(current, 9), F, eta=eta))
                count+=1
                if current in j:
                    break
    return count/trials/len(i)

g = GNWNetwork('only_one_zero_in_node.tsv')
# g = GNWNetwork('test.tsv')
# g = GNWNetwork('determining_rel_paper.tsv')
minb=100
for i in range(1):
    g.reset_random_f(seed=i)
    basins, cc, num_basins = connected_components(g.tf_num, g.F)
    minb = num_basins if num_basins<minb else minb
#     print basins
#     if i%1000==0:
#         print i,minb
#     print("Num basins:{} Largest basin:{} Num basins:: \t1:{}\t2:{}\t3:{}\t4:{}".format(
#             num_basins,max([len(b) for b in basins]), 
#             sum([len(b)==1 for b in basins]),
#             sum([len(b)==2 for b in basins]),
#         sum([len(b)==3 for b in basins]),
#          sum([len(b)==4 for b in basins])))
    print("Num basins:{} Largest basin:{} {}".format( num_basins,max([len(b) for b in basins]),min([len(b) for b in basins])))
    
    for i,basin_i in enumerate(basins):
        for j,basin_j in enumerate(basins):
            if i!=j:
                print('MFPT from basin {} to {} is {}'.format(
                        i,j,basin_transition(basin_i,basin_j,g.F,trials=10,eta=0.01)))