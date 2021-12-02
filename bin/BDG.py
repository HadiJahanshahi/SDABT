# no import is needed

class BDG:
    """
    Bug dependency graph
    """
    def __init__(self):
        self.network            = {}
        self.undirected_network = {}
        self.bugs_dictionary    = {}
        self.degree             = {}
        self.depth              = {}
        self.priority           = {}
        self.severity           = {}
        self.n_nodes            = 0
        self.n_arcs             = 0
        self.n_clusters         = 0
    """
    Node operation
    """
    def add_node (self, bug):
        self.network[bug]               = []
        self.undirected_network[bug]    = []
        bug.network                     = self
        self.n_nodes                   += 1
        self.bugs_dictionary[bug.idx]   = bug
    def remove_node(self, bug):
        if len(bug.depends_on)>0:
            for bug_dep in bug.depends_on:
                self.network[bug_dep].remove(bug)
                self.undirected_network[bug_dep].remove(bug)
                self.undirected_network[bug].remove(bug_dep)
        if len(bug.blocks)>0:
            for bug_bloc in bug.blocks:
                self.network[bug].remove(bug_bloc)
                self.undirected_network[bug_bloc].remove(bug)           
                self.undirected_network[bug].remove(bug_bloc)           
        self.n_arcs -= (len(bug.blocks) + len(bug.depends_on))
        del self.network[bug]
        del self.undirected_network[bug]
        del self.bugs_dictionary[bug.idx]
        self.n_nodes -= 1
    """
    Arc operation
    """
    def add_arc (self, bug1, bug2):
        self.network[bug1].append(bug2)
        self.undirected_network[bug1].append(bug2)
        self.undirected_network[bug2].append(bug1)
        self.n_arcs += 1
    def remove_arc (self, bug1, bug2):
        self.network[bug1].remove(bug2)
        self.undirected_network[bug1].remove(bug2)
        self.undirected_network[bug2].remove(bug1)
        self.n_arcs -= 1
    """
    Subgraph enumeration
    """
    def cluster_update(self):
        for bug in self.network:
            bug.updated = False
        cl_num = 0
        for bug in self.network:
            if bug.updated == False:
                bug.cluster_update(cl_num)
                cl_num += 1
        self.n_clusters = cl_num
    """
    Graph metrics
    """    
    def update_degree(self):
        for bug in self.network: # First, we make sure all of them are updated.
            bug.update_node_degree()        
        self.degree    = {bug: bug.degree for bug in self.network}
    def update_depth(self):
        for bug in self.network: # First, we make sure all of them are updated.
            bug.update_node_depth()
        self.depth     = {bug: bug.depth for bug in self.network}
    def update_priority(self):
        self.priority  = {bug: bug.priority for bug in self.network}
    def update_severity(self):
        self.severity  = {bug: bug.severity for bug in self.network}
    """
    How the object of the class to be printed (presented)
    """        
    def __repr__(self):
        return f'Bug network with {self.n_nodes} nodes and {self.n_arcs} arcs.'
    def __str__(self):
        return f'BDG with {self.n_nodes} nodes and {self.n_arcs} arcs.'
