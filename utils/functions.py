from utils.prerequisites import * # Packages and these functions: isNaN, isnotNaN, 
                                # convert_string_to_array, mean_list, and string_to_time

class Functions:
    seed_        = 0
    tag_map      = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    @staticmethod
    def max_dictionary(dic):
        """
        Finding the max value of a dictionary. In case of tie, it will choose randomly.
        """
        max_dict         = max(list(dic.values()))
        Functions.seed_ += 1 
        random.seed(Functions.seed_)
        return random.choice([key for key, value in dic.items() if (value == max_dict)])
    @staticmethod
    def sum_dict (dic1 , dic2):
        """
        sum of two dictionaries by keys
        """
        dic1 = OrderedDict(sorted(dic1.items()))
        dic2 = OrderedDict(sorted(dic2.items()))
        return {key: dic1.get(key, 0) + dic2.get(key, 0) for key in set(dic1) | set(dic2)}
    @staticmethod
    def mean_list (x):
        '''
        Returns the mean of a list
        '''
        return np.mean(list(x))
    @staticmethod
    def undirected_network(network):
        '''
        This will convert a directed graph to undirected.
        '''
        undirected_net = copy.deepcopy(network)
        for node1 in undirected_net.network.keys():
            for node2 in undirected_net[node1]:
                assert node1 not in undirected_net[node2]
                undirected_net[node2].append(node1)
        return undirected_net

    @staticmethod
    def start_of_the_day(date):
        return np.datetime64(date)

    @staticmethod
    def end_of_the_day(date):
        return np.datetime64(date) + np.timedelta64(1, 'D')
    
    @staticmethod
    def read_files(project_name):
        try:
            list_of_developers = pickle.load(open(os.path.join('dat', project_name, 'list_of_developers.txt'), "rb"))
            time_to_fix_LDA    = pd.read_csv(os.path.join('dat', project_name, 'time_to_fix_LDA.csv'))
            time_to_fix_LDA.columns.values[0] = 'developer'
        except:
            list_of_developers = None
            time_to_fix_LDA    = None
        try:
            with open(os.path.join('dat', project_name, "feasible_bugs_actual.txt"), "rb") as fp:   #Pickling
                feasible_bugs_actual = pickle.load(fp)
        except:
            feasible_bugs_actual = None
            
        bug_evolution_db   = pd.read_csv(os.path.join('dat', project_name, 'bug_evolution_data_new.csv'))
        Whole_dataset      = pd.read_csv(os.path.join('dat', project_name, 'whole_dataset_new.csv'))
        # formating update
        try:
            bug_evolution_db.time       = bug_evolution_db.time.map(lambda x: string_to_time(x))
        except ValueError:
            bug_evolution_db.time       = bug_evolution_db.time.map(lambda x: string_to_time(x, '%m/%d/%Y %H:%M'))
        try:
            Whole_dataset.creation_time = Whole_dataset.creation_time.map(lambda x: string_to_time(x))
        except ValueError:
            Whole_dataset.creation_time = Whole_dataset.creation_time.map(lambda x: string_to_time(x,
                                                                                                   '%Y-%m-%d %H:%M:%S'))
        # setting index
        Whole_dataset = Whole_dataset.set_index(['id'])
        try:
            SVM_model = pickle.load(open(os.path.join('..','dat', project_name, 'SVM.sav'), 'rb'))
        except:
            SVM_model = None
        try:
            Whole_dataset['lemma'] = Whole_dataset['lemma'].map(eval)
        except:
            pass
        bug_evolution_db              = bug_evolution_db[bug_evolution_db.time<'2020'] # until the end of 2019
        """ Sorting evolutionary DB by time and status """
        custom_dict                   = {'introduced':0, 'NEW':1, 'ASSIGNED':2, 'REOPENED':3, 'assigned_to':4, 'RESOLVED':5, 
                                         'UPDATE':6, 'blocks':7, 'depends_on':8, 'VERIFIED':9, 'CLOSED':10}
        bug_evolution_db['rank']      = bug_evolution_db['status'].map(custom_dict)
        bug_evolution_db.sort_values(['time', 'rank'], inplace= True)
        bug_evolution_db.drop('rank', axis=1, inplace=True)
        bug_evolution_db.reset_index(drop = True, inplace = True)
        #converting severity to numbers
        Whole_dataset['severity_num'] = Whole_dataset['severity'].replace(['normal', 'critical', 'major', 'minor',
                                                                           'trivial', 'blocker', 'enhancement', 'S1', 'S2', 'S3', 'S4', '--', None],
                                                                          [3,5,4,2,1,6,0,5,4,3,1,3,3])
        #converting priority to numbers >> P1 is the most important and P5 is the least important one
        if ('P1' in Whole_dataset.priority.unique()) and (len(Whole_dataset.priority.unique()) in [5,6]):
            Whole_dataset['priority_num'] = Whole_dataset['priority'].replace(['--', 'P5', 'P4', 'P3','P2','P1'],
                                                                              [0,1,2,3,4,5])
        elif ('highest' in Whole_dataset.priority.unique()) and (len(Whole_dataset.priority.unique())==5):
            Whole_dataset['priority_num'] = Whole_dataset['priority'].replace(['lowest', 'low', 'medium', 'high', 'highest'],
                                                                              [1,2,3,4,5])
        else:
            raise Exception (f'undefined priority levels {Whole_dataset.priority_num.unique()}')
        Whole_dataset['assigned_to_detail.email'] = Whole_dataset['assigned_to_detail.email'].map(lambda x: x.lower())
        Whole_dataset = Whole_dataset[(Whole_dataset.creation_time<'2020-01-01')].copy()

        glove_embeddings_index = dict()
        with open(os.path.join('dat', 'Embeddings', 'glove.6B.100d.txt'), encoding="utf8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                glove_embeddings_index[word] = coefs

        return [bug_evolution_db, Whole_dataset, list_of_developers, 
                time_to_fix_LDA, SVM_model, feasible_bugs_actual, glove_embeddings_index]
    
    @staticmethod
    def max_tuple_index(tuple_):
        return max(tuple_,key=itemgetter(1))[0]
    
    @staticmethod
    def model_kp(p, c, T, LogToConsole=False, verbose=False):
        """
        Knapsak model (used for both RABT and DABT with different inputs.)
        """
        p     = np.array(p)
        c     = np.round(np.array(c),6)
        model = Model()
        model.params.LogToConsole = LogToConsole
        model.params.TimeLimit = 60
        m, n  = np.array(p).shape
        "i=1 to m developers and j=1 to n bugs"
        assert p.shape == c.shape
        # x_{ij} \in \{0,1\}
        x    = model.addVars(m, n, vtype= GRB.BINARY, name='x')
        # sum_{i=1}^{m}{sum_{j=1}^{n}{P_{ij} X_{ij}}}
        model.setObjective(quicksum(p[i,j]* x[i,j] for i in range(m) for j in range(n)), GRB.MAXIMIZE)
        # sum_{j=1}^{n}{C_{ij} X_{ij}} \le T_i
        for i in range(m):
            model.addConstr(quicksum(c[i,j]* x[i,j] for j in range(n)) <= T[i])
        # sum_{i=1}^{m}{X_{ij}} \le 1
        for j in range(n):
            model.addConstr(quicksum(x[i,j] for i in range(m)) <= 1)
        if verbose:
            model.write('model_kp.lp')
        model.optimize()
        return model, x
    
    @staticmethod
    def model_sdabt(p, c, simultaneous_job, Total_limit, L_C, T, blocking_ids, remaining_time, LogToConsole=False, verbose=False, TimeLimit = 1200):
        """[S-DABT Model]

        Args:
            p ([list]):                    [Suitability of a bug for a developer]
            c ([list]):                    [Bug fixing cost]
            simultaneous_job ([list]):     [The capacity of a developer to simultaneously solve bugs]
            Total_limit ([int]):           [Project Horizon]
            L_C ([list]):                  [Bug fixing cost used to calculate p]
            T ([list]):                    [binary schedule: Developer remaining capacity in each slot]
            blocking_ids ([list]):         [list of the ids of blocking bug according to possible_bugs list for each bug]
            remaining_time([list]):        [Fixing time of the blocking bugs that are already assigned]
            LogToConsole (bool, optional): [Whether to print the result of the IP model]. Defaults to False.
            verbose (bool, optional):      [Whether to save the model in a file]. Defaults to False.

        Returns:
            [GRB.model, GRB.variables]: [Returning the variables and the model at the end]
        """
        p                         = np.array(p)
        c                         = np.round(np.array(c),6)
        model                     = Model()
        model.params.LogToConsole = LogToConsole
        model.params.TimeLimit    = TimeLimit
        n_bugs, n_devs            = np.array(p).shape
        x = {}
        "i=1 to m developers and j=1 to n bugs"
        for bug_id in range(n_bugs):
            for dev_id in range(n_devs):
                n_slots = simultaneous_job[dev_id]
                for slot_n in range(n_slots):
                    for time in range(Total_limit):
                        x[bug_id, dev_id, slot_n, time] = (model.addVar(vtype= GRB.BINARY,
                                                                        name=f'x[{bug_id},{dev_id},{slot_n},{time}]'))
                        
        obj_func = 0
        for bug_id in range(n_bugs):
            for dev_id in range(n_devs):
                n_slots = simultaneous_job[dev_id]
                for slot_n in range(n_slots):
                    for time in range(Total_limit):
                        obj_func += p[bug_id,dev_id]* x[bug_id,dev_id,slot_n,time]
        model.setObjective(obj_func, GRB.MAXIMIZE)
        
        for bug_id in range(n_bugs):
            single_assignment_constraint = 0
            for dev_id in range(n_devs):
                n_slots = simultaneous_job[dev_id]
                for slot_n in range(n_slots):
                    for time in range(Total_limit):
                        single_assignment_constraint += x[bug_id,dev_id,slot_n,time]
            model.addConstr(single_assignment_constraint <= 1, name = "eq:IP-single-assignment")
            
        for bug_id in range(n_bugs):
            for dev_id in range(n_devs):
                n_slots = simultaneous_job[dev_id]
                for slot_n in range(n_slots):
                    for time in range(Total_limit):
                        C_i_d = L_C[bug_id][dev_id]
                        if (time+int(np.ceil(C_i_d))-1+1) <= Total_limit:
                            for time_prime in range(time, min(time+int(np.ceil(C_i_d))-1+1, Total_limit)):
                                model.addConstr(x[bug_id,dev_id,slot_n,time] <= T[dev_id][slot_n][time_prime], 
                                                name = "eq:IP-slot-time-availability")
              
        for bug_id in range(n_bugs):
            for dev_id in range(n_devs):
                n_slots = simultaneous_job[dev_id]
                for slot_n in range(n_slots):
                    for time in range(Total_limit):
                        C_i_d = L_C[bug_id][dev_id]
                        if (time+C_i_d-1) >= Total_limit:
                            model.addConstr(x[bug_id,dev_id,slot_n,time] == 0, name = "eq:IP-bug-solution-check")
                            
        for dev_id in range(n_devs):
            n_slots = simultaneous_job[dev_id]
            for slot_n in range(n_slots):
                for time in range(Total_limit):
                    if T[dev_id][slot_n][time] == 1:
                        one_bug_at_time_t = 0
                        for bug_id in range(n_bugs):
                            C_i_d = L_C[bug_id][dev_id]
                            for time_prime in range(max(time-int(np.ceil(C_i_d))+1, 0), time+1):
                                if T[dev_id][slot_n][time_prime] == 1:
                                    one_bug_at_time_t += x[bug_id,dev_id,slot_n,time_prime]
                        model.addConstr(one_bug_at_time_t <= 1, name = "eq:IP-one-bug-at-time-t")    

        # for dev_id in (range(n_devs)):
        #     n_slots = simultaneous_job[dev_id]
        #     for slot_n in range(n_slots):
        #         for time in range(Total_limit):
        #             if T[dev_id][slot_n][time] == 1:
        #                 for bug_id in (range(n_bugs)):
        #                     LHS = 0
        #                     C_i_d = L_C[bug_id][dev_id]
        #                     if (time+C_i_d) <= Total_limit:
        #                         for bug_id_prime in range(n_bugs):
        #                             C_i_d_prime = L_C[bug_id_prime][dev_id]
        #                             if bug_id != bug_id_prime:
        #                                 for time_prime in range(time, min(time+int(np.ceil(C_i_d))-1+1, Total_limit)):
        #                                     if (time_prime+C_i_d_prime-1) <= Total_limit:
        #                                         LHS += x[bug_id_prime,dev_id,slot_n,time_prime]
        #                         model.addConstr(LHS <= ((1-x[bug_id,dev_id,slot_n,time])*Total_limit), 
        #                                         name = "eq:IP-one-bug-at-each-slot")  

        for bug_id in range(n_bugs):
            if len(blocking_ids[bug_id])>0:
                blocked_bug = bug_id
                for blocking_bug in blocking_ids[bug_id]:
                    LHS_1 = 0
                    LHS_2 = 0
                    RHS   = 0
                    for dev_id in range(n_devs):
                        C_i_d   = L_C[blocking_bug][dev_id]
                        n_slots = simultaneous_job[dev_id]
                        for slot_n in range(n_slots):
                            for time in range(Total_limit):
                                LHS_1 += x[blocked_bug ,dev_id,slot_n,time]
                                LHS_2 += time * x[blocked_bug ,dev_id,slot_n,time]
                                RHS   += (time+C_i_d-1) * x[blocking_bug,dev_id,slot_n,time]
                    model.addConstr(((1-LHS_1)*Total_limit) + LHS_2 >= RHS+0.0001, name = "eq:IP-dependency1")
                    
        for bug_id in range(n_bugs):
            if len(blocking_ids[bug_id])>0:
                blocked_bug = bug_id
                for blocking_bug_idx in range(len(blocking_ids[bug_id])):
                    if len(remaining_time[blocked_bug])>0:
                        max_end_time_of_old_parent = remaining_time[blocked_bug][blocking_bug_idx]
                        if max_end_time_of_old_parent != None:                        
                            LHS_1 = 0
                            LHS_2 = 0
                            for dev_id in range(n_devs):
                                n_slots = simultaneous_job[dev_id]
                                for slot_n in range(n_slots):
                                    for time in range(Total_limit):
                                        LHS_1 += x[blocked_bug ,dev_id,slot_n,time]
                                        LHS_2 += time * x[blocked_bug ,dev_id,slot_n,time]
                            model.addConstr(((1-LHS_1)*Total_limit) + LHS_2 >= max_end_time_of_old_parent +0.0001, 
                                            name = "eq:IP-dependency-assigned-not-solved")
                        
        for bug_id in range(n_bugs):
            if len(blocking_ids[bug_id])>0:
                blocked_bug = bug_id
                for blocking_bug in blocking_ids[bug_id]:
                    LHS = 0
                    for dev_id in range(n_devs):
                        n_slots = simultaneous_job[dev_id]
                        for slot_n in range(n_slots):
                            for time in range(Total_limit):
                                LHS += x[blocked_bug,dev_id,slot_n,time]
                    RHS = 0
                    for dev_id_prime in range(n_devs):
                        n_slots_prime = simultaneous_job[dev_id_prime]
                        for slot_n_prime in range(n_slots_prime):     
                            for time in range(Total_limit):
                                RHS += x[blocking_bug,dev_id_prime,slot_n_prime,time]
                    model.addConstr(LHS<=RHS, name = "eq:IP-dependency3")
        if verbose:
            model.write('model_kp.lp')
        model.optimize()
        return model, x
    
    @staticmethod
    def lemmatizing(text, ls = True):
        tokens  = word_tokenize(text)
        if ls:
            lm_text = [WordNetLemmatizer().lemmatize(token, Functions.tag_map[tag[0]]) for token, tag in pos_tag(tokens) if (
                (token not in STOPWORDS) and (len(token) < 20) and (token.isalpha()))]
        else:
            lm_text = ''
            for token, tag in pos_tag(tokens):
                if ((token not in STOPWORDS) and (len(token) < 20) and (token.isalpha())):
                    lm_text += WordNetLemmatizer().lemmatize(token, Functions.tag_map[tag[0]]) + ' '
        return lm_text
    
    @staticmethod
    def convert_nan_to_space(text):
        if text != text:
            return ""
        else:
            return text.lower()

    @staticmethod
    def return_index(condition, list_):
        return [idx for idx, val in enumerate(list_) if (val == condition)]