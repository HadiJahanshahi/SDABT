from bin.BDG                 import BDG
from components.assignee     import Assignee
from components.bug          import Bug
from utils.functions         import Functions
from utils.prerequisites     import *
from utils.release_dates     import release_dates
from utils.attention_decoder import AttentionDecoder

class Discrete_event_simulation:
    """
    Simulation Process
    """
    random_state = 0
    date_number  = 0
    def __init__(self, bug_evolutionary_db, bug_info_db, list_of_developers, time_to_fix_LDA,
                 model, Tfidf_vect, project, feasible_bugs_actual, embeddings, resolution = 'actual', verbose = 0):
        if   (verbose == 'nothing') or (str(verbose) == "0"):
            self.verbose = 0
        elif (verbose == 'some')    or (str(verbose) == "1"):
            self.verbose = 1
        elif (verbose == 'all')     or (str(verbose) == "2"):
            self.verbose = 2
        else:
            raise Exception ('Verbose can be chosen from nothing, some, all, 0, 1, or 2.')
        self.project                     = project
        """ sorted evolutionary DB """
        self.bug_evolutionary_db         = bug_evolutionary_db
        """ Bug information DB """
        self.bug_info_db                 = bug_info_db
        self.bug_info_db['solving_time'] = None
        for each_bug in tqdm(self.bug_info_db.index, desc="fixing_time_calculation", position=0, leave=True):
            self.bug_info_db.loc[each_bug,'solving_time'] = self.fixing_time_calculation(each_bug)        
        """ Bug Dependency Graph """
        self.BDG                         = BDG() # initializing bug dependency graph
        """ real start of the project """
        self.birthday                    = string_to_time("2010-01-01 00:00:00", '%Y-%m-%d %H:%M:%S')
        self.last_decade                 = pd.date_range(start=self.bug_evolutionary_db.time.min().date(), 
                                                         end='01/01/2020')
        self.death                       = string_to_time("2020-01-01 00:00:00", '%Y-%m-%d %H:%M:%S')
        self.testing_time                = string_to_time("2018-01-01 00:00:00", '%Y-%m-%d %H:%M:%S')
        self.testing_time_counter        = np.where(self.last_decade == self.testing_time)[0][0]
        self.release_dates               = release_dates(self.project)
        self.time                        = None  # current timestamp
        self.filtered_date               = None
        self.filtered_date_assigned      = None
        self.day_counter                 = 0
        self.date                        = self.last_decade[self.day_counter]
        self.update                      = []    # what to update
        self.resolved_bugs               = {}    # Keeping track of resolved bugs
        self.resolution                  = resolution.lower()
        self.all_opened_bugs             = []
        self.svm_model                   = model
        self.svm_model_priority          = model
        self.lstm_model                  = None
        self.Tfidf_vect                  = Tfidf_vect
        self.max_acceptable_solving_time = self.acceptable_solving_time() #None
        self.available_developers_email  = self.possible_developers()  #list_of_developers
        self.time_to_fix_LDA             = time_to_fix_LDA
        self.project_horizon             = None # self.time_limit_calculaton()
        self.mean_solving_time           = None        
        self.developers_info             = self.developers_info_extraction()
        self.keep_track_of_resolved_bugs = {'assigned_bugs':[],       'assigned_developers'   :[], 'fixing_time_of_bugs':[],
                                            'accuracy_of_assignment'      :[],  'overdue_bugs':[],     'unresolved_bugs':[],
                                            'disregarded_dependency'      :[],  'accuracy_of_assignmentT':[],    'date': [],
                                            'accumulated_fixing_time'     :[],  'BDG_depth'   :[],          'BDG_degree':[], 
                                            'bug_degree' :[], 'bug_depth' :[],  'bug_blocking':[],  'early-on-time-late':[],
                                            'disregarded_dependency_later':[],'dev_total_time':[],      'dev_free_times':[],
                                            'utilization':[]
                                            }
        self.track_BDG_info              = {'date_open':[],    'n_of_bugs':[],     'n_of_arcs':[],          'depth_open':[],
                                            'degree_open':[], 'date_fixed':[],   'depth_fixed':[],        'degree_fixed':[],
                                            'priority_fixed':[],               'priority_open':[],
                                            'severity_fixed':[],               'severity_open':[]
                                            }
        self.assigned_bug_tracker        = pd.DataFrame(columns=self.bug_evolutionary_db.columns)
        self.summary_plus_desc           = [] # list
        self.summary_plus_desc_not_ls    = [] # not list
        self.dictionary                  = None
        self.bow_corpus                  = None
        self.tfidf_svm                   = None
        self.corpus_tfidf                = None
        self.corpus_freq                 = None
        self.SVM_Y_label                 = []
        self.LDA_time_to_solve           = []
        self.priorityY                   = []
        self.LDA_model                   = None
        self.optimal_n_topics            = None
        self.current_bug                 = None
        self.alpha                       = 0.5        # for CosTriage
        self.feasible_bugs_actual        = None       # feasible_bugs_actual
        self.embeddings                  = embeddings
        self.tokenizer                   = Tokenizer()
        self.ohe                         = OneHotEncoder()
        self.running_time                = {"training_time": 0, "model_preparation_for_testing":0, 
                                            "testing_time":0} # execution time per phase in seconds
    
    def verboseprint(self, code, **kwargs):
        """
        Args:
            code ([str]): [It tells where the print should happen and what should be printed]
        """
        if self.verbose == 0:
            """It prints nothing!"""
            pass
        if self.verbose in [1,2]:
            """It prints important Warnings."""
            if code == 'testing_phase_started':
                print(f'Starting the TESTING phase using the method {self.resolution}.\n'
                      f'Today is {self.date}.')
            elif code == 'bug_reintroduction':
                print(f'Bug ID {kwargs["bug_id"]} is already in BDG but is now introduced!')
            elif code == 'bug_removal_not_possible':
                print(f'Bug ID {kwargs["bug_id"]} can not be removed!!')
            elif code == 'bug_not_found':
                print(f'where is Bug#{kwargs["bug_id"]}.')
            elif code == "training_time":
                print (f"It took almost {round(self.running_time['training_time']/60)} mins to finish training.")
            elif code == "preparation_time":
                print (f"It took almost {round(self.running_time['model_preparation_for_testing']/60)} "
                        "mins to finish model preparation.")
            elif code == "testing_time":
                print (f"It took almost {round(self.running_time['testing_time']/60)} mins to finish testing.")
        if self.verbose == 2:
            """It prints all Warnings and daily progress."""
            if code == 'print_date':
                print(f'Bug ID {kwargs["date"]} can not be removed!!')

    def acceptable_solving_time(self):
        """We find IQR of the bugs opened during the training period."""
        training_bugs = self.bug_info_db[self.bug_info_db.creation_time < self.testing_time].copy()
        solving_times = training_bugs.solving_time
        all_numbers   = solving_times[solving_times != np.inf]
        Q1, Q3        = all_numbers.quantile([0.25,0.75])
        IQR           = Q3 - Q1
        Max_acceptable_solving_time = Q3 + (1.5*IQR)
        print('Max acceptable solving time', Max_acceptable_solving_time)
        return Max_acceptable_solving_time
    
    def possible_developers(self, list_of_developers=None):
        if list_of_developers == None:
            if self.max_acceptable_solving_time != None:
                dbcopy   = self.bug_info_db[self.bug_info_db.solving_time <= self.max_acceptable_solving_time].copy()
            else:
                dbcopy   = self.bug_info_db.copy()
            testing               = dbcopy[dbcopy['creation_time'] >= self.testing_time] # filtering on test data
            dbcopy                = dbcopy[dbcopy['creation_time'] <  self.testing_time] # filtering on training data
            dbcopy_feasible_bugs  = set(dbcopy.index) # indexes of all the bugs that have acceptable solving times
            dev_freq              = {}
            evol_db               = self.bug_evolutionary_db[((self.bug_evolutionary_db.time < self.testing_time) &
                                                              (self.bug_evolutionary_db.status == 'assigned_to'))]
            for i in range(len(evol_db)):
                dev = evol_db.iloc[i].detail
                bug = evol_db.iloc[i].bug
                if (('inbox@' in dev.lower()) or ('triaged@' in dev.lower()) 
                    or ('nobody@' in dev.lower()) or ('@js.bugs' in dev.lower())
                    or ('@seamonkey.bugs' in dev.lower()) or ('core.bugs' in dev.lower())
                    or ('gfx.bugs' in dev.lower()) or ('nss.bugs' in dev.lower()) 
                    or ('disabled.tld' in dev.lower()) or ('firefox.bugs' in dev.lower()) 
                    or ('mozilla-org.bugs' in dev.lower()) or ('mpopp@mozilla.com' in dev.lower())
                    or ('mozilla.bugs' in dev.lower()) or ('@evangelism' in dev.lower())
                    or ('libreoffice-bugs@lists.freedesktop.org' in dev.lower())):
                    """if it is still in inbox and not assigned"""
                    pass
                elif (dev.lower() not in dev_freq):
                    dev_freq[dev.lower()]  = []
                    if (bug not in dev_freq[dev.lower()]) and (bug in dbcopy_feasible_bugs):
                        dev_freq[dev.lower()].append(bug)
                else:
                    if (bug not in dev_freq[dev.lower()]) and (bug in dbcopy_feasible_bugs):
                        dev_freq[dev.lower()].append(bug)
            """
            determining how many feasible bugs each developer has fixed during the training phase
            """
            dev_freq = {key:len(val) for key, val in dev_freq.items()}
            q75, q25 = np.percentile(list(dev_freq.values()), [75 ,25])
            IQR      = q75 - q25
            """ filter out low seasonal developers (Less than IQR)"""
            key_val  = dev_freq.copy().items()
            for dev_, freq in key_val:
                if freq < IQR:
                    del dev_freq[dev_]
                if (('inbox@' in dev_.lower()) or ('triaged@' in dev_.lower()) 
                    or ('nobody@' in dev_.lower()) or ('@js.bugs' in dev_.lower())
                    or ('@seamonkey.bugs' in dev_.lower()) or ('core.bugs' in dev_.lower())
                    or ('gfx.bugs' in dev_.lower()) or ('nss.bugs' in dev_.lower()) 
                    or ('disabled.tld' in dev_.lower()) or ('firefox.bugs' in dev_.lower()) 
                    or ('mozilla-org.bugs' in dev_.lower()) or ('mpopp@mozilla.com' in dev_.lower())
                    or ('mozilla.bugs' in dev_.lower()) or ('@evangelism' in dev_.lower())
                    or ('libreoffice-bugs@lists.freedesktop.org' in dev_.lower())):
                    raise Exception ('why are bugs that are in inbox and not triaged still here?')
            dev_in_testing = testing.assigned_to.unique()
            keys_ = dev_freq.copy().keys()
            for dev_ in keys_:
                if dev_ not in dev_in_testing:
                    """
                    Removing developers that are not active in the testing phase
                    """
                    del dev_freq[dev_]            
            print('The # developers', len(dev_freq))
            return list(dev_freq.keys())
        else:
            print('The # developers', len(list_of_developers))
            return list_of_developers
        
    def time_limit_calculaton(self):
        """ WE SKIP IT FOR NOW
        
        According to Kashiwa's paper (Release-Aware Bug Triaging), each developer  
        requires the upper limit L and the assignment interval. the third quartile value of 
        the times required to fix the bugs in the dataset was calculated and rounded, to obtain L values 
        """
        L = self.bug_info_db.solving_time[self.bug_info_db.solving_time == self.bug_info_db.solving_time].quantile(0.75)
        return L
    
    def developers_info_extraction(self):
        "dev info db"
        developers_db = {}
        for dev in self.available_developers_email:
            dev_info = self.bug_info_db[self.bug_info_db['assigned_to_detail.email'].map(lambda x: x.lower()
                                                                                        ) == dev].iloc[0]
            idd      = dev_info['assigned_to_detail.id']
            email_   = dev_info['assigned_to_detail.email'].lower()
            developers_db[idd] = Assignee(email           = email_,
                                          id_             = idd,
                                          name            = dev_info['assigned_to_detail.real_name'],
                                          LDA_experience  = None,
                                          time_limit_orig = self.project_horizon)
        return developers_db
    
    def fixing_time_calculation(self, bug_id):
        """Using bug info and bug evolutionary database to calculate fixing time 
        
        Fixing time = Fixing date - Assigned date + 1

        Args:
            bug_id ([int]): [Bug ID]

        Returns:
            [int]: [Fixing Duration]
        """
        who_assigned  = self.bug_info_db.loc    [bug_id, 'assigned_to']
        filtered      = self.bug_evolutionary_db[self.bug_evolutionary_db.bug == bug_id].copy()
        try:
            assigned_filt = filtered[(filtered.status == 'assigned_to') & (filtered.detail == who_assigned)]
            assigned_time = list(assigned_filt.time)[0].date()
            assigned_idx  = assigned_filt.index[0]
            filtered_res  = filtered[filtered.index >= assigned_idx].copy() # filter on dates after assigning
            resolved_filt = filtered_res[(filtered_res.status == 'RESOLVED') | (filtered_res.status == 'CLOSED')]
            resolved_time = list(resolved_filt.time)[0].date()
            solving_time  = (resolved_time - assigned_time).days + 1
        except IndexError:
            solving_time = np.inf
        return solving_time
    
    def update_developers(self):
        new_db = {}
        for dev in self.developers_info:
            if (self.developers_info[dev].email.lower()) in self.available_developers_email:
                new_db[dev]                 = self.developers_info[dev]
                new_db[dev].time_limit      = self.project_horizon
                new_db[dev].time_limit_orig = self.project_horizon
            # else:
            #     raise Exception (f"Developer {self.developers_info[dev].email.lower()} is not in available_developers_email.")
            #     """ WE MAY NEED ANOTHER LOOP FOR self.available_developers_email in that case."""
        for dev in self.available_developers_email:
            dev_info = self.bug_info_db[self.bug_info_db['assigned_to_detail.email'].map(lambda x: x.lower()
                                                                                        ) == dev].iloc[0]
            idd      = dev_info['assigned_to_detail.id']
            if idd not in new_db:
                email_   = dev_info['assigned_to_detail.email'].lower()
                new_db[idd] = Assignee(email           = email_,
                                       id_             = idd,
                                       name            = dev_info['assigned_to_detail.real_name'],
                                       LDA_experience  = None,
                                       time_limit_orig = self.project_horizon)
        return new_db.copy()
    
    def corpus_update(self):
        """Creating a vocabulary corpus consisting of words in summary and description of the bugs. 
        """
        all_words_freq = {}
        for idx, bug in self.resolved_bugs.items():
            temp_sum = bug.sum_desc_lemmatized
            for word in temp_sum:
                if word not in all_words_freq:
                    all_words_freq[word] = 0
                all_words_freq[word] += 1
        for idx, bug in self.BDG.bugs_dictionary.items():
            temp_sum = bug.sum_desc_lemmatized
            for word in temp_sum:
                if word not in all_words_freq:
                    all_words_freq[word] = 0
                all_words_freq[word] += 1
        self.corpus_freq = []
        for word, freq in all_words_freq.items():
            """Removing infrequent or too frequent words."""
            if (freq > 15) and (freq < (len(self.bug_info_db)/2)):
                self.corpus_freq.append(word)
                
    def filter_low_freq (self, text, StopWords = STOPWORDS):
        """Filtering stopwords and peculiar vocabularies

        Args:
            text ([str]): [A sentence to be cleaned]
            StopWords ([list], optional): [List of stopwords]. Defaults to STOPWORDS.

        Returns:
            [list]: [cleaned list of meaningful vocabularies in the given texts]
        """
        text = [word for word in text if ((word in self.corpus_freq) and (word not in StopWords)
                                          and (len(word) < 20)  and (len(word) > 1))]
        return text
    
    @staticmethod
    def no_transform(text):
        return text
    
    def create_db_for_SVM_LDA (self):
        training_bugs = list(self.bug_info_db[self.bug_info_db.creation_time <self.testing_time].index)
        time_to_fix = []
        """resolved bugs"""
        for bug in self.resolved_bugs.values():
            if (bug.valid4assignment) and (bug.time_to_solve in [0, np.inf]):
                bug.valid4assignment     = False # ignoring the bugs with fixing time equal to zero (Same day fixing)
                bug.reason_of_invalidity = 'No fixing time'
#            elif (bug.valid4assignment) and (bug.time_to_solve > 0):
            elif (bug.idx in training_bugs) and (bug.time_to_solve not in [0, np.inf]):
                time_to_fix.append(bug.time_to_solve)
        """reopened bugs"""
        for bug in self.BDG.bugs_dictionary.values():
            # if (bug.valid4assignment) and (bug.time_to_solve > 0):
            if (bug.idx in training_bugs) and (bug.time_to_solve not in [0, np.inf]):# and (bug.valid4assignment):
                time_to_fix.append(bug.time_to_solve)
        """finding the max_acceptale_solving_time"""
        Q1, Q3                           = pd.Series(time_to_fix).quantile([0.25, 0.75])
        self.project_horizon             = int(Q3) # self.time_limit_calculaton()
        self.mean_solving_time           = np.mean(time_to_fix)
        IQR                              = Q3 - Q1
        self.max_acceptable_solving_time = Q3 + (1.5*IQR)
        print('NEW max_acceptable_solving_time', self.max_acceptable_solving_time)
        """resolved bugs"""
        for bug in self.resolved_bugs.values():
            bug.valid4assignment = bug.valid_bug_for_assignment(self.available_developers_email,
                                                                self.max_acceptable_solving_time,
                                                                self.bug_evolutionary_db)
            if   (bug.valid4assignment) and (bug.time_to_solve > self.max_acceptable_solving_time):
                bug.valid4assignment     = False # ignoring the bugs with ourlier fixing time
                bug.reason_of_invalidity = 'Lengthy solving time'
            elif ((bug.valid4assignment) and 
                  (bug.time_to_solve not in [0, np.inf]) and 
                  (bug.time_to_solve <= self.max_acceptable_solving_time)):
                self.summary_plus_desc.append(bug.sum_desc_lemmatized)
                self.summary_plus_desc_not_ls.append(bug.sum_desc_lem_not_ls)
                self.SVM_Y_label.append(bug.assigned_to)
                self.LDA_time_to_solve.append(bug.time_to_solve)
                self.priorityY.append(bug.priority)
        """reopened bugs"""
        for bug in self.BDG.bugs_dictionary.values():
            bug.valid4assignment = bug.valid_bug_for_assignment(self.available_developers_email,
                                                                self.max_acceptable_solving_time,
                                                                self.bug_evolutionary_db)
            if   (bug.valid4assignment) and (bug.time_to_solve > self.max_acceptable_solving_time):
                bug.valid4assignment     = False # ignoring the bugs with ourlier fixing time
                bug.reason_of_invalidity = 'Lengthy solving time'
            elif ((bug.valid4assignment) and 
                  (bug.time_to_solve not in [0, np.inf]) and 
                  (bug.time_to_solve <= self.max_acceptable_solving_time)):
                self.summary_plus_desc.append(bug.sum_desc_lemmatized)
                self.summary_plus_desc_not_ls.append(bug.sum_desc_lem_not_ls)
                self.SVM_Y_label.append(bug.assigned_to)
                self.LDA_time_to_solve.append(bug.time_to_solve)
                self.priorityY.append(bug.priority)
        self.dictionary   = Dictionary(self.summary_plus_desc)
        self.dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=100000)
        self.bow_corpus   = [self.dictionary.doc2bow(doc) for doc in self.summary_plus_desc]
        self.tfidf        = models.TfidfModel(self.bow_corpus)
        self.corpus_tfidf = self.tfidf[self.bow_corpus]
        
    def SVM_model(self):
        self.corpus_update() # updating the corpus needed for filter_low_freq
        self.tfidf_svm                  = TfidfVectorizer(tokenizer=self.no_transform, 
                                                          preprocessor=self.filter_low_freq)
        X_tfidf                         = self.tfidf_svm.fit_transform(self.summary_plus_desc)
        self.svm_model                  = svm.SVC(C=1000.0, kernel='linear', degree=5, 
                                                  gamma=0.001, probability=True)
        self.svm_model.fit(X_tfidf, self.SVM_Y_label)
        self.svm_model_priority         = svm.SVR(C=1000.0, kernel='linear', degree=5, gamma=0.001)
        self.svm_model_priority.fit(X_tfidf, self.priorityY)
        self.available_developers_email = list(self.svm_model.classes_)
        self.developers_info            = self.update_developers() # updating available developers info
    
    def modeling_lstm(self, input_dim, output_dim, embedding_shape,
                      embedding_matrix = None, size_of_vocabulary = None, model_weights = None):
        self.lstm_model = Sequential()
        self.lstm_model.add(Embedding(size_of_vocabulary, embedding_shape, weights=[embedding_matrix],
                                    input_length=input_dim,  trainable=True)) 
        self.lstm_model.add(Bidirectional(LSTM(200)))
        self.lstm_model.add(AttentionDecoder(150, 200))
        self.lstm_model.add(Dense(100, activation='relu'))
        self.lstm_model.add(Dense(output_dim, activation='softmax'))
        if model_weights != None:
            self.lstm_model.load_weights(model_weights)
        self.lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

    def LSTM_model(self, pad_length = 20, epochs = 30):
        embedding_shape        = len(self.embeddings['hi'])
        train_docs_reply       = self.summary_plus_desc_not_ls
        self.tokenizer.fit_on_texts(self.summary_plus_desc_not_ls)
        x_tr_seq               = self.tokenizer.texts_to_sequences(train_docs_reply)  # text to sequence
        x_tr_seq               = pad_sequences(x_tr_seq,  maxlen=pad_length)     # pad sequence
        size_of_vocabulary     = len(self.tokenizer.word_index) + 1                   # +1 for padding
        embedding_matrix       = np.zeros((size_of_vocabulary, embedding_shape))
        for word, i in self.tokenizer.word_index.items():
            embedding_vector = self.embeddings.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        ytrain_reply           = self.SVM_Y_label
        self.ohe.fit(np.array(ytrain_reply).reshape(-1, 1))
        ytrain_reply_enc = self.ohe.transform(np.array(ytrain_reply).reshape(-1, 1))
        n_outputs        = len(np.unique(ytrain_reply))
        self.modeling_lstm(input_dim = pad_length, output_dim = n_outputs, embedding_shape = embedding_shape,
                           embedding_matrix = embedding_matrix, size_of_vocabulary = size_of_vocabulary)       
        self.lstm_model.fit(np.array(x_tr_seq), np.array(ytrain_reply_enc.toarray()), epochs=epochs, verbose=0)

    @staticmethod
    def resort_dict (dictionary, new_keys):
        """ needed to sort developer list according to the SVM model """
        assert len(dictionary) == len(new_keys)
        new_dic = {}
        for dev_email in new_keys:
            for dev_id, dev_info in dictionary.items():
                if dev_info.search_by_email(dev_email.lower()):
                    new_dic[dev_id] = dev_info
                    break
        return new_dic        
        
    @staticmethod
    def symmetric_kl_divergence(p, q):
        """ Caluculates symmetric Kullback-Leibler divergence.
        """
        
        return np.sum([stats.entropy(p, q), stats.entropy(q, p)])
    
    def arun_metric(self, min_topics=5, max_topics=25, iteration=5):
        """ Caluculates Arun et al metric.."""
        corpus_length_vector = np.array([sum(frequency for _, frequency in document) for document in self.corpus_tfidf])   
        Kl_matrix  = []
        for j in tqdm(range(iteration), desc="LDA_best_n_topics", position = 0):
            result = []
            topic  = []
            for i in range(min_topics, max_topics):
                # Instanciates LDA.
                lda = models.ldamodel.LdaModel(corpus      = self.corpus_tfidf,
                                               id2word     = self.dictionary,
                                               num_topics  = i,
                                               iterations  = 80,
                                               random_state= Discrete_event_simulation.random_state+j+i*10)
                Discrete_event_simulation.random_state += 1
                # Caluculates raw LDA matrix.
                matrix                     = lda.expElogbeta
                # Caluculates SVD for LDA matris.
                U, document_word_vector, V = np.linalg.svd(matrix)
                # Gets LDA topics.
                lda_topics                 = lda[self.corpus_tfidf]
                # Caluculates document-topic matrix.
                term_document_matrix       = matutils.corpus2dense(lda_topics, lda.num_topics).transpose()
                document_topic_matrix      = corpus_length_vector.dot(term_document_matrix)
                document_topic_vector      = document_topic_matrix + 0.0001
                document_topic_norm        = np.linalg.norm(corpus_length_vector)
                document_topic_vector      = document_topic_vector / document_topic_norm
                result.append(self.symmetric_kl_divergence(document_word_vector,document_topic_vector))
                topic.append(i)
            Kl_matrix.append(result)
        ouput                 = np.array(Kl_matrix).mean(axis=0)
        self.optimal_n_topics = topic[ouput.argmin()]
    
    @staticmethod
    def average_days (list_, idx):
        if len(list_[idx]) > 0:
            return np.ceil(list_[idx].mean())
        else: 
            return None

    def LDA_Topic_Modeling (self, min_values=5):
        """ Finding the optimal number of topics using arun_metric """
        self.arun_metric()
        """applying LDA"""
        self.LDA_model       = models.LdaMulticore(self.corpus_tfidf, num_topics=self.optimal_n_topics,
                                                   id2word=self.dictionary, passes=20, workers=4)
        LDA_category         = [Functions.max_tuple_index(self.LDA_model[i]) for i in self.bow_corpus]
        categoris            = np.unique(LDA_category)
        """Creating Dev/LDA/time-to-fix table"""
        self.time_to_fix_LDA = pd.DataFrame(None, index = self.available_developers_email,
                                            columns = categoris)
        for dev in self.available_developers_email:
            index                      = Functions.return_index(dev, self.SVM_Y_label)
            filtered_LDA_cat           = np.array(LDA_category)[index]
            filtered_LDA_time_to_solve = np.array(self.LDA_time_to_solve)[index]
            for cat in categoris:
                self.time_to_fix_LDA.loc[dev, cat] = self.average_days(filtered_LDA_time_to_solve,
                                                                       Functions.return_index(cat, filtered_LDA_cat))
        """updating None Values of table"""
        self.dev_profile_collaborative(min_values)
        for dev_id, dev_info in self.developers_info.items():
            dev_info.LDA_experience = list(self.time_to_fix_LDA.loc[dev_info.email])
    
    def dev_profile_collaborative(self, min_value=5):
        """
        min_value=1: the updated date cannot be less than a day.
        """
        dev_profile     = self.time_to_fix_LDA.copy()
        dev_profile_upd = self.time_to_fix_LDA.copy()
        theta           = dev_profile_upd.shape[1]/2
        for dl in dev_profile.index:
            Pu   = dev_profile[dev_profile.index==dl]
            F_Pu = Pu.max(axis=1)[0]
            for bug in dev_profile.columns:
                if (dev_profile.at[dl,bug]==None):
                    Nu = []
                    for de in dev_profile.index:
                        if ((de!=dl) and (dev_profile.at[de,bug]!=None)):
                            Nu.append(de)
                    Nu_upd = []
                    for Pv_dev in Nu:
                        Num = 0
                        den = 0
                        Pv_dev_prof = dev_profile[dev_profile.index==Pv_dev]
                        Pv_sub      = []
                        Pu_sub      = []
                        for i in dev_profile.columns:
                            if (((Pu[i][0])!=None) & ((Pv_dev_prof[i][0])!=None)):
                                Pu_sub.append(Pu[i][0])
                                Pv_sub.append(Pv_dev_prof[i][0])
                        if len(Pv_sub) > 0:
                            weight     = min((len(Pv_sub)/theta),1)
                            cosine_sim = (np.dot(np.array(Pu_sub),np.array(Pv_sub)) /
                                          (np.linalg.norm(Pu_sub)*np.linalg.norm(Pv_sub)))
                            sim_pu_pv  = (cosine_sim)*weight
                            Nu_upd.append((sim_pu_pv,Pv_dev))
                    Nu_new = sorted(Nu_upd,reverse=True)
                    for i, val in enumerate(Nu_new[:10]):
                        F_Pv = dev_profile[dev_profile.index==val[1]].max(axis=1)[0]
                        Num += ((val[0])*(dev_profile.at[val[1],bug]/F_Pv))
                        den += val[0]
                    try:
                        dev_profile_upd.at[dl,bug] = np.ceil(max(round((F_Pu*(Num/den)),2), min_value))
                    except ZeroDivisionError:
                        """
                           It may happen that none of the similar dev has experience with the category 
                        that is None for the current developer. Then "den" will become zero.
                        """
                        dev_profile_upd.at[dl,bug] = np.ceil(min_value)
        self.time_to_fix_LDA = dev_profile_upd
    
    def predict_LDA (self, bug_n):
        doc2bow = self.dictionary.doc2bow(self.BDG.bugs_dictionary[bug_n].sum_desc_lemmatized)
        self.BDG.bugs_dictionary[bug_n].LDA_category = Functions.max_tuple_index(self.LDA_model[doc2bow])
    
    def update_date(self):
        self.day_counter += 1
        self.date         = self.last_decade[self.day_counter]
        if self.project_horizon != None:
            for dev in self.developers_info:
                if self.date >= self.testing_time:
                    self.developers_info[dev].increase_time_limit(self.resolution)
                else:
                    self.developers_info[dev].increase_time_limit()
        Discrete_event_simulation.date_number += 1
        if self.date == self.death:
            self.running_time["testing_time"] = round(time.time() - self.start_testing_time)
            self.verboseprint(code="testing_time")
            """
            Updating bug validity at the end of the testing phase.
            In this case, we will have an updated bug_info_db with 'valid' column.
            """
            for idx in tqdm(self.bug_info_db.index, desc="Updating DB before finishing", position=0, leave=True):
                try:
                    bug_info = self.BDG.bugs_dictionary[idx]
                except:
                    bug_info = self.resolved_bugs[idx]
                bug_info.valid4assignment = bug_info.valid_bug_for_assignment(self.available_developers_email,
                                                                              self.max_acceptable_solving_time,
                                                                              self.bug_evolutionary_db)
                self.bug_info_db.loc[idx, "valid"]                = bug_info.valid4assignment
                self.bug_info_db.loc[idx, "reason_of_invalidity"] = bug_info.reason_of_invalidity

    @staticmethod
    def CosTriage(L_S, L_C, alpha):
        L_S = np.array(L_S)
        L_C = np.array(L_C)
        return (alpha * (L_S/L_S.max())) + ((1-alpha) * ((1/L_C)/(1/L_C.min())))

    def Cost_Priority (self, L_S, L_C):
        L_S      = np.array(L_S)
        scaler   = MinMaxScaler(feature_range = (1,5))
        L_S_norm = np.array([i[0] for i in scaler.fit_transform(np.array(L_S).reshape(-1,1))])
        L_C      = np.array(L_C)
        scaler   = MinMaxScaler(feature_range = (1,5))
        L_C_norm = np.array([i[0] for i in scaler.fit_transform(np.array(L_C).reshape(-1,1))])
        return (self.alpha * (L_S_norm/L_S_norm.max())) + ((1-self.alpha) * ((1/L_C_norm)/(1/L_C_norm.min())))
    
    def filter_on_the_date (self, which_data):
        mask = (which_data['time'] > Functions.start_of_the_day(self.date)) & (
                which_data['time'] <= Functions.end_of_the_day(self.date))
        return which_data.loc[mask].copy()

    def find_assignment_divergence(self, bug_to_choose):
        if self.resolution == 'actual':
            date_diff = 0
        else:
            assigned_dates = self.bug_evolutionary_db[((self.bug_evolutionary_db.bug==bug_to_choose) & 
                                                       (self.bug_evolutionary_db.status=='assigned_to')
                                                       )].time.to_list()
            if len(assigned_dates) == 1:
                "assigned only once during its life cycle"
                date_diff = (assigned_dates[0] - self.date).days
            else:
                "assigned multiple times during its life cycle"
                all_diffs = [(assigned_date- self.date).days for assigned_date in assigned_dates]
                date_diff = min(all_diffs)
        return date_diff

    def track_and_assign (self, bug_to_choose, dev_info, date_formatted):
        selected_bug  = self.BDG.bugs_dictionary[bug_to_choose]
        selected_bug.update_node_degree()
        bug_degree    = selected_bug.degree
        selected_bug.update_node_depth ()
        bug_depth     = selected_bug.depth
        bug_blocking  = len(selected_bug.depends_on)
        bug_priority  = selected_bug.priority
        bug_severity  = selected_bug.severity
        date_diff     = self.find_assignment_divergence(bug_to_choose)            
        self.keep_track_of_resolved_bugs['bug_blocking'       ].append(bug_blocking  )
        self.keep_track_of_resolved_bugs['bug_depth'          ].append(bug_depth     )
        self.keep_track_of_resolved_bugs['bug_degree'         ].append(bug_degree    )
        self.keep_track_of_resolved_bugs['early-on-time-late' ].append(date_diff     )
        self.track_BDG_info             ['date_fixed'         ].append(date_formatted)
        self.track_BDG_info             ['depth_fixed'        ].append(bug_depth     )
        self.track_BDG_info             ['degree_fixed'       ].append(bug_degree    )
        self.track_BDG_info             ['priority_fixed'     ].append(bug_priority  )
        self.track_BDG_info             ['severity_fixed'     ].append(bug_severity  )
        self.keep_track_of_resolved_bugs['assigned_bugs'      ].append(bug_to_choose )
        self.keep_track_of_resolved_bugs['assigned_developers'].append(
            dev_info.email)
        self.keep_track_of_resolved_bugs['fixing_time_of_bugs'].append(
            selected_bug.solving_time_after_simulation)
        self.keep_track_of_resolved_bugs['accumulated_fixing_time'].append(
            selected_bug.solving_time_after_simulation_accumulated)
        self.keep_track_of_resolved_bugs['accuracy_of_assignment'].append(
            selected_bug.assignment_accuracy)
        self.keep_track_of_resolved_bugs['accuracy_of_assignmentT'].append(
            selected_bug.assignment_accuracyT)
        assigned_index = np.searchsorted(self.release_dates, self.date)
        solving_date  = self.date + np.timedelta64(
            math.ceil(selected_bug.solving_time_after_simulation_accumulated), 'D')
        solving_index  = np.searchsorted(self.release_dates, solving_date)
        if assigned_index != solving_index:
            self.keep_track_of_resolved_bugs['overdue_bugs'].append(bug_to_choose)
        if solving_date >= self.death:
            if bug_to_choose not in self.keep_track_of_resolved_bugs['overdue_bugs']:
                self.keep_track_of_resolved_bugs['unresolved_bugs'].append(bug_to_choose)
        self.assigned_bug_tracker = self.assigned_bug_tracker.append(
            pd.DataFrame([[bug_to_choose,'ASSIGNED_TO', dev_info.email ,solving_date]], 
                            columns=self.bug_evolutionary_db.columns))
        if len(selected_bug.depends_on)>0:
            selected_bug.ignored_dependency = 1
        else:
            selected_bug.ignored_dependency = 0
        self.keep_track_of_resolved_bugs['disregarded_dependency'].append(
            selected_bug.ignored_dependency)                                                  
        try:
            self.bug_evolutionary_db = self.bug_evolutionary_db.drop(
                self.bug_evolutionary_db[self.bug_evolutionary_db[
                    (self.bug_evolutionary_db.bug == bug_to_choose) & 
                    ((self.bug_evolutionary_db.status == 'CLOSED') | 
                        (self.bug_evolutionary_db.status == 'RESOLVED'))]].index)
        except ValueError:
            pass

    def triage(self):
        if Discrete_event_simulation.date_number == 0:                                               ####################
            """                                                                                      ####################
            It is the start of the training phase.                                                   ####################
            """                                                                                      ####################
            self.start_training = time.time()                                                        #     TRIAGING     #
        date_formatted = f'{self.date.year}-{self.date.month}-{self.date.day}'                       #      STARTS      #
        """ Updating based on bug evolutionary info """                                              #       HERE       #
        self.filtered_date = self.filter_on_the_date(self.bug_evolutionary_db) #updating filtered_date ##################
        for i in range(len(self.filtered_date)):                                                     ####################
            bug_id           = int(self.filtered_date.iloc[i].bug)                                   ####################
            self.current_bug = bug_id                                                                ####################
            status_i         = self.filtered_date.iloc[i].status
            self.time        = self.filtered_date.iloc[i].time
            detail           = self.filtered_date.iloc[i].detail
            if bug_id not in self.all_opened_bugs:
                """Add it to the list of opened bugs"""
                self.all_opened_bugs.append(bug_id)
            if   status_i.lower() == 'introduced' :
                # creating a bug.
                if bug_id not in self.BDG.bugs_dictionary:
                    """ Enusring the bug is not added before """ 
                    if (self.date < self.testing_time):
                        ac_sol_t = 100# self.max_acceptable_solving_time 
                        # 100 # We do not accept bugs with fixing time more than 100 days;
                    else:
                        ac_sol_t = self.max_acceptable_solving_time
                    Bug(idx                     = bug_id,
                        creation_time           = self.day_counter, 
                        severity                = self.bug_info_db.loc[bug_id, 'severity_num'         ],
                        priority                = self.bug_info_db.loc[bug_id, 'priority_num'         ],
                        comment_count           = self.bug_info_db.loc[bug_id, 'n_comments'           ],
                        last_status             = self.bug_info_db.loc[bug_id, 'status'               ],
                        description             = self.bug_info_db.loc[bug_id, 'description'          ],
                        summary                 = self.bug_info_db.loc[bug_id, 'summary'              ],
                        component               = self.bug_info_db.loc[bug_id, 'component'            ],
                        assigned_to_id          = self.bug_info_db.loc[bug_id, 'assigned_to_detail.id'],
                        assigned_to             = self.bug_info_db.loc[bug_id, 'assigned_to'          ],
                        time_to_solve           = self.bug_info_db.loc[bug_id, 'solving_time'         ],
                        LDA_category            = None,
                        dev_list                = self.available_developers_email,
                        acceptable_solving_time = ac_sol_t,
                        network                 = self.BDG,
                        bug_db                  = self.bug_evolutionary_db
                       )
                    self.bug_info_db.loc[self.bug_info_db.index==bug_id, 'valid'] = self.BDG.bugs_dictionary[bug_id].valid4assignment
                    self.update.append(bug_id)
                    if (self.date >= self.testing_time):
                        self.predict_LDA(bug_id)
                else:
                    self.verboseprint(code = 'bug_reintroduction', bug_id = bug_id)

            elif status_i.lower() == 'blocks'     :
                blocked_blocking = int(detail)
                if ((bug_id in list(self.bug_evolutionary_db.bug)) and
                    (blocked_blocking in list(self.bug_evolutionary_db.bug))):
                    if (bug_id in self.BDG.bugs_dictionary) and (bug_id not in self.update):
                        self.update.append(bug_id)
                    if (blocked_blocking in self.BDG.bugs_dictionary) and (blocked_blocking not in self.update):
                        self.update.append(blocked_blocking)
                    """ assert both were openned before """
                    if (blocked_blocking not in self.all_opened_bugs) or (bug_id not in self.all_opened_bugs):
                        raise Exception (f"bug id is {bug_id}, blocked-blocking is {blocked_blocking}, and /n {self.filtered_date}")
                    """different scenarios"""
                    """ I) If both are in BDG >> update both """
                    if (bug_id in self.BDG.bugs_dictionary) and (blocked_blocking in self.BDG.bugs_dictionary):
                        if (self.BDG.bugs_dictionary[blocked_blocking]) not in (self.BDG.bugs_dictionary[bug_id].blocks): 
                            self.BDG.bugs_dictionary[bug_id].blocks_bug(self.BDG.bugs_dictionary[blocked_blocking], 
                                                                        'mutual')
                        """ II) If blocking is in BDG but not blocked bug >> only update blocked"""
                    elif (bug_id not in self.BDG.bugs_dictionary) and (blocked_blocking in self.BDG.bugs_dictionary):
                        if (self.BDG.bugs_dictionary[blocked_blocking]) not in (self.resolved_bugs[bug_id].blocks): 
                            self.resolved_bugs[bug_id].blocks_bug(self.BDG.bugs_dictionary[blocked_blocking], 
                                                                  'one-sided')
                        """ III) If blocked is in BDG but not blocking bug >> only update blocking"""
                    elif (bug_id in self.BDG.bugs_dictionary) and (blocked_blocking not in self.BDG.bugs_dictionary):
                        if (self.BDG.bugs_dictionary[bug_id]) not in (self.resolved_bugs[blocked_blocking].depends_on):
                            self.resolved_bugs[blocked_blocking].depends_on_bug(self.BDG.bugs_dictionary[bug_id], 
                                                                                'one-sided')
                        """ IV) If none is in BDG >> update both """
                    elif (bug_id not in self.BDG.bugs_dictionary) and (blocked_blocking not in self.BDG.bugs_dictionary):
                        if (self.resolved_bugs[blocked_blocking]) not in (self.resolved_bugs[bug_id].blocks):
                            self.resolved_bugs[bug_id].blocks_bug(self.resolved_bugs[blocked_blocking],
                                                                  'try_mutual', add_arc=False)
                    else:
                        raise Exception ('None of the 4 conditions has met!')
                
            elif status_i.lower() == 'depends_on' :
                blocked_blocking = int(detail)
                if ((bug_id in list(self.bug_evolutionary_db.bug)) and
                    (blocked_blocking in list(self.bug_evolutionary_db.bug))):
                    if (bug_id in self.BDG.bugs_dictionary) and (bug_id not in self.update):
                        self.update.append(bug_id)
                    if (blocked_blocking in self.BDG.bugs_dictionary) and (blocked_blocking not in self.update):
                        self.update.append(blocked_blocking)
                    """ assert both were openned before """
                    if (blocked_blocking not in self.all_opened_bugs) or (bug_id not in self.all_opened_bugs):
                        raise Exception (f"bug id is {bug_id}, blocked-blocking is {blocked_blocking}, and /n {self.filtered_date}")
                    """different scenarios"""
                    """ I) If both are in BDG >> update both """
                    if (bug_id in self.BDG.bugs_dictionary) and (blocked_blocking in self.BDG.bugs_dictionary):
                        if ((self.BDG.bugs_dictionary[blocked_blocking]) not in (
                            self.BDG.bugs_dictionary[bug_id].depends_on)): 
                            self.BDG.bugs_dictionary[bug_id].depends_on_bug(self.BDG.bugs_dictionary[blocked_blocking],
                                                                            'mutual')
                        """ II) If blocking is in BDG but not blocked bug >> only update blocked"""
                    elif (bug_id not in self.BDG.bugs_dictionary) and (blocked_blocking in self.BDG.bugs_dictionary):
                        if (self.BDG.bugs_dictionary[blocked_blocking]) not in (self.resolved_bugs[bug_id].depends_on): 
                            self.resolved_bugs[bug_id].depends_on_bug(self.BDG.bugs_dictionary[blocked_blocking],
                                                                      'one-sided')
                        """ III) If blocked is in BDG but not blocking bug >> only update blocking"""
                    elif (bug_id in self.BDG.bugs_dictionary) and (blocked_blocking not in self.BDG.bugs_dictionary):
                        if (self.BDG.bugs_dictionary[bug_id]) not in (self.resolved_bugs[blocked_blocking].blocks):
                            self.resolved_bugs[blocked_blocking].blocks_bug(self.BDG.bugs_dictionary[bug_id], 
                                                                            'one-sided')
                        """ IV) If none is in BDG >> update both """
                    elif (bug_id not in self.BDG.bugs_dictionary) and (blocked_blocking not in self.BDG.bugs_dictionary):
                        if (self.resolved_bugs[blocked_blocking]) not in (self.resolved_bugs[bug_id].depends_on):
                            self.resolved_bugs[bug_id].depends_on_bug(self.resolved_bugs[blocked_blocking], 
                                                                      'try_mutual', add_arc=False)
                    else:
                        raise Exception ('None of the 4 conditions has met!')
                    
            elif status_i.lower() == 'assigned_to':
                """
                If we are in Waybackmachine mode or still in training phase or 
                the bug is not assigned in real world.
                """
                dev_email = detail.lower()
                if bug_id in self.resolved_bugs:
                    """ If the bug is solved before being assigned! """
                    pass
                elif ((self.resolution == 'actual')   or
                      (self.date < self.testing_time) or
                      (not self.BDG.bugs_dictionary[bug_id].valid4assignment)):
                    if dev_email in self.available_developers_email: # it is a feasible developer
                        for dev_id, dev_info in self.developers_info.items():
                            if dev_info.search_by_email(dev_email):
                                found = True
                                if (bug_id in self.BDG.bugs_dictionary):
                                    if ((self.date < self.testing_time) or 
                                        (not self.BDG.bugs_dictionary[bug_id].valid4assignment)):
                                        mode_='not_tracking'
                                        if self.BDG.bugs_dictionary[bug_id] not in dev_info.bugs:
                                            """ If the bug is not reassigned """
                                            self.BDG.bugs_dictionary[bug_id].update_node_degree()
                                            self.BDG.bugs_dictionary[bug_id].update_node_depth ()
                                            dev_info.assign_and_solve_bug(bug         = self.BDG.bugs_dictionary[bug_id],
                                                                          time_       = self.day_counter,
                                                                          mode_       = mode_,
                                                                          resolution_ = self.resolution,
                                                                          T_or_P      = 'triage')
                                            if self.BDG.bugs_dictionary[bug_id].valid4assignment:
                                                solving_date  = self.date + np.timedelta64(
                                                    math.ceil(self.BDG.bugs_dictionary[bug_id
                                                    ].solving_time_after_simulation_accumulated), 'D')
                                                if solving_date < self.testing_time:
                                                    self.assigned_bug_tracker = self.assigned_bug_tracker.append(
                                                        pd.DataFrame([[bug_id,'ASSIGNED_TO', dev_info.email ,solving_date]],
                                                                     columns=self.bug_evolutionary_db.columns))
                                    elif ((self.date >= self.testing_time) and 
                                          (self.resolution == 'actual') and
                                          (self.BDG.bugs_dictionary[bug_id].valid4assignment)):
                                        mode_='tracking'
                                        if self.BDG.bugs_dictionary[bug_id] not in dev_info.bugs:
                                            """ If the bug is not reassigned """
                                            if self.BDG.bugs_dictionary[bug_id].LDA_category == None:
                                                self.predict_LDA(bug_id)
                                            dev_info.assign_and_solve_bug(bug         = self.BDG.bugs_dictionary[bug_id],
                                                                          time_       = self.day_counter,
                                                                          mode_       = mode_,
                                                                          resolution_ = self.resolution,
                                                                          T_or_P      = 'triage')
                                            """ We keep track iff we are in testing phase """
                                            self.track_and_assign(bug_id, dev_info, date_formatted)
                                    else:
                                        raise Exception("problem with if statement.")
                                break
                        if not found:
                            raise Exception(f"Developer not found for bug id {bug_id}")
            
            elif status_i.lower() == 'reopened'   :
                """ 
                Maybe the bug is assigned but not solved yet. So we cannot re-open it.
                """
                if (bug_id in self.resolved_bugs) and (self.date < self.testing_time):
                    if (bug_id not in self.update):
                        self.update.append(bug_id)
                    self.resolved_bugs[bug_id].reopen(self.BDG, self.day_counter)
                    del self.resolved_bugs[bug_id]

            elif (status_i.lower() == 'resolved') or (status_i.lower() == 'closed'):
                if bug_id in self.BDG.bugs_dictionary:
                    if ((self.date < self.testing_time) or 
                        (not self.BDG.bugs_dictionary[bug_id].valid4assignment)): #or
                        # (self.BDG.bugs_dictionary[bug_id].open[0] < self.testing_time_counter)):
                        """
                        If we are in training phase or 
                        the bug is not assigned to an active developer
                        """
                        try:
                            self.update.remove(bug_id)
                        except ValueError:
                            pass
                        if bug_id not in self.resolved_bugs:
                            self.resolved_bugs[bug_id] = self.BDG.bugs_dictionary[bug_id]
                            self.BDG.bugs_dictionary[bug_id].resolve_bug(self.day_counter)
                                        
        """ Assign based on strategies other than Actual (It is done once a day) """
        if (self.date >= self.testing_time) and (self.resolution != 'actual'):
            """ If we are in testing mode and the bug is assigned to a valid developer """
            possible_bugs = []
            """ valid bugs and the ones that are not already assigned. 
            or if assigned, they should have been solved up until now.
            """
            assigned_not_solved = self.assigned_bug_tracker[self.assigned_bug_tracker.time>=self.date].copy()
            for bug_id, bug_info in self.BDG.bugs_dictionary.items():
                if ((bug_info.valid4assignment) and 
                    (bug_id not in assigned_not_solved.bug.values)):
                    # and (bug_info.open[0] >= self.testing_time_counter):
                    if (self.feasible_bugs_actual != None):
                        if (bug_id in self.feasible_bugs_actual):
                            possible_bugs.append(bug_id)
                    else:
                        possible_bugs.append(bug_id)
            if len(possible_bugs) > 0:
                """ If there exists any bug valid for assignment """
                
                if   (self.resolution == 'dabt'):
                    """ Dependency Aware Bug Triage """
                    possible_bugs_old = possible_bugs.copy()
                    possible_bugs     = []
                    for bug_id in possible_bugs_old:
                        if ((len(self.BDG.bugs_dictionary[bug_id].depends_on) == 0)): #and 
                             ## PROBABLY I NEED TO REMOVE BELOW LINE ##
                            #(self.BDG.bugs_dictionary[bug_id].open[0] >= self.testing_time_counter)):
                            possible_bugs.append(bug_id)
                    if len(possible_bugs) > 0: 
                        c = []
                        for dev in self.developers_info:
                            bug_c = []
                            for bug_id in possible_bugs:
                                if self.BDG.bugs_dictionary[bug_id].LDA_category == None:
                                    self.predict_LDA(bug_id)
                                LDA_cat = self.BDG.bugs_dictionary[bug_id].LDA_category
                                bug_c.append(self.developers_info[dev].LDA_experience[LDA_cat])
                            c.append(bug_c)
                        # The available time is determined by maximum time limits of all slots.
                        T = np.array([max(dev_info.time_limit) for dev_id, dev_info in self.developers_info.items()], dtype=object)
                        p = []
                        for bug_id in possible_bugs:
                            bug_sum_desc = self.BDG.bugs_dictionary[bug_id].sum_desc_lemmatized
                            Test_X_Tfidf = self.tfidf_svm.transform([bug_sum_desc])
                            L_S          = self.svm_model.predict_proba(Test_X_Tfidf)[0]
                            # sorting in the same way as L_S
                            L_C          = list(self.time_to_fix_LDA.loc[self.svm_model.classes_,
                                                                         self.BDG.bugs_dictionary[bug_id].LDA_category])
                            Sh = self.CosTriage(L_S, L_C, self.alpha)
                            p.append(Sh)
                        p            = np.array(p).T
                        model_, var_ = Functions.model_kp(p, c, T)
                        for v in model_.getVars():
                            if round(v.X) > 0:
                                dev_n , bug_idx    = [int(i) for i in re.findall(r'\d+', v.VarName)]
                                assigned_developer = list(self.developers_info.keys())[dev_n]
                                which_bug          = possible_bugs[bug_idx]
                                dev_info           = self.developers_info[assigned_developer]
                                if which_bug not in self.resolved_bugs:
                                    dev_info.assign_and_solve_bug(bug         = self.BDG.bugs_dictionary[which_bug],
                                                                  time_       = self.day_counter,
                                                                  mode_       ='tracking',
                                                                  resolution_ = self.resolution,
                                                                  T_or_P      = 'triage')
                                    self.track_and_assign(which_bug, dev_info, date_formatted)
                                    
                elif   (self.resolution == 'sdabt'):
                    """ Dependency and Schedule Aware Bug Triage """
                    # revising the list of possible_bugs based on dependency to infeasible bugs
                    remove_bugs = []
                    for bug_id in possible_bugs:
                        if ((len(self.BDG.bugs_dictionary[bug_id].depends_on) > 0)):
                            should_be_eliminated = True
                            for blocking_bug_info in self.BDG.bugs_dictionary[bug_id].depends_on:
                                blocking_bug_id = blocking_bug_info.idx
                                for idx_, second_search_bug_id in enumerate(possible_bugs):
                                    if blocking_bug_id == second_search_bug_id:
                                        should_be_eliminated = False
                                assigned_not_solved = self.assigned_bug_tracker[self.assigned_bug_tracker.time>=self.date].copy()
                                if blocking_bug_id in assigned_not_solved.bug.values:
                                    should_be_eliminated = False
                                if should_be_eliminated:
                                    remove_bugs.append(bug_id)
                                    break
                    possible_bugs = [i for i in possible_bugs if i not in remove_bugs]                            

                    # list of the ids of blocking bug according to possible_bugs list for each bug
                    blocking_ids = []
                    for bug_id in possible_bugs:
                        if ((len(self.BDG.bugs_dictionary[bug_id].depends_on) == 0)): 
                            blocking_ids.append([])
                        else:
                            blocking_ids_tmp = []
                            for blocking_bug_info in self.BDG.bugs_dictionary[bug_id].depends_on:
                                blocking_bug_id = blocking_bug_info.idx
                                for idx_, second_search_bug_id in enumerate(possible_bugs):
                                    if blocking_bug_id == second_search_bug_id:
                                        # adding the index of blocking bugs according to possible_bugs list. 
                                        blocking_ids_tmp.append(idx_)
                            blocking_ids.append(blocking_ids_tmp)
                    
                    # Fixing time of the blocking bugs that are already assigned
                    remaining_time_of_blocking_bugs = []
                    for bug_id in possible_bugs:
                        if ((len(self.BDG.bugs_dictionary[bug_id].depends_on) == 0)): 
                            remaining_time_of_blocking_bugs.append([])
                        else:
                            temp_blocking_ids = []
                            assigned_not_solved = self.assigned_bug_tracker[self.assigned_bug_tracker.time>=self.date].copy()
                            for blocking_bug_info in self.BDG.bugs_dictionary[bug_id].depends_on:
                                blocking_bug_id = blocking_bug_info.idx
                                if blocking_bug_id not in assigned_not_solved.bug.values:
                                    temp_blocking_ids.append(None)
                                else:
                                    blocking_bug_row = assigned_not_solved[assigned_not_solved.bug == blocking_bug_id]
                                    date_diff        = (blocking_bug_row.time - self.date)[0].days
                                    temp_blocking_ids.append(date_diff)
                            remaining_time_of_blocking_bugs.append(temp_blocking_ids)
                            
                    if len(possible_bugs) > 0: 
                        c                = []
                        simultaneous_job = []
                        T                = [] #dev schedule
                        for dev in self.developers_info:
                            bug_c = []
                            simultaneous_job.append(self.developers_info[dev].simultaneous_cap)
                            T.append(self.developers_info[dev].schedule)
                            for bug_id in possible_bugs:
                                if self.BDG.bugs_dictionary[bug_id].LDA_category == None:
                                    self.predict_LDA(bug_id)
                                LDA_cat = self.BDG.bugs_dictionary[bug_id].LDA_category
                                bug_c.append(self.developers_info[dev].LDA_experience[LDA_cat])
                            c.append(bug_c)
                        T       = np.array(T, dtype=object)
                        p       = []
                        L_C_all = []
                        for bug_id in possible_bugs:
                            bug_sum_desc = self.BDG.bugs_dictionary[bug_id].sum_desc_lemmatized
                            Test_X_Tfidf = self.tfidf_svm.transform([bug_sum_desc])
                            L_S          = self.svm_model.predict_proba(Test_X_Tfidf)[0]
                            # sorting in the same way as L_S
                            L_C          = list(self.time_to_fix_LDA.loc[self.svm_model.classes_,
                                                                         self.BDG.bugs_dictionary[bug_id].LDA_category])
                            L_C_all.append(L_C)
                            Sh = self.CosTriage(L_S, L_C, self.alpha)
                            p.append(Sh)
                        model_, var_ = Functions.model_sdabt(p, c, simultaneous_job, self.project_horizon, L_C_all, T, blocking_ids,
                                                             remaining_time_of_blocking_bugs)
                        for v in model_.getVars():
                            if round(v.X) > 0:
                                bug_idx, dev_n, slot_n, time_   = [int(i) for i in re.findall(r'\d+', v.VarName)]
                                assigned_developer              = list(self.developers_info.keys())[dev_n]
                                which_bug                       = possible_bugs[bug_idx]
                                dev_info                        = self.developers_info[assigned_developer]
                                bug                             = self.BDG.bugs_dictionary[which_bug]
                                fixing_time                     = int(np.ceil(dev_info.LDA_experience [bug.LDA_category]))
                                if which_bug not in self.resolved_bugs:
                                    dev_info.assign_and_solve_bug(bug         = self.BDG.bugs_dictionary[which_bug],
                                                                  time_       = self.day_counter,
                                                                  mode_       ='tracking',
                                                                  resolution_ = self.resolution,
                                                                  T_or_P      = 'Triage',
                                                                  which_slot  = slot_n,
                                                                  what_time   = time_)
                                    self.track_and_assign(which_bug, dev_info, date_formatted)

                elif (self.resolution == 'rabt'):
                    c = []
                    for dev in self.developers_info:
                        bug_c = []
                        for bug_id in possible_bugs:
                            if self.BDG.bugs_dictionary[bug_id].LDA_category == None:
                                self.predict_LDA(bug_id)
                            LDA_cat = self.BDG.bugs_dictionary[bug_id].LDA_category
                            bug_c.append(self.developers_info[dev].LDA_experience[LDA_cat])
                        c.append(bug_c)
                    T             = np.array([max(dev_info.time_limit) for dev_id, dev_info in self.developers_info.items()], dtype=object)
                    sum_desc_list = []
                    for bug_id in possible_bugs:
                        sum_desc_list.append((self.BDG.bugs_dictionary[bug_id].sum_desc_lemmatized))
                    Test_X_Tfidf = self.tfidf_svm.transform(sum_desc_list)
                    p            = self.svm_model.predict_proba(Test_X_Tfidf).T
                    model_, var_ = Functions.model_kp(p, c, T)
                    for v in model_.getVars():
                        if round(v.X) > 0:
                            dev_n , bug_idx    = [int(i) for i in re.findall(r'\d+', v.VarName)]
                            assigned_developer = list(self.developers_info.keys())[dev_n]
                            which_bug          = possible_bugs[bug_idx]
                            dev_info           = self.developers_info[assigned_developer]
                            if which_bug not in self.resolved_bugs:
                                dev_info.assign_and_solve_bug(bug         = self.BDG.bugs_dictionary[which_bug],
                                                              time_       = self.day_counter,
                                                              mode_       ='tracking',
                                                              resolution_ = self.resolution,
                                                              T_or_P      = 'Triage')
                                self.track_and_assign(which_bug, dev_info, date_formatted)
                                
                elif (self.resolution == 'costriage'):
                    for bug_id in possible_bugs:
                        if self.BDG.bugs_dictionary[bug_id].LDA_category == None:
                            self.predict_LDA(bug_id)
                    for bug_id in possible_bugs:
                        bug_sum_desc = self.BDG.bugs_dictionary[bug_id].sum_desc_lemmatized
                        Test_X_Tfidf = self.tfidf_svm.transform([bug_sum_desc])
                        L_S          = self.svm_model.predict_proba(Test_X_Tfidf)[0]
                        # sorting in the same way as L_S
                        L_C          = list(self.time_to_fix_LDA.loc[self.svm_model.classes_,
                                                                     self.BDG.bugs_dictionary[bug_id].LDA_category])
                        Sh           = self.CosTriage(L_S, L_C, self.alpha)
                        dev          = self.svm_model.classes_ [Sh.argmax()]
                        for dev_id, dev_inf in self.developers_info.items():
                            if dev_inf.search_by_email(dev):
                                break
                        dev_info     = self.developers_info[dev_id]
                        if bug_id not in self.resolved_bugs:
                            dev_info.assign_and_solve_bug(bug         = self.BDG.bugs_dictionary[bug_id],
                                                          time_       = self.day_counter,
                                                          mode_       ='tracking',
                                                          resolution_ = self.resolution,
                                                          T_or_P      = 'Triage')
                            self.track_and_assign(bug_id, dev_info, date_formatted)
                            
                elif (self.resolution == 'cbr'):
                    for bug_id in possible_bugs:
                        if self.BDG.bugs_dictionary[bug_id].LDA_category == None:
                            self.predict_LDA(bug_id)
                    for bug_id in possible_bugs:
                        bug_sum_desc = self.BDG.bugs_dictionary[bug_id].sum_desc_lemmatized
                        Test_X_Tfidf = self.tfidf_svm.transform([bug_sum_desc])
                        dev          = self.svm_model.predict_proba(Test_X_Tfidf)[0]
                        dev          = self.svm_model.classes_ [dev.argmax()]
                        for dev_id, dev_inf in self.developers_info.items():
                            if dev_inf.search_by_email(dev):
                                break
                        dev_info     = self.developers_info[dev_id]
                        if bug_id not in self.resolved_bugs:
                            dev_info.assign_and_solve_bug(bug         = self.BDG.bugs_dictionary[bug_id],
                                                          time_       = self.day_counter,
                                                          mode_       ='tracking',
                                                          resolution_ = self.resolution,
                                                          T_or_P      = 'Triage')
                            self.track_and_assign(bug_id, dev_info, date_formatted)

                elif (self.resolution == 'random'):
                    for bug_id in possible_bugs:
                        if self.BDG.bugs_dictionary[bug_id].LDA_category == None:
                            self.predict_LDA(bug_id)                    
                    for bug_id in possible_bugs:
                        if bug_id not in self.resolved_bugs:
                            np.random.seed(Discrete_event_simulation.random_state)
                            dev_id   = np.random.choice(list(self.developers_info.keys()))
                            Discrete_event_simulation.random_state += 1
                            dev_info = self.developers_info[dev_id]
                            dev_info.assign_and_solve_bug(bug         = self.BDG.bugs_dictionary[bug_id],
                                                          time_       = self.day_counter,
                                                          mode_       ='tracking',
                                                          resolution_ = self.resolution,
                                                          T_or_P      = 'Triage')
                            self.track_and_assign(bug_id, dev_info, date_formatted)
                        
                elif (self.resolution == 'deeptriage'):
                    for bug_id in possible_bugs:
                        if bug_id not in self.resolved_bugs:
                            bug_sum_desc        = self.BDG.bugs_dictionary[bug_id].sum_desc_lem_not_ls
                            x_test_seq          = self.tokenizer.texts_to_sequences(bug_sum_desc)
                            x_test_seq          = pad_sequences(x_test_seq, maxlen=20)
                            developer_index     = self.lstm_model.predict_classes(x_test_seq,verbose=0)
                            dev                 = [self.ohe.categories_[0][i] for i in developer_index][0]
                            for dev_id, dev_inf in self.developers_info.items():
                                if dev_inf.search_by_email(dev):
                                    break
                            dev_info     = self.developers_info[dev_id]
                            dev_info.assign_and_solve_bug(bug         = self.BDG.bugs_dictionary[bug_id],
                                                          time_       = self.day_counter,
                                                          mode_       ='tracking',
                                                          resolution_ = self.resolution,
                                                          T_or_P      = 'Triage')
                            self.track_and_assign(bug_id, dev_info, date_formatted)
                    
                else:
                    raise Exception (f"The resoultion {self.resolution} is not defined.")

        """ Updating whatever we assigned """
        self.filtered_date_assigned = self.filter_on_the_date(self.assigned_bug_tracker) # updating assigned bugs
        if len(self.filtered_date_assigned)>0:
            if (self.date < self.testing_time):
                """
                It updates the number of current assigned bugs for each developer.
                It will be used to estimate their simultanous job capacity at the beginning of
                the testing phase. The history is only recorded during the trainig phase. 
                """            
                for dev_id, dev_info in self.developers_info.items():
                    dev_email       = dev_info.email
                    dev_todays_task = sum(self.filtered_date_assigned.detail == dev_email)
                    dev_info.n_current_assigned.append(dev_todays_task)
            else:
                for i in range(len(self.filtered_date_assigned)):
                    bug_id      = int(self.filtered_date_assigned.iloc[i].bug)
                    self.time   = self.filtered_date_assigned.iloc[i].time
                    detail      = self.filtered_date_assigned.iloc[i].detail
                    dev_email   = detail.lower()
                    if bug_id not in self.all_opened_bugs:
                        raise Exception (f'Bug {bug_id} is solved but not opened!')
                    elif (not self.BDG.bugs_dictionary[bug_id].valid4assignment):
                        """
                        If we assigned an infeasible bug.
                        """
                        raise Exception ('How is it possible to assign an infeasible bug?')
                    else:
                        try:
                            if len(self.BDG.bugs_dictionary[bug_id].depends_on)>0:
                                self.BDG.bugs_dictionary[bug_id].ignored_dependency = 1
                            else:
                                self.BDG.bugs_dictionary[bug_id].ignored_dependency = 0
                            self.keep_track_of_resolved_bugs['disregarded_dependency_later'].append(
                                self.BDG.bugs_dictionary[bug_id].ignored_dependency)                            
                            self.update.remove(bug_id)
                        except ValueError:
                            self.verboseprint(code = 'bug_removal_not_possible', bug_id = bug_id)
                        COMMENT = True
                        if bug_id not in self.resolved_bugs:
                            if not COMMENT:
                                """
                                We now check the dependency in the assignment date, not fixing date.
                                So, it is commented out.
                                """
                                if len(self.BDG.bugs_dictionary[bug_id].depends_on)>0:
                                    self.BDG.bugs_dictionary[bug_id].ignored_dependency = 1
                                else:
                                    self.BDG.bugs_dictionary[bug_id].ignored_dependency = 0
                                self.keep_track_of_resolved_bugs['disregarded_dependency'].append(
                                    self.BDG.bugs_dictionary[bug_id].ignored_dependency)
                            self.resolved_bugs[bug_id] = self.BDG.bugs_dictionary[bug_id]
                            self.BDG.bugs_dictionary[bug_id].resolve_bug(self.day_counter)
                        else:
                            self.verboseprint(code = 'bug_not_found', bug_id = bug_id)
                    
        #os.system('cls') # clear screen
        self.BDG.update_degree   ()
        self.BDG.update_depth    ()
        self.BDG.update_priority ()
        self.BDG.update_severity ()
        if (self.date >= self.testing_time):
            self.keep_track_of_resolved_bugs['BDG_depth' ].append(np.mean(list(self.BDG.depth .values())))
            self.keep_track_of_resolved_bugs['BDG_degree'].append(np.mean(list(self.BDG.degree.values())))
            self.keep_track_of_resolved_bugs['date'      ].append(date_formatted)
        self.track_BDG_info ['date_open'  ].append(date_formatted)
        assert self.BDG.n_nodes == len(self.BDG.bugs_dictionary)
        self.track_BDG_info ['n_of_bugs'    ].append(self.BDG.n_nodes)
        self.track_BDG_info ['n_of_arcs'    ].append(self.BDG.n_arcs )
        self.track_BDG_info ['depth_open'   ].append(np.mean(list(self.BDG.depth .values())))
        self.track_BDG_info ['degree_open'  ].append(np.mean(list(self.BDG.degree.values())))
        self.track_BDG_info ['priority_open'].append(np.mean(list(self.BDG.priority.values())))
        self.track_BDG_info ['severity_open'].append(np.mean(list(self.BDG.severity.values())))
        self.verboseprint(code = 'print_date', date = date_formatted)
        self.update_date()
        total_time     = []
        available_time = []
        if (self.date >= self.testing_time):
            business = []
            for value in self.developers_info.values():
                busy = False
                for sch in value.schedule:
                    total_time.append(len(sch))
                    available_time.append(sum(sch))
                    if sum(sch) != (self.project_horizon):
                        busy = True
                business.append(busy)
            self.keep_track_of_resolved_bugs['utilization'].append(sum(business)/len(business))
            self.keep_track_of_resolved_bugs['dev_free_times'].append(available_time)
            self.keep_track_of_resolved_bugs['dev_total_time'].append(total_time)

        if self.date == self.testing_time:
            self.running_time["training_time"]   = round(time.time() - self.start_training)
            self.verboseprint(code = 'training_time')
            self.start_model_updates_for_testing = time.time()
            self.verboseprint(code = 'testing_phase_started')
            """Prepare data for SVM and LDA""" 
            self.create_db_for_SVM_LDA()
            self.SVM_model()
            self.LDA_Topic_Modeling(self.mean_solving_time) 
            # sorting dev info according to SVM labels.
            # This is crucial to have a consistent result 
            self.developers_info  = self.resort_dict(self.developers_info, self.svm_model.classes_)
            for dev_id, dev_info in self.developers_info.items():
                """
                we need to calculate the simultaneous capcity of each developer after the end of
                the training phase. Their schedule and time limit will be reset accordingly.
                """
                try:
                    dev_info.calculating_simultaneous_capcity("IQR")
                except:
                    print('dev_id', dev_id)
                    print('dev_info', dev_info)
                    print('dev_info.email', dev_info.email)
                    print('dev_info.n_current_assigned', dev_info.n_current_assigned)
                    raise Exception ('Empty simulatanous capacity')
            with open(f"dat/{self.project}/list_of_developers.txt", "wb") as fp:   #Pickling
                pickle.dump(list(self.time_to_fix_LDA.index), fp)
            self.time_to_fix_LDA.to_csv(f"dat/{self.project}/time_to_fix_LDA.csv")
            pickle.dump(self.svm_model,          open(f"dat/{self.project}/SVM.sav", 'wb'))
            pickle.dump(self.svm_model_priority, open(f"dat/{self.project}/SVM_priority.sav", 'wb'))
            pickle.dump(self.tfidf_svm,          open(f"dat/{self.project}/Tfidf_vect.pickle", "wb"))
            if self.resolution == 'deeptriage':
                self.LSTM_model()
                """saving models"""
                pickle.dump(self.lstm_model,         open(f"dat/{self.project}/LSTM.sav", 'wb'))
            for bug_info in self.BDG.bugs_dictionary.values():
                """Updating the validity of all bugs after having the models ready"""
                bug_info.valid4assignment = bug_info.valid_bug_for_assignment(self.available_developers_email,
                                                                              self.max_acceptable_solving_time,
                                                                              self.bug_evolutionary_db)
                self.bug_info_db.loc[self.bug_info_db.index==bug_info.idx, 'valid'] = bug_info.valid4assignment
            for bug_info in self.resolved_bugs.values(): 
                """Updating the validity of all RESOLVED bugs after having the models ready"""
                bug_info.valid4assignment = bug_info.valid_bug_for_assignment(self.available_developers_email,
                                                                              self.max_acceptable_solving_time,
                                                                              self.bug_evolutionary_db)                
                self.bug_info_db.loc[self.bug_info_db.index==bug_info.idx, 'valid'] = bug_info.valid4assignment
            self.running_time["model_preparation_for_testing"] = round(time.time() - self.start_model_updates_for_testing)
            self.verboseprint(code = 'preparation_time')
            self.start_testing_time                            = time.time()



        """ END OF THE DAY """

    def prioritization(self):
        if Discrete_event_simulation.date_number == 0:
            """
            It is the start of the training phase.
            """
            self.start_training = time.time()
        if self.date == self.testing_time:                                              ####################
            self.running_time["training_time"]   = round(time.time() - self.start_training) ################
            self.verboseprint(code = 'training_time')                                   ####################
            self.start_model_updates_for_testing = time.time()                          ####################
            self.verboseprint(code = 'testing_phase_started')                           ####################
            """Prepare data for SVM and LDA"""                                          #  PRIORITIZATION  #
            self.create_db_for_SVM_LDA()                                                #      STARTS      #                 
            self.SVM_model()                                                            #       HERE       #
            # sorting dev info according to SVM labels.                                 #################### 
            # This is crucial to have a consistent result                               ####################
            self.developers_info = self.resort_dict(self.developers_info, self.svm_model.classes_) #########
            self.LDA_Topic_Modeling(self.mean_solving_time)                             ####################
            """saving models"""
            with open(f"dat/{self.project}/list_of_developers.txt", "wb") as fp:   #Pickling
                pickle.dump(list(self.time_to_fix_LDA.index), fp)
            self.time_to_fix_LDA.to_csv(f"dat/{self.project}/time_to_fix_LDA.csv")
            pickle.dump(self.svm_model, open(f"dat/{self.project}/SVM.sav", 'wb'))
            pickle.dump(self.tfidf_svm, open(f"dat/{self.project}/Tfidf_vect.pickle", "wb"))
            for bug_info in self.BDG.bugs_dictionary.values():
               """Updating the validity of all bugs after having the models ready"""
               bug_info.valid4assignment = bug_info.valid_bug_for_assignment(
                   self.available_developers_email,
                   self.max_acceptable_solving_time,
                   self.bug_evolutionary_db)
            self.running_time["model_preparation_for_testing"] = round(time.time() - self.start_model_updates_for_testing)
            self.verboseprint(code = 'preparation_time')
            self.start_testing_time                            = time.time()

        date_formatted = f'{self.date.year}-{self.date.month}-{self.date.day}'
        """ Updating based on bug evolutionary info """
        self.filtered_date = self.filter_on_the_date(self.bug_evolutionary_db) #updating filtered_date
 
        for i in range(len(self.filtered_date)):
            bug_id           = int(self.filtered_date.iloc[i].bug)
            self.current_bug = bug_id
            status_i         = self.filtered_date.iloc[i].status
            self.time        = self.filtered_date.iloc[i].time
            detail           = self.filtered_date.iloc[i].detail
            if bug_id not in self.all_opened_bugs:
                """Add it to the list of opened bugs"""
                self.all_opened_bugs.append(bug_id)
            if   (status_i.lower() == 'introduced' ):
                # creating a bug.
                if bug_id not in self.BDG.bugs_dictionary:
                    # Enusring the bug is not added before
                    if (self.date < self.testing_time):
                        ac_sol_t = self.max_acceptable_solving_time #None
                    else:
                        ac_sol_t = self.max_acceptable_solving_time
                    Bug(idx                     = bug_id,
                        creation_time           = self.day_counter, 
                        severity                = self.bug_info_db.loc[bug_id, 'severity_num'         ],
                        priority                = self.bug_info_db.loc[bug_id, 'priority_num'         ],
                        comment_count           = self.bug_info_db.loc[bug_id, 'n_comments'           ],
                        last_status             = self.bug_info_db.loc[bug_id, 'status'               ],
                        description             = self.bug_info_db.loc[bug_id, 'description'          ],
                        summary                 = self.bug_info_db.loc[bug_id, 'summary'              ],
                        component               = self.bug_info_db.loc[bug_id, 'component'            ],
                        assigned_to_id          = self.bug_info_db.loc[bug_id, 'assigned_to_detail.id'],
                        assigned_to             = self.bug_info_db.loc[bug_id, 'assigned_to'          ],                        
                        time_to_solve           = self.fixing_time_calculation(bug_id),
                        LDA_category            = None,
                        dev_list                = self.available_developers_email,
                        acceptable_solving_time = ac_sol_t,
                        network                 = self.BDG,
                        bug_db                  = self.bug_evolutionary_db
                       )
                    self.update.append(bug_id)
                    if (self.date >= self.testing_time):
                        self.predict_LDA(bug_id)
                else:
                    self.verboseprint(code = 'bug_reintroduction', bug_id = bug_id)

            elif (status_i.lower() == 'blocks'     ):
                blocked_blocking = int(detail)
                if ((bug_id in list(self.bug_evolutionary_db.bug)) and
                    (blocked_blocking in list(self.bug_evolutionary_db.bug))):
                    if (bug_id in self.BDG.bugs_dictionary) and (bug_id not in self.update):
                        self.update.append(bug_id)
                    if (blocked_blocking in self.BDG.bugs_dictionary) and (blocked_blocking not in self.update):
                        self.update.append(blocked_blocking)
                    """ assert both were openned before """
                    if (blocked_blocking not in self.all_opened_bugs) or (bug_id not in self.all_opened_bugs):
                        raise Exception (f"bug id is {bug_id}, blocked-blocking is {blocked_blocking}, and /n {self.filtered_date}")
                    """different scenarios"""
                    """ I) If both are in BDG >> update both """
                    if (bug_id in self.BDG.bugs_dictionary) and (blocked_blocking in self.BDG.bugs_dictionary):
                        if (self.BDG.bugs_dictionary[blocked_blocking]) not in (self.BDG.bugs_dictionary[bug_id].blocks): 
                            self.BDG.bugs_dictionary[bug_id].blocks_bug(self.BDG.bugs_dictionary[blocked_blocking], 
                                                                        'mutual')
                        """ II) If blocking is in BDG but not blocked bug >> only update blocked"""
                    elif (bug_id not in self.BDG.bugs_dictionary) and (blocked_blocking in self.BDG.bugs_dictionary):
                        if (self.BDG.bugs_dictionary[blocked_blocking]) not in (self.resolved_bugs[bug_id].blocks): 
                            self.resolved_bugs[bug_id].blocks_bug(self.BDG.bugs_dictionary[blocked_blocking], 
                                                                  'one-sided')
                        """ III) If blocked is in BDG but not blocking bug >> only update blocking"""
                    elif (bug_id in self.BDG.bugs_dictionary) and (blocked_blocking not in self.BDG.bugs_dictionary):
                        if (self.BDG.bugs_dictionary[bug_id]) not in (self.resolved_bugs[blocked_blocking].depends_on):
                            self.resolved_bugs[blocked_blocking].depends_on_bug(self.BDG.bugs_dictionary[bug_id], 
                                                                                'one-sided')
                        """ IV) If none is in BDG >> update both """
                    elif (bug_id not in self.BDG.bugs_dictionary) and (blocked_blocking not in self.BDG.bugs_dictionary):
                        if (self.resolved_bugs[blocked_blocking]) not in (self.resolved_bugs[bug_id].blocks):
                            self.resolved_bugs[bug_id].blocks_bug(self.resolved_bugs[blocked_blocking],
                                                                  'try_mutual', add_arc=False)
                    else:
                        raise Exception ('None of the 4 conditions has met!')
                
            elif (status_i.lower() == 'depends_on' ) :
                blocked_blocking = int(detail)
                if ((bug_id in list(self.bug_evolutionary_db.bug)) and
                    (blocked_blocking in list(self.bug_evolutionary_db.bug))):
                    if (bug_id in self.BDG.bugs_dictionary) and (bug_id not in self.update):
                        self.update.append(bug_id)
                    if (blocked_blocking in self.BDG.bugs_dictionary) and (blocked_blocking not in self.update):
                        self.update.append(blocked_blocking)
                    """ assert both were openned before """
                    if (blocked_blocking not in self.all_opened_bugs) or (bug_id not in self.all_opened_bugs):
                        raise Exception (f"bug id is {bug_id}, blocked-blocking is {blocked_blocking}, and /n {self.filtered_date}")
                    """different scenarios"""
                    """ I) If both are in BDG >> update both """
                    if (bug_id in self.BDG.bugs_dictionary) and (blocked_blocking in self.BDG.bugs_dictionary):
                        if ((self.BDG.bugs_dictionary[blocked_blocking]) not in (
                            self.BDG.bugs_dictionary[bug_id].depends_on)): 
                            self.BDG.bugs_dictionary[bug_id].depends_on_bug(self.BDG.bugs_dictionary[blocked_blocking],
                                                                            'mutual')
                        """ II) If blocking is in BDG but not blocked bug >> only update blocked"""
                    elif (bug_id not in self.BDG.bugs_dictionary) and (blocked_blocking in self.BDG.bugs_dictionary):
                        if (self.BDG.bugs_dictionary[blocked_blocking]) not in (self.resolved_bugs[bug_id].depends_on): 
                            self.resolved_bugs[bug_id].depends_on_bug(self.BDG.bugs_dictionary[blocked_blocking],
                                                                      'one-sided')
                        """ III) If blocked is in BDG but not blocking bug >> only update blocking"""
                    elif (bug_id in self.BDG.bugs_dictionary) and (blocked_blocking not in self.BDG.bugs_dictionary):
                        if (self.BDG.bugs_dictionary[bug_id]) not in (self.resolved_bugs[blocked_blocking].blocks):
                            self.resolved_bugs[blocked_blocking].blocks_bug(self.BDG.bugs_dictionary[bug_id], 
                                                                            'one-sided')
                        """ IV) If none is in BDG >> update both """
                    elif (bug_id not in self.BDG.bugs_dictionary) and (blocked_blocking not in self.BDG.bugs_dictionary):
                        if (self.resolved_bugs[blocked_blocking]) not in (self.resolved_bugs[bug_id].depends_on):
                            self.resolved_bugs[bug_id].depends_on_bug(self.resolved_bugs[blocked_blocking], 
                                                                      'try_mutual', add_arc=False)
                    else:
                        raise Exception ('None of the 4 conditions has met!')

            elif (status_i.lower() == 'assigned_to'):
                """
                If we are in Waybackmachine mode or still in training phase or 
                the bug is not assigned in real world.
                """
                dev_email = detail.lower()
                if bug_id in self.resolved_bugs:
                    """ If the bug is solved before being assigned! """
                    pass
                elif ((self.resolution == 'actual')   or
                      (self.date < self.testing_time) or
                      (not self.BDG.bugs_dictionary[bug_id].valid4assignment)):
                    if dev_email in self.available_developers_email: # it is a feasible developer
                        for dev_id, dev_info in self.developers_info.items():
                            if dev_info.search_by_email(dev_email):
                                if bug_id in self.BDG.bugs_dictionary:                                    
                                    if ((self.date < self.testing_time) or 
                                        (not self.BDG.bugs_dictionary[bug_id].valid4assignment)):
                                        mode_='not_tracking'
                                        if self.BDG.bugs_dictionary[bug_id] not in dev_info.bugs:
                                            """ If the bug is not reassigned """
                                            self.BDG.bugs_dictionary[bug_id].update_node_degree()
                                            self.BDG.bugs_dictionary[bug_id].update_node_depth ()
                                            dev_info.assign_bug(self.BDG.bugs_dictionary[bug_id],
                                                                self.day_counter,
                                                                mode_=mode_)
                                    elif ((self.date >= self.testing_time) and 
                                          (self.resolution == 'actual') and
                                          (self.BDG.bugs_dictionary[bug_id].valid4assignment)):
                                        mode_='tracking'
                                        if self.BDG.bugs_dictionary[bug_id] not in dev_info.bugs:
                                            """ If the bug is not reassigned """
                                            if self.BDG.bugs_dictionary[bug_id].LDA_category == None:
                                                self.predict_LDA(bug_id)
                                            dev_info.assign_and_solve_bug(self.BDG.bugs_dictionary[bug_id],
                                                                          self.day_counter,
                                                                          mode_       = mode_,
                                                                          resolution_ = self.resolution,
                                                                          T_or_P      = 'prioritization')
                                            """ We keep track iff we are in testing phase """
                                            self.track_and_assign(bug_id, dev_info, date_formatted)
                                    else:
                                        raise Exception("problem with if statement.")

                                found = True
                                break
                        if not found:
                            raise Exception(f"Developer not found for bug id {bug_id}")

                """
                check validity
                """
                if bug_id in self.BDG.bugs_dictionary:
                    if self.BDG.bugs_dictionary[bug_id].valid4assignment:
                        valid = True
                    else:
                        valid = False
                elif bug_id in self.resolved_bugs:
                    if self.resolved_bugs[bug_id].valid4assignment:
                        valid = True
                    else:
                        valid = False
                else:
                    raise Exception(f'Can not check the validity of the bug {bug_id}.')                   

                if ((self.resolution != 'actual') and
                    (self.date >= self.testing_time)      and
                    (valid)):
                    mode_='tracking'
                    """
                    We are using a new resolution to prioritize &
                    We are in a time of valid assignment &
                    We are in testing phase 
                    """
                    possible_bugs = []
                    """ valid bugs and the ones that are not already assigned. """
                    assigned_not_solved = self.assigned_bug_tracker[self.assigned_bug_tracker.time>=self.date].copy()
                    for bug_id_, bug_info in self.BDG.bugs_dictionary.items():
                        if ((bug_info.valid4assignment) and 
                            (bug_id_ not in assigned_not_solved.bug.values)): 
                            if (self.feasible_bugs_actual != None):
                                if (bug_id_ in self.feasible_bugs_actual):
                                    possible_bugs.append(bug_id_)
                            else:
                                possible_bugs.append(bug_id_)
                    if len(possible_bugs) > 0:
                        """ If there exists any bug valid for prioritization """
                        if   (self.resolution == 'max_depth_degree'):
                            bug_to_choose     = 0
                            max_depth_degree_ = -1
                            for bug_id_ in possible_bugs:
                                self.BDG.bugs_dictionary[bug_id_].update_node_degree()
                                self.BDG.bugs_dictionary[bug_id_].update_node_depth ()
                                bug_degree    = self.BDG.bugs_dictionary[bug_id_].degree
                                bug_depth     = self.BDG.bugs_dictionary[bug_id_].depth
                                if (bug_degree + bug_depth) > max_depth_degree_:
                                    bug_to_choose     = bug_id_
                                    max_depth_degree_ = (bug_degree + bug_depth)
                            dev_email = self.BDG.bugs_dictionary[bug_to_choose].assigned_to
                            for dev_id, dev_info in self.developers_info.items():
                                if dev_info.search_by_email(dev_email):
                                    break
                            dev_info.assign_and_solve_bug(self.BDG.bugs_dictionary[bug_to_choose],
                                                          self.day_counter,
                                                          mode_       = mode_,
                                                          resolution_ = self.resolution,
                                                          T_or_P      = 'prioritization')
                            """ We keep track iff we are in testing phase """
                            self.track_and_assign(bug_to_choose, dev_info, date_formatted)
                        ##################
                        elif (self.resolution == 'random'):
                            np.random.seed(Discrete_event_simulation.random_state)
                            bug_to_choose = np.random.choice(possible_bugs)
                            Discrete_event_simulation.random_state += 1                            
                            dev_email     = self.BDG.bugs_dictionary[bug_to_choose].assigned_to
                            for dev_id, dev_info in self.developers_info.items():
                                if dev_info.search_by_email(dev_email):
                                    break
                            dev_info.assign_and_solve_bug(self.BDG.bugs_dictionary[bug_to_choose],
                                                          self.day_counter,
                                                          mode_       = mode_,
                                                          resolution_ = self.resolution,
                                                          T_or_P      = 'prioritization')
                            """ We keep track iff we are in testing phase """
                            self.track_and_assign(bug_to_choose, dev_info, date_formatted)
                        ##################
                        elif (self.resolution == 'max_priority'):
                            #  >> P1 is the most important and P5 is the least important one <<
                            bug_to_choose     = 0
                            max_priority_     = -1
                            """ The GREATER the priority number is, the BETTER it will be."""
                            for bug_id_ in possible_bugs:
                                priority      = self.BDG.bugs_dictionary[bug_id_].priority
                                if priority is not None:
                                    if priority > max_priority_:
                                        bug_to_choose = bug_id_
                                        max_priority_ = priority
                            dev_email = self.BDG.bugs_dictionary[bug_to_choose].assigned_to
                            for dev_id, dev_info in self.developers_info.items():
                                if dev_info.search_by_email(dev_email):
                                    break
                            dev_info.assign_and_solve_bug(self.BDG.bugs_dictionary[bug_to_choose],
                                                          self.day_counter,
                                                          mode_       = mode_,
                                                          resolution_ = self.resolution,
                                                          T_or_P      = 'prioritization')
                            """ We keep track iff we are in testing phase """
                            self.track_and_assign(bug_to_choose, dev_info, date_formatted)
                        ##################
                        elif (self.resolution == 'max_severity'):
                            bug_to_choose     = 0
                            max_severity_     = -1
                            """ The LARGER the severity number is, the BETTER it will be."""
                            for bug_id_ in possible_bugs:
                                severity      = self.BDG.bugs_dictionary[bug_id_].severity
                                if severity > max_severity_:
                                    bug_to_choose = bug_id_
                                    max_severity_ = severity
                            dev_email = self.BDG.bugs_dictionary[bug_to_choose].assigned_to
                            for dev_id, dev_info in self.developers_info.items():
                                if dev_info.search_by_email(dev_email):
                                    break
                            dev_info.assign_and_solve_bug(self.BDG.bugs_dictionary[bug_to_choose],
                                                          self.day_counter,
                                                          mode_       = mode_,
                                                          resolution_ = self.resolution,
                                                          T_or_P      = 'prioritization')
                            """ We keep track iff we are in testing phase """
                            self.track_and_assign(bug_to_choose, dev_info, date_formatted)
                        ##################
                        elif (self.resolution == 'cost_estimation'):
                            """Choose a bug with the minimum fixing time"""
                            for bug_id in possible_bugs:
                                if self.BDG.bugs_dictionary[bug_id].LDA_category == None:
                                    self.predict_LDA(bug_id)
                            fixing_time_all = []
                            for bug_id in possible_bugs:
                                # sorting in the same way as L_S
                                L_C          = self.time_to_fix_LDA.loc[self.BDG.bugs_dictionary[bug_id].assigned_to,
                                                                        self.BDG.bugs_dictionary[bug_id].LDA_category]
                                fixing_time_all.append(L_C)
                            # finding where the cost is minimum
                            min_cost_index = np.argmin(L_C)
                            bug_to_choose  = possible_bugs[min_cost_index]
                            dev_email      = self.BDG.bugs_dictionary[bug_to_choose].assigned_to
                            for dev_id, dev_info in self.developers_info.items():
                                if dev_info.search_by_email(dev_email):
                                    break
                            dev_info.assign_and_solve_bug(self.BDG.bugs_dictionary[bug_to_choose],
                                                          self.day_counter,
                                                          mode_       = mode_,
                                                          resolution_ = self.resolution,
                                                          T_or_P      = 'prioritization')
                            """ We keep track iff we are in testing phase """
                            self.track_and_assign(bug_to_choose, dev_info, date_formatted)
                        ##################
                        elif (self.resolution == 'priority_estimation'):
                            """Choose a bug with the maximum predicted priority
                               As the priority 5 means the highest priority, we use ``argmax'' here.
                            """
                            priority_estimated_all = []
                            for bug_id in possible_bugs:
                                bug_sum_desc = self.BDG.bugs_dictionary[bug_id].sum_desc_lemmatized
                                Test_X_Tfidf = self.tfidf_svm.transform([bug_sum_desc])
                                L_S          = self.svm_model_priority.predict(Test_X_Tfidf)[0]
                                priority_estimated_all.append(L_S)
                            max_prio_index = np.argmax(priority_estimated_all)
                            bug_to_choose  = possible_bugs[max_prio_index]
                            dev_email      = self.BDG.bugs_dictionary[bug_to_choose].assigned_to
                            for dev_id, dev_info in self.developers_info.items():
                                if dev_info.search_by_email(dev_email):
                                    break
                            dev_info.assign_and_solve_bug(self.BDG.bugs_dictionary[bug_to_choose],
                                                          self.day_counter,
                                                          mode_       = mode_,
                                                          resolution_ = self.resolution,
                                                          T_or_P      = 'prioritization')
                            """ We keep track iff we are in testing phase """
                            self.track_and_assign(bug_to_choose, dev_info, date_formatted)
                        ##################
                        elif (self.resolution == 'cost_priority_estimation'):
                            """Choose a bug with the maximum predicted priority and minimum fixing cost
                               alpha balances priority vs. fixing cost.
                            """
                            for bug_id in possible_bugs:
                                if self.BDG.bugs_dictionary[bug_id].LDA_category == None:
                                    self.predict_LDA(bug_id)
                            fixing_time_all = []
                            priority_estimated_all = []
                            for bug_id in possible_bugs:
                                # sorting in the same way as L_S
                                L_C          = self.time_to_fix_LDA.loc[self.BDG.bugs_dictionary[bug_id].assigned_to,
                                                                        self.BDG.bugs_dictionary[bug_id].LDA_category]
                                fixing_time_all.append(L_C)
                                bug_sum_desc = self.BDG.bugs_dictionary[bug_id].sum_desc_lemmatized
                                Test_X_Tfidf = self.tfidf_svm.transform([bug_sum_desc])
                                L_S          = self.svm_model_priority.predict(Test_X_Tfidf)[0]
                                priority_estimated_all.append(L_S)
                            cost_priority_array = self.Cost_Priority(priority_estimated_all, fixing_time_all)

                            # finding where the formula is maximized
                            estimation_index = np.argmax(cost_priority_array)
                            bug_to_choose    = possible_bugs[estimation_index]
                            dev_email        = self.BDG.bugs_dictionary[bug_to_choose].assigned_to
                            for dev_id, dev_info in self.developers_info.items():
                                if dev_info.search_by_email(dev_email):
                                    break
                            dev_info.assign_and_solve_bug(self.BDG.bugs_dictionary[bug_to_choose],
                                                          self.day_counter,
                                                          mode_       = mode_,
                                                          resolution_ = self.resolution,
                                                          T_or_P      = 'prioritization')
                            """ We keep track iff we are in testing phase """
                            self.track_and_assign(bug_to_choose, dev_info, date_formatted)
                        ##################
                        else:
                            raise Exception (f"The resoultion {self.resolution} is not defined.")

            elif (status_i.lower() == 'reopened'   ) :
                """ 
                Maybe the bug is assigned but not solved yet. So we cannot re-open it.
                """
                if (bug_id in self.resolved_bugs) and (self.date < self.testing_time):
                    if (bug_id not in self.update):
                        self.update.append(bug_id)
                    self.resolved_bugs[bug_id].reopen(self.BDG, self.day_counter)
                    del self.resolved_bugs[bug_id]

            elif (status_i.lower() == 'resolved'   ) or (status_i.lower() == 'closed'):
                if bug_id in self.BDG.bugs_dictionary:
                    if ((self.date < self.testing_time) or 
                        (not self.BDG.bugs_dictionary[bug_id].valid4assignment) or
                        (self.BDG.bugs_dictionary[bug_id].open[0] < self.testing_time_counter)):
                        """
                        If we are in training phase or 
                        the bug is not assigned to an active developer
                        """
                        try:
                            self.update.remove(bug_id)
                        except ValueError:
                            pass
                        if bug_id not in self.resolved_bugs:
                            self.resolved_bugs[bug_id] = self.BDG.bugs_dictionary[bug_id]
                            self.BDG.bugs_dictionary[bug_id].resolve_bug(self.day_counter)

        """ Updating whatever we assigned """
        self.filtered_date_assigned = self.filter_on_the_date(self.assigned_bug_tracker) # updating assigned bugs
        for i in range(len(self.filtered_date_assigned)):
            bug_id      = int(self.filtered_date_assigned.iloc[i].bug)
            self.time   = self.filtered_date_assigned.iloc[i].time
            detail      = self.filtered_date_assigned.iloc[i].detail
            if bug_id not in self.all_opened_bugs:
                raise Exception (f'Bug {bug_id} is solved but not opened!')
            elif (self.date < self.testing_time) and (not self.BDG.bugs_dictionary[bug_id].valid4assignment):
                """
                If we are in Waybackmachine mode or still in training phase or 
                the bug is not assigned in real world.
                """
                raise Exception ('How is it possible to assign an infeasible bug or a bug in training phase?')
            else:
                try:
                    if len(self.BDG.bugs_dictionary[bug_id].depends_on)>0:
                        self.BDG.bugs_dictionary[bug_id].ignored_dependency = 1
                    else:
                        self.BDG.bugs_dictionary[bug_id].ignored_dependency = 0
                    self.keep_track_of_resolved_bugs['disregarded_dependency_later'].append(
                        self.BDG.bugs_dictionary[bug_id].ignored_dependency)
                    self.update.remove(bug_id)
                except ValueError:
                    self.verboseprint(code = 'bug_removal_not_possible', bug_id = bug_id)
                COMMENT = True
                if bug_id not in self.resolved_bugs:
                    if not COMMENT:
                        """It is commented out as the dependency is checked in the assigning time"""
                        if len(self.BDG.bugs_dictionary[bug_id].depends_on)>0:
                            self.BDG.bugs_dictionary[bug_id].ignored_dependency = 1
                        else:
                            self.BDG.bugs_dictionary[bug_id].ignored_dependency = 0
                        self.keep_track_of_resolved_bugs['disregarded_dependency'].append(
                            self.BDG.bugs_dictionary[bug_id].ignored_dependency)
                    self.resolved_bugs[bug_id] = self.BDG.bugs_dictionary[bug_id]
                    self.BDG.bugs_dictionary[bug_id].resolve_bug(self.day_counter)
                else:
                    self.verboseprint(code = 'bug_not_found', bug_id = bug_id)
                    
        #os.system('cls') # clear screen
        self.BDG.update_degree   ()
        self.BDG.update_depth    ()
        self.BDG.update_priority ()
        self.BDG.update_severity ()
        if (self.date >= self.testing_time):
            self.keep_track_of_resolved_bugs['BDG_depth' ].append(np.mean(list(self.BDG.depth .values())))
            self.keep_track_of_resolved_bugs['BDG_degree'].append(np.mean(list(self.BDG.degree.values())))
            self.keep_track_of_resolved_bugs['date'      ].append(date_formatted)
        self.track_BDG_info ['date_open'  ].append(date_formatted)
        assert self.BDG.n_nodes == len(self.BDG.bugs_dictionary)
        self.track_BDG_info ['n_of_bugs'    ].append(self.BDG.n_nodes)
        self.track_BDG_info ['n_of_arcs'    ].append(self.BDG.n_arcs )
        self.track_BDG_info ['depth_open'   ].append(np.mean(list(self.BDG.depth .values())))
        self.track_BDG_info ['degree_open'  ].append(np.mean(list(self.BDG.degree.values())))
        self.track_BDG_info ['priority_open'].append(np.mean(list(self.BDG.priority.values())))
        self.track_BDG_info ['severity_open'].append(np.mean(list(self.BDG.severity.values())))
        self.verboseprint(code = 'print_date', date = date_formatted)
        self.update_date()