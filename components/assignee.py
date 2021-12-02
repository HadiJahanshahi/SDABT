import numpy as np
from components.bug import Bug

class Assignee:
    n_of_dev = 0
    def __init__(self, email, id_, name, LDA_experience, time_limit_orig):
        self.email               = email.lower()
        self.n                   = Assignee.n_of_dev
        self.id_                 = id_
        if name != name:
            name = ''
        self.name                = name.lower()
        self.bugs                = []
        self.working_time        = []
        self.components          = []              # in training mode
        self.components_tracking = []              # in testing mode
        self.accuracy            = []
        self.accuracyT           = []
        self.simultaneous_cap    = 0               # capacity of solving bugs simultaneously
        self.LDA_experience      = LDA_experience
        self.n_assigned_bugs     = 0
        self.n_current_assigned  = []              # Keeping track of the capacity of a developer (# of simultaneous jobs)
        self.filled_capacity     = [0]             # It tracks how much of each slot is occupied
        self.time_limit_orig     = time_limit_orig # it will never change (Time horizon)
#        self.time_limit          = time_limit_orig  # it will change (It is for the older version (DABT) with single slot)
        self.time_limit          = [time_limit_orig] # The complement of filled_capacity, tracking how much of each slot is free
        self.schedule            = []              #T_{jt}^d (binary schedule)
        if len(self.filled_capacity)>0:
            self.update_schedule()                 # at the beginning, we create developers without considering the resolution
        Assignee.add_dev()
        
    def calculating_simultaneous_capcity(self, strategy:str):
        """
        It finds the capcity of a developer based on the history of the assignments.
        Then, it resets filled_capacity, time_limit, and schedule for the testing phase.
        
        Args:
            strategy (str): [It can be either "max" or "IQR"]
        """
        if strategy.lower() == "max":
            """
            We consider the busiest day as the limit for the capacity.
            """
            self.simultaneous_cap = max(self.n_current_assigned) # maximum simultaneous jobs
        elif strategy.lower() == "iqr":
            """
            We consider upper bound for the noises (i.e., Q3 + (1.5*IQR)) as the capacity limit.
            """
            only_working_days     = np.array(self.n_current_assigned)[np.array(self.n_current_assigned)>0]
            Q1, Q3                = np.quantile(only_working_days, [0.25,0.75])
            IQR                   = Q3 - Q1
            self.simultaneous_cap = int(Q3 + (1.5*IQR))
        else:
            raise Exception(f"Startegy can be either Max or IQR. You passed {strategy}.")
        self.filled_capacity  = [0 for i in range(self.simultaneous_cap)]
        self.time_limit       = [self.time_limit_orig for i in range(self.simultaneous_cap)]
        self.update_schedule()

    def assign_bug(self, bug:Bug, time_:int, mode_:str):
        """[Assign a bug to a developer]
        Older and simplere version for the training period.
        Args:
            bug   (Bug)    : [Which bug to assign]
            time_ (int)    : [Assignment date]
            mode_ (str)    : [Whether it is tracking or not]
        """
        assert bug not in self.bugs
        self.bugs.append(bug)
        self.components.append(bug.component) 
        self.n_assigned_bugs    += 1
        bug.assigned_to_rec      = self.email
        bug.assigned_time.append(time_)

    def which_slot_to_assign(self):
        """
        We need to assign the new bug to the slot that has the max available times.
        """
        which_slot = self.time_limit.index(max(self.time_limit))
        what_time  = self.time_limit_orig - max(self.time_limit)
        return which_slot, what_time

    def assign_and_solve_bug(self, bug:Bug, time_:int, mode_:str, resolution_:str,
                             T_or_P:str = 'triage', which_slot:int=None, what_time:int=None):
        """[Assign a bug to a developer]

        Args:
            bug (Bug)        : [Which bug to assign]
            time_ (int)      : [Assignment date]
            mode_ (str)      : [Whether it is tracking or not]
            resolution_ (str): [Whether it is Actual assignment or other bug triaging policies.]
            T_or_P (str)     : [Whether it is a Triage or Prioritization task]
            which_slot(int)  : [To which slot of the developer we should assign]
            what_time(int)   : [To which day of a slot we should assign the bug]

            During the not_tracking mode (i.e., training time), we do not know how many
            slots a developer has. Therefore, we assume each developer has one slot and
            update it for the testing phase (i.e., tracking mode).
        """
        if (which_slot == None) and (mode_ == "tracking"):
            which_slot, what_time = self.which_slot_to_assign()
        elif (which_slot == None) and (mode_ == "not_tracking"):
            which_slot, what_time = 0, "-"
        if mode_ == "tracking":
            if bug.component in self.components:
                # only having the same COMPONENT is enough to say the assignment is accurate.
                bug.assignment_accuracy = 1
                self.accuracy.append(1)
            else:
                bug.assignment_accuracy = 0
                self.accuracy.append(0)
            if self.email.lower() == bug.assigned_to.lower():
                # It has to be assigned to the same DEVELOPER to be accurate.
                bug.assignment_accuracyT = 1
                self.accuracyT.append(1)
            else:
                bug.assignment_accuracyT = 0
                self.accuracyT.append(0)
        if bug not in self.bugs:
            self.n_assigned_bugs    += 1
            self.bugs.append(bug)
            # time to solve based on LDA table
        # if ((resolution_ == 'actual') or (T_or_P.lower() == 'prioritization')) and (mode_ == "not_tracking"):
        if (mode_ == "not_tracking"):
            """
            previously, for bug fixing, we considered the real time, but now we use LDA time. 
            so, we bypass this in the testing phase.
            """
            solving_time_ = bug.time_to_solve
        else:
            try:
                solving_time_ = int(self.LDA_experience [bug.LDA_category])
            except:
                print('self.LDA_experience', self.LDA_experience)
                print('self.email', self.email)
                print('bug.LDA_category', bug.LDA_category)
                print('bug.idx', bug.idx)
                raise Exception('Error for the above information')
        
        if (resolution_ == 'sdabt') and (mode_ == "tracking"):
            """
            If we are in testing mode and use S-DABT, we now exactly when to assign a bug.
            """
            starting_time = what_time
        else:
            """
            Otherwise, we just check the occupied schedule of a developer and assign
            the new bug after that.
            """
            starting_time = self.filled_capacity[which_slot]
        bug.solving_time_after_simulation_accumulated         = starting_time + solving_time_
        bug.solving_time_after_simulation                     = solving_time_
        assert bug.solving_time_after_simulation             != None
        assert bug.solving_time_after_simulation_accumulated != None
        if mode_ == 'tracking':
 #           self.time_limit      -= solving_time_ # It is for the older version (DABT)
            self.working_time.append(solving_time_)
            if (resolution_ != 'sdabt'):
                """
                In S-DABT, we have a different stratgey to find what time to assign a bug.
                So, we cannot updated filled_capacity and time_limit in this way. 
                We update it later in the day, when we move to the next day.
                """
                self.filled_capacity[which_slot] += solving_time_
                self.time_limit[which_slot]      -= solving_time_
            else:
                """
                In S-DABT, schedule is what we update more often.
                We add 0 (busy) for the days a bug takes to be solved.
                """
                assert all(self.schedule[which_slot][starting_time:(starting_time + solving_time_)]) == 1
                self.schedule[which_slot][starting_time:(starting_time + solving_time_)] = [0 for i in range(solving_time_)]

            self.components_tracking.append(bug.component)
        else:
            """we update components only in training phase"""
            self.components.append(bug.component)
        bug.assigned_to_rec   = self.email
        bug.assigned_time.append(time_)
    
    def update_schedule(self, resolution= "actual"):
        """
        1 means an available slot and 0 means an unavilable slot.
        Which resolution to use for updating the schedule
        """
        if resolution != "sdabt":
            self.schedule = []
            for filled_capacity in (self.filled_capacity):
                try:
                    self.schedule.append([0 if i<filled_capacity else 1 for i in range(self.time_limit_orig)])
                except:
                    pass
        else: #S-DABT
            """" For S-DABT, IP model updates schedule. No need to do it again."""
            pass
                
    def search_by_email(self, email_):
        if self.email == email_.lower():
            return True
        return False

    def search_by_id(self, id_):
        if self.id_ == id_:
            return True
        return False

    def increase_time_limit(self, resolution = "actual"):
        """ at the end of the day, we need to add to the time_limit of each developer by 1 """
        # it cannot exceed time_limit_orig
        # self.time_limit = min (self.time_limit+1, self.time_limit_orig) (older version with single slot)
        if resolution != "sdabt":
            self.time_limit      = [(min(t_limit+1, self.time_limit_orig)) for t_limit in self.time_limit]
            self.filled_capacity = [(self.time_limit_orig - t_limit) for t_limit in self.time_limit]
            self.update_schedule(resolution = resolution)
            """" The time limit of each slot cannot exceed the project horizon (time_limit_orig) """
        else: #S-DABT
            self.schedule        = [schedule[1:]+[1] for schedule in self.schedule]
            """
            Free up one day and removed the passed day. We directly update schedule in S-DABT
            There is no need to change time_limit or filled_capacity.
            """
            
    @classmethod
    def add_dev(cls): #class method not for an object
        cls.n_of_dev += 1
