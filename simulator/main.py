from bin.BDG import BDG
from components.assignee import Assignee
from components.bug import Bug
from utils.functions import Functions
from utils.prerequisites import *  # Packages and these functions: isNaN, isnotNaN,
                                   # convert_string_to_array, mean_list, and string_to_time
from utils.report import Report

from simulator.sdabt import Discrete_event_simulation


#######################################
##                                   ##
##      author: Hadi Jahanshahi      ##
##     hadi.jahanshahi@ryerson.ca    ##
##          Data Science Lab         ##
##                                   ##
#######################################


parser = argparse.ArgumentParser(description='The parser will handle hyperparameters of the model')

def str2bool(v):
    if isinstance(v, bool):
       return v
    elif v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser.add_argument(
    '--project', 
    default = 'EclipseJDT',
    type    = str,
    help    = 'it can be selected from this list: [LibreOffice, Mozilla, EclipseJDT]'
)

parser.add_argument(
    '--resolution',
    default = 'Actual',
    type    = str,
    help    = 'it can be selected from this list: [Actual, DABT, RABT, CosTriage, CBR, DABT, SDABT]'
)

parser.add_argument(
    '--n_days',
    type    = int,
#	default = 6781,
    help    = 'How many days we need to train?'
)

parser.add_argument(
    '--verbose',
    default = 0,
    type    = int,
    help    = 'it can be either: [0, 1, 2, nothing, some, all]'
)

parser.add_argument(
    '--alpha_testing',
	default=False,
    type=str2bool,
    help='If want to test the model for different alpha values, we make it True'
)

parser.add_argument(
    '--part',
	default='0',
    type=str,
    help="""It is used for parallel running when we want to divide runs to multiple folds
    For instance, 1/4 says we want to fold 1 out of 4 folds we have.
    """
)

wayback_param        = parser.parse_args()
project              = wayback_param.project
verbose              = wayback_param.verbose
file_name            = wayback_param.resolution
Tfidf_vect           = None
alpha_testing        = wayback_param.alpha_testing
if wayback_param.part == '0':
    loop_range = np.arange(0, 1.01 ,0.1)
else:
    part, total_folds = [int(i) for i in wayback_param.part.split('/')]
    epsilon = 0.0001
    steps = round(11/(total_folds))/10
    start = (part-1) * steps
    if part == total_folds:
        end = 1 + epsilon
    else:
        end = (part) * (steps - epsilon)
    loop_range = np.arange(start,end,0.1)
part                 = wayback_param.part
[bug_evolutionary_db, bug_info_db, list_of_developers, 
 time_to_fix_LDA, SVM_model, feasible_bugs_actual, embeddings] = Functions.read_files(project)

simul_triage         = Discrete_event_simulation(bug_evolutionary_db, bug_info_db, list_of_developers,
												 time_to_fix_LDA, SVM_model, Tfidf_vect, project, feasible_bugs_actual,
												 embeddings, resolution = file_name, verbose=verbose)
stop_date            = len(pd.date_range(start=simul_triage.bug_evolutionary_db.time.min().date(), end='31/12/2019'))
stop_training        = len(pd.date_range(start=simul_triage.bug_evolutionary_db.time.min().date(), end='31/12/2017'))#training
stop_testing         = len(pd.date_range(start='01/01/2018', end='31/12/2019')) # whole testing period

if alpha_testing == False:
    try:
        for i in tqdm(range(stop_date), desc="simulating days", position=0, leave=True):
            simul_triage.triage()
            #if (simul_Actual_time_half.date > simul_Actual_time_half.testing_time):
        
		with open(f'dat/{project}/{file_name}_{i}_triage.pickle', 'wb') as file:
			pickle.dump(simul_triage, file) # use `pickle.load` to do the reverse
    except Exception as e:
        # winsound.PlaySound("*", winsound.SND_ALIAS) #It only works on Windows not Linux
        raise e
        
elif alpha_testing == True:
    """ Testing the tradeoff for different alpha values."""
    for i in tqdm(range(stop_training), desc="training days", position=0, leave=True):
		simul_triage.triage()
    """ Keep a backup of the model at the end of the training """
	with open(f'dat/{project}/{file_name}_{i}_triage.pickle', 'wb') as file:
		pickle.dump(simul_triage, file) # use `pickle.load` to do the reverse
    for alpha in loop_range:
		simul_triage_copy               = copy.deepcopy(simul_triage)
		simul_triage_copy.alpha         = alpha
		simul_triage_copy.start_model_updates_for_testing = time.time()
        for i in tqdm(range(stop_testing), desc=f"testing days for alpha={round(alpha,1)}", position=0, leave=True):
			simul_triage_copy.triage()
		with open(f'dat/{project}/{file_name}_{i}_{round(alpha,1)}_triage.pickle', 'wb') as file:
			pickle.dump(simul_triage_copy, file) # use `pickle.load` to do the reverse

else:
    raise Exception('Double-Check str2boolean function of alpha_testing variable')

# python3.7 simulator/main.py --project=Mozilla --resolution=Actual --n_days=7511 
# python3.7 simulator/main.py --project=EclipseJDT --resolution=Actual --n_days=6644 
# python3.7 simulator/main.py --project=LibreOffice --resolution=Actual --n_days=3438 
# python3.7 simulator/main.py --project=Mozilla --resolution=SDABT --n_days=7511 
# python3.7 simulator/main.py --project=EclipseJDT --resolution=SDABT --n_days=6644 
# python3.7 simulator/main.py --project=LibreOffice --resolution=SDABT --n_days=3438 
# python3.7 simulator/main.py --project=Mozilla --resolution=DABT --n_days=7511 
# python3.7 simulator/main.py --project=EclipseJDT --resolution=DABT --n_days=6644 
# python3.7 simulator/main.py --project=LibreOffice --resolution=DABT --n_days=3438 
# python3.7 simulator/main.py --project=Mozilla --resolution=Random --n_days=7511 
# python3.7 simulator/main.py --project=EclipseJDT --resolution=Random --n_days=6644 
# python3.7 simulator/main.py --project=LibreOffice --resolution=Random --n_days=3438 
# python3.7 simulator/main.py --project=Mozilla --resolution=RABT --n_days=7511 
# python3.7 simulator/main.py --project=EclipseJDT --resolution=RABT --n_days=6644 
# python3.7 simulator/main.py --project=LibreOffice --resolution=RABT --n_days=3438 
# python3.7 simulator/main.py --project=Mozilla --resolution=CosTriage --n_days=7511 
# python3.7 simulator/main.py --project=EclipseJDT --resolution=CosTriage --n_days=6644 
# python3.7 simulator/main.py --project=LibreOffice --resolution=CosTriage --n_days=3438 
# python3.7 simulator/main.py --project=Mozilla --resolution=CBR --n_days=7511 
# python3.7 simulator/main.py --project=EclipseJDT --resolution=CBR --n_days=6644 
# python3.7 simulator/main.py --project=LibreOffice --resolution=CBR --n_days=3438 
# python3.7 simulator/main.py --project=Mozilla --resolution=SDABT --n_days=7511 --alpha_testing=yes --part=1/4
# python3.7 simulator/main.py --project=Mozilla --resolution=SDABT --n_days=7511 --alpha_testing=yes --part=2/4
# python3.7 simulator/main.py --project=Mozilla --resolution=SDABT --n_days=7511 --alpha_testing=yes --part=3/4
# python3.7 simulator/main.py --project=Mozilla --resolution=SDABT --n_days=7511 --alpha_testing=yes --part=4/4
# python3.7 simulator/main.py --project=EclipseJDT --resolution=SDABT --n_days=6644 --alpha_testing=yes --part=1/2
# python3.7 simulator/main.py --project=EclipseJDT --resolution=SDABT --n_days=6644 --alpha_testing=yes --part=2/2
# python3.7 simulator/main.py --project=LibreOffice --resolution=SDABT --n_days=3438 --alpha_testing=yes --part=1/2
# python3.7 simulator/main.py --project=LibreOffice --resolution=SDABT --n_days=3438 --alpha_testing=yes --part=2/2
# To find site-packages location: python -m site
# creating example.pth there 
# PATH="$PATH:/home/hadi/Hadi_progress-1/Scripts/Bugzilla_Mining/OOP/