# SDABT
S-DABT: Schedule and Dependency-Aware Bug Triage in Open-Source Bug Tracking Systems

**Authors**: Hadi Jahanshahi*, Mucahit Cevik, and Ayse Basar

\* Author to whom correspondence should be addressed. email: hadi . jahanshahi [at] reyerson.ca

## Folders and their contents 

### bin
It includes the bug dependency graph (`BDG`), defined in the paper. 

It includes graph operations, e.g., adding or removing arcs and nodes, together with graph-related updates, e.g., updating depth, degree, severity, and priority of the bugs in the BDG.

### components
It includes two main classes: **developers** and **bugs**. 
* `assignee.py` has the `Assignee` class which includes the information of the developers and track the assigned bugs to them and the accuracy of those assignments. It also keeps their LDA experience and the time limit $L$ of them.
* `bug.py` has the `Bug` class which includes all the essential information of each bugs, including ID, severity, priority, depth, degree, status, summary, description, fixing time, and so on. It has its methods to track the assigned developer and assignment time, compute the accuracy of the assignment, check the validity of the bugs for assignment based on preprocessing steps in the paper, update its blocking information, and change its status to fixed or reopenned. 


### dat
It includes all the datasets used in the paper. The datasets are related to the extracted bugs from three software projects, Mozilla, LibreOffice, and EclipseJDT.

### imgs
It includes the images used in the paper in a vector format.

### simulator
This folder contains two important files: `main.py` and `wayback.py`.

*  `wayback.py` codes the process of the Wayback machine and its elements. The main variables are
  *  `keep_track_of_resolved_bugs` which keeps all the info related to the resolved bugs during the testing phase.
  *  `track_BDG_info` keeps track of the BDG during the life span of the project.
  *  `verbose` defines how to print the output during the running time, e.g.. `nothing`, `some`, or `all` the information should be printed.
  It has also some important methods, including
  * `acceptable_solving_time` which determines the acceptable solving time based on the IQR.
  * `possible_developers` which finds the list of feasible developers at the end of the training phase.
  * `fixing_time_calculation` which uses bug info and evolutionary database to calculate fixing time according to the Costriage paper.
  * `track_and_assign` which assigns the bugs to proper developers and tracks the info of the assigned/fixed bug.
  * `triage` module to apply triage algorithms. Researchers can manipulate this method and add their own triage algorithms to the Wayback Machine. `DABT`, `SDABT`, `RABT`, `CosTriage`, `CBR`, `Actual` and `Random` triage are already implemented.

* `main.py` is needed to run the Wayback Machine. 
To run the code, a sample command might be as follows. 

```python
python simulator/main.py --project=LibreOffice --resolution=SDABT --n_days=3438  --verbose=0
```

If you want to do parallel runs for testing the hyperparameter of the model $\alpha$, you need to run as follows:
```python
python simulator/main.py --project=EclipseJDT --resolution=SDABT --n_days=6644 --alpha_testing=yes --part=1/2
python simulator/main.py --project=EclipseJDT --resolution=SDABT --n_days=6644 --alpha_testing=yes --part=2/2
```

Our model is compatible with Python 3.7 and higher.

Regarding the options available for the `main.py` file:
  * `--resolution` defines the **strategy/algorithm** to take
  * `project` can be `Mozilla`, `LibreOffice`, or `EclipseJDT`. A user can also extract and add their own ITS database. 
  * `--n_days` defines the number of days from the beginning to the end of the lifespan. Based on our database, it should be 3438 days for LibreOffice,  7511 days for Mozilla, and 6644 days for EclipseJDT.
  * `--verbose` indicates how to print the output and can be either: ```[0, 1, 2, nothing, some, all]```.
  * `--alpha_testing` is used when you want to test parameter $\alpha$ of the model. It can be: ```[yes, y, True, no, n, False]```.
  * `--part` is used for parallel running. It is specially used when alpha_testing is True. It should be in the format of #/#. For instance, 1/2 says that run the model for the first half of the options for alpha, (e.g., 0, 0.1, 0.2, 0.3, 0.4, 0.5) and 2/2 will do it for the second half (e.g., 0.6, 0.7, 0.8, 0.9 ,1.0)

More details on the simulator are commented on in the files.

### utils
It contains `attention_decoder` for the DeepTriage strategy. `debugger` to search over the variables in case of a bug. `functions` which includes useful, fundamental functions. `prerequisites` including all the packages needed to run the Wayback Machine. `release_dates` that holds the release dates of the projects during their testing phase. If a user wants to add a new project, they have to manually add the release dates here. `report` gives a full report of the Wayback Machine outputs if needed.


## Prerequisites:
 * networkx 
 * random
 * tqdm
 * time
 * collections
 * statistics
 * numpy
 * pandas
 * datetime
 * os
 * ast
 * json
 * pickle
 * copy
 * matplotlib
 * gensim 
 * nltk 
 * sklearn 
 * tensorflow
 * gurobipy 
 * plotly

____________
The output of each run will be saved in the `dat` folder automatically. 


# The importance of the problem with a simple example

![Developers' schedule](https://raw.githubusercontent.com/HadiJahanshahi/SDABT/main/imgs/Schedule-slot-developers.png)

Any questions? Please do not hesitate to contact me: hadi . jahanshahi [at] ryerson.ca
