from utils.prerequisites import * # Packages and these functions: isNaN, isnotNaN, 
                                # convert_string_to_array, mean_list, and string_to_time
class Report:
    @staticmethod
    def report_everything(simulator, project, file_name, alpha = "", save_=True):
        n_bugs            = len(simulator.keep_track_of_resolved_bugs["assigned_bugs"])
        print(f"** The number of bugs resolved {n_bugs}")
        dev_info = np.unique(simulator.keep_track_of_resolved_bugs["assigned_developers"], return_counts=True)
        print(f"** The standard deviation of # of solved bugs is {dev_info[1].std().round(1)}, "
             f"and its average is {dev_info[1].mean().round(1)}.")
        avg_fixing_time   = np.array(simulator.keep_track_of_resolved_bugs['fixing_time_of_bugs']
                                     ).mean().round(1)
        print(f"** The avarege fixing time is {avg_fixing_time}.")
        accuracy          = np.array(simulator.keep_track_of_resolved_bugs['accuracy_of_assignment']).mean().round(3)
        print(f"** The accuracy of assignment is {accuracy*100}%.")
        accuracyT         = np.array(simulator.keep_track_of_resolved_bugs['accuracy_of_assignmentT']).mean().round(3)
        print(f"** The extreme accuracy of assignment is {accuracyT*100}%.")
        n_overdue_bugs    = len(np.array(simulator.keep_track_of_resolved_bugs['overdue_bugs']))
        n_unresolved_bugs = len(np.array(simulator.keep_track_of_resolved_bugs['unresolved_bugs']))
        perc_overdue_all  = round((n_overdue_bugs+n_unresolved_bugs)/n_bugs*100, 1)
        print(f"** The percentage of overdue bugs is {perc_overdue_all}%")
        perc_ignored_dep  = np.array(simulator.keep_track_of_resolved_bugs['disregarded_dependency']).mean().round(3)
        print(f"** The percentage of ignored dependency is {perc_ignored_dep*100}%")
        mean_depth        = np.mean(simulator.keep_track_of_resolved_bugs['BDG_depth'])
        mean_degree       =  np.mean(simulator.keep_track_of_resolved_bugs['BDG_degree'])
        print(f'** The average depth of the graph is {round(mean_depth,2)}.')
        print(f'** The average degree of the graph is {round(mean_degree,2)}.')
        print('#################')
        print(project, file_name)
        print(f'n_assigned={n_bugs}, n_assigned_dev={len(dev_info[1])}, mean|std={dev_info[1].mean().round(1)}|'
              f'{dev_info[1].std().round(1)}, mean_fixing={avg_fixing_time},\n overdue={perc_overdue_all}, '
              f'accuracy{accuracy*100}, ignored_BDG={perc_ignored_dep*100}, average depth={round(mean_depth,2)}, '
              f'average degree={round(mean_degree,2)}')
        print('#################')
        # save it
        if save_:
            with open(f'dat/{project}/{file_name}{alpha}.txt', 'w') as file:
                file.write(json.dumps(simulator.keep_track_of_resolved_bugs)) # use `json.loads` to do the reverse
            with open(f'dat/{project}/{file_name}{alpha}.pickle', 'wb') as file2:
                pickle.dump(simulator, file2) # use pickle.load(open(f'dat/{project}/{file_name}.pickle', "rb"))