## This file is only used for debugging mode
## it explore different bugs' and developers' characteristic to see where the error comes from.
class Debugging:
    @staticmethod
    def bug_info (simulator, bug_id):
        return simulator.BDG.bugs_dictionary[bug_id].__dict__
    def bug_resolved (simulator, bug_id):
        return simulator.resolved_bugs[bug_id].__dict__
    def bug_info_db (simulator, bug_id):
        return simulator.bug_info_db.loc[bug_id]
    def bug_info_evol (simulator, bug_id):
        return simulator.bug_evolutionary_db[
            simulator.bug_evolutionary_db.bug == bug_id]
    def dev_info (simulator, dev_info):
        if str(dev_info).isdigit():
            return simulator.developers_info[dev_info].__dict__
        else:
            for dev_id, dev_inf in simulator.developers_info.items():
                if dev_inf.search_by_email(dev_info):
                    return simulator.developers_info[dev_id].__dict__