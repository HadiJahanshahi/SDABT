from utils.prerequisites import string_to_time
def release_dates(project_name):
    """Release dates of different projects

    Args:
        project_name ([str]): Name of the project you are interested in

    Raises:
        Exception: [When the name of the project does not exist in the database, it returns an exception.]

    Returns:
        [list]: [List of release dates of the given project]
    """
    if project_name == 'EclipseJDT':
        release_dates               = [string_to_time("2018-03-18 00:00:00", '%Y-%m-%d %H:%M:%S'), #4.7.3
                                       string_to_time("2018-06-19 00:00:00", '%Y-%m-%d %H:%M:%S'), #4.8
                                       string_to_time("2018-09-19 00:00:00", '%Y-%m-%d %H:%M:%S'), #4.8.1
                                       string_to_time("2018-12-19 00:00:00", '%Y-%m-%d %H:%M:%S'), #4.9
                                       string_to_time("2019-03-20 00:00:00", '%Y-%m-%d %H:%M:%S'), #4.10
                                       string_to_time("2019-06-19 00:00:00", '%Y-%m-%d %H:%M:%S'), #4.12
                                       string_to_time("2019-09-18 00:00:00", '%Y-%m-%d %H:%M:%S'), #4.13
                                       string_to_time("2019-12-18 00:00:00", '%Y-%m-%d %H:%M:%S')] #4.14
    elif project_name == 'Mozilla':
        release_dates               = [string_to_time("2018-01-23 00:00:00", '%Y-%m-%d %H:%M:%S'), #Firefox58
                                       string_to_time("2018-03-13 00:00:00", '%Y-%m-%d %H:%M:%S'), #Firefox59
                                       string_to_time("2018-05-09 00:00:00", '%Y-%m-%d %H:%M:%S'), #Firefox60
                                       string_to_time("2018-06-26 00:00:00", '%Y-%m-%d %H:%M:%S'), #Firefox61
                                       string_to_time("2018-09-05 00:00:00", '%Y-%m-%d %H:%M:%S'), #Firefox62
                                       string_to_time("2018-10-23 00:00:00", '%Y-%m-%d %H:%M:%S'), #Firefox63
                                       string_to_time("2018-12-11 00:00:00", '%Y-%m-%d %H:%M:%S'), #Firefox64
                                       string_to_time("2019-01-29 00:00:00", '%Y-%m-%d %H:%M:%S'), #Firefox65
                                       string_to_time("2019-03-19 00:00:00", '%Y-%m-%d %H:%M:%S'), #Firefox66
                                       string_to_time("2019-05-21 00:00:00", '%Y-%m-%d %H:%M:%S'), #Firefox67
                                       string_to_time("2019-07-09 00:00:00", '%Y-%m-%d %H:%M:%S'), #Firefox68
                                       string_to_time("2019-09-03 00:00:00", '%Y-%m-%d %H:%M:%S'), #Firefox69
                                       string_to_time("2019-10-22 00:00:00", '%Y-%m-%d %H:%M:%S'), #Firefox70
                                       string_to_time("2019-12-03 00:00:00", '%Y-%m-%d %H:%M:%S')] #Firefox71           
    elif project_name == 'LibreOffice':
        release_dates               = [string_to_time("2018-01-29 00:00:00", '%Y-%m-%d %H:%M:%S'), #6.0.0
                                       string_to_time("2018-02-05 00:00:00", '%Y-%m-%d %H:%M:%S'), #6.0.1
                                       string_to_time("2018-02-26 00:00:00", '%Y-%m-%d %H:%M:%S'), #6.0.2
                                       string_to_time("2018-04-02 00:00:00", '%Y-%m-%d %H:%M:%S'), #6.0.3
                                       string_to_time("2018-05-07 00:00:00", '%Y-%m-%d %H:%M:%S'), #6.0.4
                                       string_to_time("2018-06-18 00:00:00", '%Y-%m-%d %H:%M:%S'), #6.0.5
                                       string_to_time("2018-07-30 00:00:00", '%Y-%m-%d %H:%M:%S'), #6.0.6
                                       string_to_time("2018-08-06 00:00:00", '%Y-%m-%d %H:%M:%S'), #6.1.0
                                       string_to_time("2018-09-16 00:00:00", '%Y-%m-%d %H:%M:%S'), #6.1.1
                                       string_to_time("2018-09-24 00:00:00", '%Y-%m-%d %H:%M:%S'), #6.1.2
                                       string_to_time("2018-10-29 00:00:00", '%Y-%m-%d %H:%M:%S'), #6.1.3
                                       string_to_time("2018-12-17 00:00:00", '%Y-%m-%d %H:%M:%S'), #6.1.4
                                       string_to_time("2019-02-04 00:00:00", '%Y-%m-%d %H:%M:%S'), #6.2.0
                                       string_to_time("2019-02-25 00:00:00", '%Y-%m-%d %H:%M:%S'), #6.2.1
                                       string_to_time("2019-03-18 00:00:00", '%Y-%m-%d %H:%M:%S'), #6.2.2
                                       string_to_time("2019-04-15 00:00:00", '%Y-%m-%d %H:%M:%S'), #6.2.3
                                       string_to_time("2019-05-20 00:00:00", '%Y-%m-%d %H:%M:%S'), #6.2.4
                                       string_to_time("2019-07-01 00:00:00", '%Y-%m-%d %H:%M:%S'), #6.2.5
                                       string_to_time("2019-08-05 00:00:00", '%Y-%m-%d %H:%M:%S'), #6.3.0
                                       string_to_time("2019-08-26 00:00:00", '%Y-%m-%d %H:%M:%S'), #6.3.1
                                       string_to_time("2019-09-23 00:00:00", '%Y-%m-%d %H:%M:%S'), #6.3.2
                                       string_to_time("2019-10-28 00:00:00", '%Y-%m-%d %H:%M:%S'), #6.3.3
                                       string_to_time("2019-12-09 00:00:00", '%Y-%m-%d %H:%M:%S')] #6.3.4
    else:
        raise Exception ('Make sure you have added the new project release dates here.')
    return release_dates