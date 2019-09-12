import os


def create_dirs(dirs):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    :return exit_code: 0:success -1:failed
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)


def clean_dir(dir, prefix):
    """
    Clean the given dir from every file starting with prefix
    :param dir:
    :param filename:
    :return:
    """
    try:
        for filename in os.listdir(dir):
            if filename.startswith(prefix):
                os.remove(os.path.join(dir, filename))
    except Exception as err:
        print("Cleaning directory error: {0}".format(err))
