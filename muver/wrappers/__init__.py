import os
import configparser
import subprocess


PATHS = dict()
config = configparser.ConfigParser()
config.read(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '../..', 'paths.cfg'))
for key, value in config.items('paths'):
    PATHS[key] = value

experiment_directory = None

def quiet_call(call_list, stdout='muver_externals_output.txt', stderr='muver_externals_error.txt'):
    '''
    Call outside program while suppressing messages to stdout and stderr.

    stdout_file -- captures stdout (default os.devnull)
    '''
    
    if experiment_directory:
        stdout = os.path.join(experiment_directory, stdout)
        stderr = os.path.join(experiment_directory, stderr)

    with open(stdout, 'a+') as OUT,\
         open(stderr, 'a+') as ERR:
        OUT.seek(0, os.SEEK_END)
        ERR.seek(0, os.SEEK_END)
        return subprocess.call(
            call_list, stdout=OUT, stderr=ERR)
