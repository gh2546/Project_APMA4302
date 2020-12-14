# Import modules
import os
import sys
import shutil
import subprocess
import time
import re
from datetime import date
from datetime import datetime

# Specifying the number of processors and course, result directory
num_procs = [6, 5, 4, 3, 2, 1]
path_dir = "../KLexpansion_results/"

code_dir = "../KLexpansion"
path_dir_list = []
exect_command = []

for nprocs in num_procs:
    path_dir_list.append(path_dir + '{0}'.format(nprocs))
    exect_command.append("mpirun -n {0} klexpansion > out.log".format(nprocs))


def purge(dir, pattern):
    for f in os.listdir(dir):
        if re.search(pattern, f):
            os.remove(os.path.join(dir, f))

for i in range(len(exect_command)):
    os.chdir(code_dir)
    print("***********************************************************************")
    print("Launching new job. \n")
    print("Code is stored in the dir:{}".format(code_dir))
    print("Command executed on the shell is: {0} ".format(exect_command[i]))
    print("Job has been launched on {0}.".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    p = subprocess.Popen([exect_command[i]], shell = True)
    p.wait()
    print("Job has been completed on {0} and output has been stored in out.log file.".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    print("Copying the data from code directory:{0} to result directory:{1}".format(code_dir, path_dir_list[i]))
    shutil.copytree(code_dir, path_dir_list[i])
    print("Cleaning the code directory and preparing it for new job.")
    purge(code_dir, "Solution")

    print("Putting the scheduler on sleep for 10 seconds.")
    print("***********************************************************************")
    time.sleep(10)
