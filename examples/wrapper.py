# execute with: mpiexec -np 4 python example_mpi.py
from parallel_decorators import vectorize_parallel, is_master
import os
import subprocess


# define function for one run
@vectorize_parallel(method='MPI')
def run(i):
    print("Doing run {:d}".format(i))
    # create and switch to run dir
    directory = 'run{:04d}'.format(i)
    if not os.path.exists(directory):
        os.mkdir(directory)
    os.chdir(directory)
    # redirect standard output
    logfile = open("log.out", 'w')
    subprocess.call(["ls", "-l"], stdout=logfile)
    # go back
    os.chdir("../")

# execute all the runs
number_runs = 10
run(range(number_runs))

if is_master():
    print("Run finished.")
