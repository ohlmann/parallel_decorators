# execute with: mpiexec -np 4 python example_mpi.py
from parallel_decorators import vectorize_parallel, is_master
from time import sleep
import random


@vectorize_parallel(method='MPI', use_progressbar=True, label='computation')
def foo(i):
    sleep(0.1+random.random()*0.5)
    return i**2


@vectorize_parallel(method='MPI', use_progressbar=True, label='computation')
def bar(i):
    if i == 10000:
        return 1/0
    else:
        return i


result = foo(range(1, 100))

if is_master():
    print(result[-4:])

result2 = bar(range(1, 20000))
