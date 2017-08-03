# execute with: python example_processes.py
from parallel_decorators import vectorize_parallel


@vectorize_parallel(method='processes', num_procs=2, use_progressbar=True)
def foo(i):
    return i**2


@vectorize_parallel(method='processes', num_procs=2, use_progressbar=True)
def bar(i):
    if i == 10000:
        return 1/0
    else:
        return i


result = foo(range(1, 20000))
print(result[-4:])

result2 = bar(range(1, 20000))
