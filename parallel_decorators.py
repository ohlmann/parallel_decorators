# coding: utf-8
"""This module contains decorators for parallel vectorization of functions.

The decorators vectorize the function over the first argument and return a
list of the returned results of the functions.
The main decorator is vectorize_parallel that supports parallelization via the
multiprocessing module or via MPI.


Example for multiprocessing:

>>> from parallel_decorators import vectorize_parallel
>>> @vectorize_parallel(method='processes', num_procs=2)
... def power(x, y):
...     return x**y
>>> result = power(range(5), 3)
[0, 1, 8, 27, 64]


Example for MPI:

>>> from parallel_decorators import vectorize_parallel, is_master
>>> @vectorize_parallel(method='MPI')
... def power(x, y):
...     return x**y
>>> # computation in parallel here
>>> result = power(range(5), 3)
[0, 1, 8, 27, 64]
>>> if is_master();
...    # use results on master core
...    print(result)

Then start script with
$ mpiexec -np <num> python script.py
"""
from functools import wraps
import traceback


def is_master():
    """return True if current rank is master"""
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    return rank == 0


def is_iterable(xs):
    """returns True if xs is an iterable"""
    # try to get iterator; if error occurs, xs is not iterable
    try:
        iter(xs)
    except TypeError:
        return False
    return True


def vectorize(f):
    """function wrapper that vectorizes f over the first argument
    if the first argument is an iterable"""
    @wraps(f)
    def newfun(xs, *args, **kwargs):
        if not is_iterable(xs):
            # no iteration, simply call function
            return f(xs, *args, **kwargs)
        # gather results in a list
        result = []
        for x in xs:
            result.append(f(x, *args, **kwargs))
        return result
    return newfun


def vectorize_queue(num_procs=2, use_progressbar=False):
    """function wrapper that vectorizes f over the first argument
    if the first argument is an iterable.
    This function wrapper uses the multiprocessing module to implement
    parallelism for the vectorization.

    Usage:
    >>> @vectorize_queue(4)
    ... def power(x, y):
    ...    return x**y
    >>> power(4, 3)
    64
    >>> power(range(5), 3)
    [0, 1, 8, 27, 64]
    """
    if use_progressbar:
        try:
            from progressbar import Bar, AdaptiveETA, Percentage, ProgressBar,\
                    FormatLabel
        except ModuleNotFoundError:
            print("Progressbar requested, but module progressbar not found."
                  " Disabling progressbar.")
            use_progressbar = False

    def decorator(f):
        """the decorator function we return"""
        @wraps(f)
        def newfun(xs, *args, **kwargs):
            """the function that replaces the wrapped function"""
            if not is_iterable(xs):
                # no iteration, simply call function
                return f(xs, *args, **kwargs)

            from multiprocessing import Process, Queue

            if use_progressbar:
                widgets = [FormatLabel(f.__name__), ' ', Percentage(),
                           Bar(), AdaptiveETA()]
                pbar = ProgressBar(widgets=widgets, maxval=len(xs))
                pbar.start()

            task_queue = Queue()
            done_queue = Queue()

            # fill tasks, first argument in tuple for ordering
            for i, x in enumerate(xs):
                task_queue.put((i, x))

            # define worker function
            # this definition is inherited by all processes
            # -> important that f, args and kwargs are known by childs
            def worker(in_queue, out_queue):
                for i, x in iter(in_queue.get, 'STOP'):
                    try:
                        res = f(x, *args, **kwargs)
                        out_queue.put((i, res, None))
                    except Exception as e:
                        print("Caught exception in parallel vectorized "
                              "function:")
                        # print out traceback
                        traceback.print_exc()
                        print()
                        out_queue.put((i, None, e))

            # start workers
            for i in range(num_procs):
                Process(target=worker, args=(task_queue, done_queue)).start()

            # get results, ordering done by first argument
            result = [None] * len(xs)
            errors = []
            for i in range(len(xs)):
                j, res, e = done_queue.get()
                result[j] = res
                # caught exception?
                if e is not None:
                    errors.append(e)
                if use_progressbar:
                    pbar.update(i)

            # stop workers
            for i in range(num_procs):
                task_queue.put('STOP')

            if use_progressbar:
                pbar.finish()

            # error ocurred?
            if len(errors) > 0:
                print("Caught at least one error during execution:")
                raise errors[0]

            return result
        return newfun
    return decorator


def vectorize_mpi(f):
    """function wrapper that vectorizes f over the first argument
    if the first argument is an iterable.
    This function wrapper uses the mpi4py module to implement
    parallelism for the vectorization.

    Usage:

    >>> @vectorize_mpi
    ... def power(x, y):
    ...    return x**y
    >>> # computation in parallel here
    >>> result = power(range(5), 3)
    [0, 1, 8, 27, 64]
    >>> if is_master();
    ...    # use results on master core
    ...    print(result)

    then start script with
    $ mpiexec -np <num> python script.py
    """
    @wraps(f)
    def newfun(xs, *args, **kwargs):
        """the function that replaces the wrapped function"""
        if not is_iterable(xs):
            # no iteration, simply call function
            return f(xs, *args, **kwargs)

        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()

        # only one process
        if size == 1:
            return vectorize(f)(xs, *args, **kwargs)

        result = [None] * len(xs)

        # compute results
        # NICE TO HAVE: implement better load balancing; right now simplest
        #   distribution of tasks to processes
        for i, x in enumerate(xs):
            if rank == i % size:
                result[i] = f(x, *args, **kwargs)

        comm.Barrier()

        # communicate results

        # for the easy load balancing
        # communicate everything to root
        for i in range(len(xs)):
            if i % size == 0:
                # already there
                continue
            if rank == i % size:
                # process that sends
                comm.send(result[i], dest=0, tag=0)
            elif rank == 0:
                # root receives
                result[i] = comm.recv(source=(i % size), tag=0)
        comm.Barrier()

        # distribute data to all cores
        for i in range(len(xs)):
            if rank == 0:
                # root sends to all processes
                for j in range(size):
                    comm.send(result[i], dest=j, tag=1)
            else:
                # each process receives from root
                result[i] = comm.recv(source=0, tag=1)
        comm.Barrier()
        return result
    return newfun


def vectorize_parallel(method='processes', num_procs=2, use_progressbar=False):
    """decorator for parallel vectorization of functions.

    -- method: can be 'processes' for shared-memory parallelization or 'MPI'
       for distributed memory parallelization.
    -- num_procs: number of processors for method == 'processes'
    -- use_progressbar: for method == 'processes', this indicates if a
       progress bar should be printed; requires progressbar module

    Example for multiprocessing:

    >>> from parallel_decorators import vectorize_parallel
    >>> @vectorize_parallel(method='processes', num_procs=2)
    ... def power(x, y):
    ...     return x**y
    >>> result = power(range(5), 3)
    [0, 1, 8, 27, 64]


    Example for MPI:

    >>> from parallel_decorators import vectorize_parallel, is_master
    >>> @vectorize_parallel(method='MPI')
    ... def power(x, y):
    ...     return x**y
    >>> # computation in parallel here
    >>> result = power(range(5), 3)
    [0, 1, 8, 27, 64]
    >>> if is_master();
    ...    # use results on master core
    ...    print(result)

    Then start script with
    $ mpiexec -np <num> python script.py
    """
    if method == 'processes':
        return vectorize_queue(num_procs, use_progressbar)
    elif method == 'MPI':
        return vectorize_mpi
    else:
        return vectorize
