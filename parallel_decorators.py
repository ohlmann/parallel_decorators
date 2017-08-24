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
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        return rank == 0
    except ImportError:
        return True


def is_mpi():
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        if size > 1:
            return True
        else:
            return False
    except ImportError:
        return False


def is_iterable(xs):
    """returns True if xs is an iterable"""
    # try to get iterator; if error occurs, xs is not iterable
    try:
        iter(xs)
    except TypeError:
        return False
    return True


def staticvariables(**variables):
    def decorate(function):
        for variable in variables:
            setattr(function, variable, variables[variable])
        return function
    return decorate


def vectorize(f):
    """decorator for vectorization of functions

    Function wrapper that vectorizes f over the first argument
    if the first argument is an iterable.
    """
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


def vectorize_queue(num_procs=2, use_progressbar=False, label=None):
    """decorator for parallel vectorization of functions using processes

    Function wrapper that vectorizes f over the first argument
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
    show_progressbar = False
    if use_progressbar:
        try:
            from progressbar import Bar, AdaptiveETA, Percentage, ProgressBar,\
                    FormatLabel
            show_progressbar = True
        except ImportError:
            print("Progressbar requested, but module progressbar not found."
                  " Disabling progressbar.")

    @staticvariables(show_progressbar=show_progressbar)
    def decorator(f):
        """the decorator function we return"""
        @staticvariables(show_progressbar=decorator.show_progressbar)
        @wraps(f)
        def newfun(xs, *args, **kwargs):
            """the function that replaces the wrapped function"""
            if not is_iterable(xs):
                # no iteration, simply call function
                return f(xs, *args, **kwargs)

            show_progressbar = newfun.show_progressbar

            from multiprocessing import Process, Queue

            if show_progressbar:
                if label is None:
                    bar_label = f.__name__
                else:
                    bar_label = label
                widgets = [FormatLabel(bar_label), ' ', Percentage(),
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
                if show_progressbar:
                    pbar.update(i)

            # stop workers
            for i in range(num_procs):
                task_queue.put('STOP')

            if show_progressbar:
                pbar.finish()

            # error ocurred?
            if len(errors) > 0:
                print("Caught at least one error during execution:")
                raise errors[0]

            return result
        return newfun
    return decorator


def vectorize_mpi(use_progressbar=False, label=None, scheduling='auto'):
    """Decorator for parallel vectorization of functions using MPI

    Function wrapper that vectorizes f over the first argument
    if the first argument is an iterable.
    This function wrapper uses the mpi4py module to implement
    parallelism for the vectorization.

    Usage:

    >>> @vectorize_mpi()
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
    show_progressbar = False
    if use_progressbar:
        try:
            from progressbar import Bar, AdaptiveETA, Percentage, ProgressBar,\
                    FormatLabel
            show_progressbar = True
        except ImportError:
            print("Progressbar requested, but module progressbar not found."
                  " Disabling progressbar.")

    @staticvariables(show_progressbar=show_progressbar)
    def decorator(f):
        @wraps(f)
        @staticvariables(show_progressbar=decorator.show_progressbar)
        def newfun(xs, *args, **kwargs):
            """the function that replaces the wrapped function"""
            if not is_iterable(xs):
                # no iteration, simply call function
                return f(xs, *args, **kwargs)

            show_progressbar = newfun.show_progressbar

            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            size = comm.Get_size()
            rank = comm.Get_rank()

            if rank != 0:
                show_progressbar = False
            pbar = None

            # only one process
            if size == 1:
                return vectorize(f)(xs, *args, **kwargs)

            result = [None] * len(xs)
            error = None

            comm.Barrier()

            # simple task distribution for less than 4 tasks or if indicated by
            # parameter scheduling
            # otherwise use slave-master model
            if (scheduling == 'static' or (scheduling == 'auto' and size < 4))\
                    and not (scheduling == 'dynamic'):
                # compute results
                for i, x in enumerate(xs):
                    if rank == i % size:
                        result[i] = f(x, *args, **kwargs)
                # communicate results
                # for the easy load balancing
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
            else:
                if rank == 0:
                    # create progressbar
                    if show_progressbar:
                        if label is None:
                            bar_label = f.__name__
                        else:
                            bar_label = label
                        widgets = [FormatLabel(bar_label), ' ', Percentage(),
                                   Bar(), AdaptiveETA()]
                        pbar = ProgressBar(widgets=widgets, maxval=len(xs))
                        pbar.start()
                    # master process -> handles distribution of tasks
                    all_sent = False
                    current = 0
                    ranks = [None] * len(xs)
                    reqs_sent = []
                    reqs_rcvd = []
                    completed_reqs = []
                    # send first batch of tasks
                    for i in range(1, size):
                        ranks[current] = i
                        reqs_sent.append(comm.isend((current, None),
                                         dest=i, tag=0))
                        reqs_rcvd.append(comm.irecv(source=i, tag=current))
                        if current < len(xs):
                            current += 1
                        else:
                            break
                    for r in reqs_sent:
                        r.wait()
                    while True:
                        new_reqs = []
                        for i, r in enumerate(reqs_rcvd):
                            # check for completed requests
                            completed, data = r.test()
                            if completed:
                                if data is None:
                                    continue
                                completed_reqs.append(i)
                                if data[2] is not None:
                                    error = data[2]
                                result[data[0]] = data[1]
                                # check if all tasks have been distributed
                                if current >= len(xs):
                                    all_sent = True
                                    continue
                                ranks[current] = ranks[data[0]]
                                # send new taks and get result (asynchronously)
                                comm.send((current, None),
                                          dest=ranks[data[0]], tag=0)
                                new_reqs.append(comm.irecv(
                                    source=ranks[data[0]], tag=current))
                                current += 1
                        for r in new_reqs:
                            reqs_rcvd.append(r)
                        if all_sent and len(completed_reqs) == len(xs):
                            # send None to all processes to exit loop
                            req_finished = []
                            for r in range(1, size):
                                req_finished.append(comm.isend(
                                    (None, None), dest=r, tag=0))
                            for req in req_finished:
                                req.wait()
                            break
                        # check if error occurred and propagate to slaves
                        if error is not None:
                            for r in range(1, size):
                                comm.send((None, error), dest=r, tag=0)
                            break
                        # update progressbar
                        if show_progressbar:
                            pbar.update(len(completed_reqs))
                else:
                    # slave processes -> do the computation
                    current, e = comm.recv(source=0, tag=0)
                    while True:
                        # compute result for index current
                        try:
                            res = f(xs[current], *args, **kwargs)
                            comm.send((current, res, None),
                                      dest=0, tag=current)
                        except Exception as e:
                            print("Caught exception in parallel vectorized "
                                  "function:")
                            # print out traceback
                            traceback.print_exc()
                            print()
                            comm.send((current, None, e), dest=0, tag=current)
                        # receive next task
                        current, e = comm.recv(source=0, tag=0)
                        # exit loop if None is sent
                        if current is None:
                            if e is not None:
                                error = e
                            break

            comm.Barrier()

            if show_progressbar and pbar is not None:
                pbar.finish()

            # check for error
            if error is not None:
                if rank == 0:
                    raise error
                return

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
    return decorator


def vectorize_parallel(method='processes', num_procs=2, use_progressbar=False,
                       label=None, scheduling='auto'):
    """Decorator for parallel vectorization of functions

    -- method: can be 'processes' for shared-memory parallelization or 'MPI'
       for distributed memory parallelization or 'adaptive' to use MPI if it is
       active (caution: mpi4py must be installed for using mpi!)
    -- num_procs: number of processors for method == 'processes'
    -- use_progressbar: this indicates if a progress bar should be printed;
       requires progressbar module
    -- label: for use_progressbar==True, this sets the label of the
       progress bar. Defaults to the name of the decorated function.
    -- scheduling: scheduling method to use for method == 'MPI'; can be
       'auto', 'static', or 'dynamic'

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
    if method == 'adaptive':
        if is_mpi():
            method = 'MPI'
        else:
            method = 'processes'
    if method == 'processes':
        return vectorize_queue(num_procs, use_progressbar, label)
    elif method == 'MPI':
        return vectorize_mpi(use_progressbar, label, scheduling)
    else:
        return vectorize
