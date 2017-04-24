# Parallel decorators
This module contains decorators for parallel vectorization of functions


The decorators vectorize the function over the first argument and return a
list of the returned results of the functions.
The main decorator is `vectorize_parallel` that supports parallelization via the
multiprocessing module or via MPI.


 * Example for multiprocessing:

        from parallel_decorators import vectorize_parallel
        @vectorize_parallel(method='processes', num_procs=2)
        def power(x, y):
            return x**y
        result = power(range(5), 3)
        [0, 1, 8, 27, 64]

 * Example for MPI:

        from parallel_decorators import vectorize_parallel, is_master
        @vectorize_parallel(method='MPI')
        def power(x, y):
            return x**y
        # computation in parallel here
        result = power(range(5), 3)
        [0, 1, 8, 27, 64]
        if is_master();
           # use results on master core
           print(result)

    Then start script with
    $ mpiexec -np <num> python script.py
