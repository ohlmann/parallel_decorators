from setuptools import setup

setup(
    name="parallel_decorators",
    version="0.1",
    author="Sebastian Ohlmann",
    author_email="sebastian.ohlmann@gmail.com",
    description=("This module contains decorators for parallel vectorization"
                 " of functions."),
    license="BSD",
    keywords="parallel multiprocessing MPI",
    url="https://github.com/ohlmann/parallel_decorators",
    py_modules=['parallel_decorators'],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
    extras_require={
        "MPI": ['mpi4py'],
        "Progress bar": ['progressbar'],
    },
)
