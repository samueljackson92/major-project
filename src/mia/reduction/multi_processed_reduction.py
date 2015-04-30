import abc
import functools
import itertools
import multiprocessing
import pandas as pd

from mia.reduction.reduction import Reduction
from mia.io_tools import iterate_directories


class MultiProcessedReduction(Reduction):

    __metaclass__ = abc.ABCMeta

    def process_images(self, *args, **kwargs):
        dirs = iterate_directories(self._img_path, self._msk_path)
        paths = [path for path in dirs]
        func, func_args = \
            self._prepare_function_for_mapping(self._process, paths, args)
        multiprocessing.freeze_support()

        n_process = kwargs['num_processes'] if 'num_processes' in kwargs else 1
        pool = multiprocessing.Pool(n_process)
        frames = pool.map(func, func_args)

        return pd.concat(frames)

    def _prepare_function_for_mapping(self, image_function, paths, args):
        """Prepare a function for use with multiprocessing.

        This will prepare the arguments for the function ina representation
        that can be mapped via multiprocessing.
        """
        func = functools.partial(self._func_star, image_function)
        func_args = [tup + arg for tup, arg in
                     zip(paths, itertools.repeat(args))]
        return func, func_args

    def _func_star(self, func, args):
        """Helper method for multiprocessing images.

        Pass the function arguments to the functions running in the child
        process

        :param args: arguments to the process_image function
        :returns: result of the process image function
        """
        return func(*args)
