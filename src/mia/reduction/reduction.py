import abc
import copy_reg
import datetime
import logging
import os.path
import time
import types

from mia.io_tools import iterate_directories

logger = logging.getLogger(__name__)


def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    return _unpickle_method, (func_name, obj, cls)


def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)

copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)


class Reduction(object):

    __metaclass__ = abc.ABCMeta

    def __init__(cls, img_path, msk_path):
        cls._img_path = img_path
        cls._msk_path = msk_path

    @abc.abstractmethod
    def process_image(self, image_path, mask_path):
        print "Hello"
        raise NotImplementedError("Method is not implemented")

    def reduce(self, *args, **kwargs):
        start_time = time.time()
        value = self.process_images(*args, **kwargs)
        end_time = time.time()

        total_time = end_time - start_time
        total_time = str(datetime.timedelta(seconds=total_time))
        logger.info("TOTAL REDUCTION TIME: %s" % total_time)

        return value

    def process_images(self, *args, **kwargs):
        image_dirs = iterate_directories(self._img_path, self._msk_path)
        for img_path, msk_path in image_dirs:
            self._process(*args, **kwargs)

    def _process(self, *args, **kwargs):
        print "Hello"
        logger.info("Processing image %s" % os.path.basename(args[0]))

        start_time = time.time()
        value = self.process_image(*args, **kwargs)
        end_time = time.time()

        total_time = end_time - start_time
        logger.info("%.2f seconds to process" % total_time)

        return value
