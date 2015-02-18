from distutils.core import setup, Extension
import numpy as np

# define the extension module
convolve_tools = Extension('convolve_tools', sources=['convolve_tools/convolve_tools.c'],
                          include_dirs=[np.get_include()])

# run the setup
setup(ext_modules=[convolve_tools])
