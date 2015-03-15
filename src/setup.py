try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

with open('./requirements.txt', 'r') as file_handle:
    requirements = map(lambda s: s.strip(), file_handle.readlines())

from distutils.core import Extension
import numpy as np

# define the extension module
convolve_tools = Extension('convolve_tools',
                           sources=['convolve_tools/convolve_tools.c'],
                           include_dirs=[np.get_include()],
			   extra_compile_args=['-std=c99'])

config = {
    'description': 'Mammogram Image Analysis package',
    'author': 'Samuel Jackson',
    'url': 'http://github.com/samueljackson92/major-project',
    'download_url': 'http://github.com/samueljackson92/major-project',
    'author_email': 'samueljackson@outlook.com',
    'version': '0.6.0',
    'install_requires': requirements,
    'entry_points': '''
        [console_scripts]
        mia=command:cli
    ''',
    'ext_modules': [convolve_tools],
    'packages': ['mia'],
    'scripts': ['command.py'],
    'name': 'mia'
}

setup(**config)
