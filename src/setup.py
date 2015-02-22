try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

with open('./requirements.txt', 'r') as file_handle:
    requirements = map(lambda s: s.strip(), file_handle.readlines())

config = {
    'description': 'Mammogram Image Analysis package',
    'author': 'Samuel Jackson',
    'url': 'http://github.com/samueljackson92/major-project',
    'download_url': 'http://github.com/samueljackson92/major-project',
    'author_email': 'samueljackson@outlook.com',
    'version': '0.4.0',
    'install_requires': requirements,
    'entry_points': '''
        [console_scripts]
        mia=pipeline:main
    ''',
    'packages': ['mammogram'],
    'scripts': ['pipeline.py'],
    'name': 'mia'
}

setup(**config)
