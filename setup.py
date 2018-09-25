from subprocess import check_output

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


__version__ = '0.0.1'
try:
    hg_id = check_output(['hg', 'id', '-i']).decode().strip('\n')
    __version__ += '-{}'.format(hg_id.replace('+', '_modified'))
except FileNotFoundError:
    pass
with open('loopyplot/version.py', 'w') as file:
    file.write('__version__ = {!r}'.format(__version__))


setup(
    name='loopyplot',
    version=__version__,
    author='Friedrich Hagedorn',
    author_email='friedrich_h@gmx.de',
    url='https://github.com/fhgd/LoopyPlot',
    description='Plot nested loop data for scientific and ' \
                'engineering tasks in Python.',
    install_requires=['numpy', 'matplotlib', 'pandas', 'pydot'],
    keywords='loopyplot, plot, matplotlib, for-loops, function, sweeps',
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GPL 3 License',
        'Topic :: Software Development :: Libraries',
        # Indicate who your project is intended for
        'Intended Audience :: Education',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
    ],
    packages=['loopyplot'],
    licence='GPL 3',

)
