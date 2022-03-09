from setuptools import setup, find_packages

setup(
    name='Vision',
    version='0.1.0',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        'pytest',
        'numpy',
        'coverage',
        'tensorflow',
        'opencv-python',
        'scikit-image',
        'matplotlib',
    ],
)
