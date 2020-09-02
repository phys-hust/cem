from setuptools import setup, find_packages

setup(
    name='cem',
    version='0.0.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'pillow',
    ],
    test_requires=[
        'pytest',
    ],
)
