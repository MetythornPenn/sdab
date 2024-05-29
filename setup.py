import os
from setuptools import setup

def read(*paths):
    with open(os.path.join(*paths), 'r') as f:
        return f.read()
    
requirements = [
    'datasets',
    'transformers',
    'torchaudio',
    'soundfile',
    'torch',
    'numpy',
]

setup(
    name='sdab',
    version='0.01',
    packages=['sdab'],
    url='https://github.com/MetythornPenn/sdab.git',
    license='Apache Software License 2.0',
    author = 'Metythorn Penn',
    author_email = 'metythorn@gmail.com',
    keywords='asr',
    description='Khmer Speech To Text Inference API using Wav2Vec2 with Pretrain Model',
    install_requires=requirements,
    long_description=(read('README.md')),
    long_description_content_type='text/markdown',
	classifiers= [
		'Development Status :: 1 - Planning',
		'Intended Audience :: Developers',
		'Natural Language :: Khmer',
		'License :: OSI Approved :: Apache Software License',
		'Operating System :: OS Independent',
		'Programming Language :: Python',
		'Programming Language :: Python :: 3',
		'Programming Language :: Python :: Implementation :: CPython',
		'Topic :: Scientific/Engineering',
	],
)