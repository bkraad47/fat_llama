from setuptools import setup, find_packages

setup(
    name='fat_llama',
    version='0.1.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'cupy-cuda12x',
        'pydub',
        'soundfile',
        'mutagen',
        'scipy',
        'tqdm',
    ],
    entry_points={
        'console_scripts': [
            'example=example:main',
        ],
    },
    github_url='https://github.com/bkraad47/fat_llama',
    author='Badruddin Kamal',
    author_email='bulkguy47@gmail.com',
    description='A package for CUDA-based upscaling and processing audio files, using FFT to add detail.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/fat_llama',  # Update this with the actual URL of your project
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    license='BSD-3-Clause',
)
