from setuptools import setup, find_packages

setup(
    name='fat_llama',
    version='1.0.2.3',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'cupy-cuda12x',
        'pydub',
        'soundfile',
        'mutagen',
        'scipy',
    ],
    package_data={
        'fat_llama': ['audio_fattener/*.py', 'tests/*.py'],
    },
    entry_points={
        'console_scripts': [
            'example=example:main',
        ],
    },
    author='RaAd',
    author_email='bulkguy47@gmail.com',
    description='fat_llama is a Python package for upscaling audio files to FLAC or WAV formats using advanced audio processing techniques. It utilizes CUDA-accelerated calculations to enhance audio quality by upsampling and adding missing frequencies through FFT (Fast Fourier Transform), resulting in richer and more detailed audio.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/bkraad47/fat_llama',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    license='BSD-3-Clause',
)
