"""setup script for the PyVideoAI package."""

from setuptools import setup
import versioneer

setup(
    name = "PyVideoAI",
    setup_requires=[],
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author = "Kiyoon Kim",
    author_email='kiyoon.kim@ed.ac.uk',
    description = "Video datasets' annotation parser and etc.",
    url = "https://github.com/kiyoon/PyVideoAI",
    packages=['pyvideoai', 'dataset_configs', 'model_configs', 'exp_configs'],
    python_requires='>=3.6',
    install_requires=['numpy>=1.16.0',
       'coloredlogs', 'verboselogs',
       'matplotlib', 'scipy', 'scikit-learn',
       'seaborn',               # confusion matrix plots
       'pandas>=1.2.4',
       'tensorboard',
       'opencv-python',
       'av',  # Video decoding
       'pretrainedmodels',      # epic models TSN TRN
       'moviepy',               # Tensorboard `add_video()`
       ],
)
