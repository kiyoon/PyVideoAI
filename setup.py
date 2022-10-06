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
       'pandas>=1.3.1',         # pandas 1.3.0 has a bug when loading pickled data
       'tensorboard',
       'opencv-python',
       'moviepy',               # Tensorboard `add_video()`
       'decord',                # Video decoding
#       'av',                    # Video decoding (you may not need it in favour of decord)
       'scikit-image',          # Only used for timecycle
       'gulpio2',               # Efficient frame dataloader
       ],
)
