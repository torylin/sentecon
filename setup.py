from setuptools import find_packages, setup

setup(
    name='sentecon',
    packages=find_packages(include=['sentecon']),
    version='0.1.8',
    description='Interpretable deep embeddings using lexicons',
    author='Victoria Lin',
    url='https://github.com/torylin/sentecon/',
    install_requires=['pandas', 'numpy', 'sentence_transformers', 'torch', 'scikit-learn', 'empath', 'liwc', 'tqdm'],
    package_data={'sentecon' :['data/*']}
)