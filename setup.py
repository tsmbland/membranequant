from setuptools import setup

setup(
    name='membranequant',
    version='1.0',
    license="MIT",
    author='Tom Bland',
    packages=['membranequant'],
    install_requires=[
        'matplotlib~=3.3.3',
        'numpy~=1.19.4',
        'opencv-python==4.2.0.34 ',
        'pandas==1.0.4',
        'scipy~=1.4.1',
        'scikit-image~=0.14.2',
        'tensorflow~=2.4.0',
        'tensorflow-probability==0.11.1',
        'joblib==0.17.0',
        'future~=0.18.2',
        'tqdm~=4.55.0']
)
