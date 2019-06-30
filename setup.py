from setuptools import setup

setup(name='xnn',
      version='0.1',
      description='The explainable neural network with sparse, orthogonal and smooth constraints',
      url='https://github.com/ZebinYang/Explainable-Neural-Networks',
      author='Zebin Yang',
      author_email='yangzb2010@hku.hk',
      license='GPL',
      packages=['xnn'],
      install_requires=[
          'matplotlib>=2.2.2','tensorflow>=2.0.0b0', 'numpy>=1.15.2'],
      zip_safe=False)
