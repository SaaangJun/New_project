from setuptools import setup, find_packages

setup(name='multiagent-gymnasium',
      version='0.0.1',
      description='Multi-Agent Goal-Driven Communication Environment... gymnasium version',
      url='https://github.com/dongjaekim-hail/multiagent-particle-envs-gymnasium',
      author='Dongjae Kim',
      author_email='dongjaekim@dankook.ac.kr',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      install_requires=['gymnasium', 'numpy-stl', 'pyglet', 'six']
)
