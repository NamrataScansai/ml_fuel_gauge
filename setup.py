from setuptools import setup, find_packages
from typing import List

def get_requirements(file_path: str)->List[str]:

    '''this will return the list of requirements'''

    requirements=[]
    with open(file_path) as f:
        requirements = f.readlines()
    requirements = [req.replace("\n",'') for req in requirements]
    
    if '-e .' in requirements:
        requirements.remove('-e .')
    return requirements
 
setup(
    name='M_Fuel_Gauge',
    version='0.0.1',
    author='NamrataC',
    author_email='namratac@scans.ai',
    packages=find_packages(),    
    install_requires=get_requirements('requirement.txt'),
    description='A package for fuel gauge analysis using machine learning',
    )