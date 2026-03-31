from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT='-e .'
def get_requirements(file_path: str) -> List[str]:
    requirements = []
    with open(file_path) as file_obj:
        for line in file_obj:
            req = line.strip()
            if req and req != "-e .":
                requirements.append(req)
    return requirements

setup(
    name='indiansignlanguage',
    version='0.0.1',
    author='Aditi Gupta',
    author_email='guptaaditi.0825@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')

)