from setuptools import setup, find_packages
 
from pkg_resources import parse_requirements
with open("requirements.txt", encoding="utf-8") as fp:
    install_requires = [str(requirement) for requirement in parse_requirements(fp)]

def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content

if __name__ == '__main__':
    setup(
        name="sonar_segmentation",
        version="1.0.0",
        author="Weichao Cai",
        author_email="caiweichao0914@gmail.com",
        description="A sonar semantic segmentation code templete",
        long_description=readme(),
        license="Apache License, Version 2.0",
        url="https://github.com/GPIOX/semantic-segmentation-code.git",
    
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
        
        
        include_package_data=True, # 一般不需要
        packages=find_packages(),
        install_requires=install_requires,
        entry_points={
            'console_scripts': [
                'test = test.help:main'
            ]
        }
    )