from setuptools import setup, find_packages

setup(
    name="nacos-config-manager",
    version="0.1",
    packages=find_packages(),
    install_requires=["nacos-sdk-python>=2.0.0", "pydantic>=1.8.2"]
)