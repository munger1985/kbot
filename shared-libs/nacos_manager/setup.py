from setuptools import setup, find_packages
import os

# 获取包的实际路径
package_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name="nacos-manager",  # pip安装用的名称（带连字符）
    version="0.1.0",
    package_dir={"": "."},  # 关键！指定包根目录为当前setup.py所在目录
    packages=find_packages(where="."),  # 从当前目录查找包
    install_requires=[
        "nacos-sdk-python>=2.0.0", 
        "pydantic>=1.8.2"
    ],
    python_requires=">=3.7",
)
