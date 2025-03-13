from setuptools import setup, find_packages

setup(
    name="model_backend",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "numpy",
        "pydantic",
        "psycopg2-binary",
        "requests",
        "torch",
        "torchvision",
        "uvicorn"
    ]
)
