from setuptools import setup, find_packages

setup(
    name="brain-stroke-prediction",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "pandas", "numpy", "scikit-learn", "imbalanced-learn",
        "xgboost", "lightgbm", "matplotlib", "seaborn", "joblib", "flask",
    ],
)