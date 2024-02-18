from setuptools import setup, find_packages

setup(
    name="vl-data-engine",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.35.0",
        "webdataset>=0.2.5",
        "Pillow>=9.0.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "pandas>=2.0.0",
        "ftfy>=6.1.0",
        "open-clip-torch>=2.20.0",
    ],
    extras_require={
        "dev": ["pytest", "black", "isort", "flake8"],
    },
    entry_points={
        "console_scripts": [
            "vl-filter=scripts.run_filter:main",
            "vl-dedup=scripts.run_dedup:main",
            "vl-pipeline=scripts.run_pipeline:main",
        ]
    },
)
