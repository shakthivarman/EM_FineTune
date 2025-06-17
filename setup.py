from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    try:
        with open("README.md", "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return "Embedding Model Finetuning Framework"

# Read requirements
def read_requirements():
    try:
        with open("requirements.txt", "r", encoding="utf-8") as fh:
            requirements = []
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#"):
                    requirements.append(line)
            return requirements
    except FileNotFoundError:
        # Fallback requirements if file doesn't exist
        return [
            "torch>=2.0.0,<2.5.0",
            "transformers>=4.34.0,<4.46.0",
            "sentence-transformers>=2.2.2,<3.0.0",
            "datasets>=2.14.0",
            "accelerate>=0.20.1,<0.35.0",
            "peft>=0.4.0,<0.13.0",
            "numpy>=1.24.0,<2.0.0",
            "scikit-learn>=1.3.0,<1.6.0",
            "wandb>=0.15.0",
            "omegaconf>=2.3.0",
            "tqdm>=4.65.0",
            "PyYAML>=6.0",
        ]

setup(
    name="embedding-finetuner",
    version="0.1.0",
    author="Embedding Finetuner Team", 
    author_email="support@example.com",
    description="Hardware-agnostic embedding model finetuning framework",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/embedding-finetuner",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "docs": [
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.2.0",
            "myst-parser>=1.0.0",
        ],
        "viz": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
        ],
        "all": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0", 
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.2.0",
            "myst-parser>=1.0.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "embedding-train=embedding_finetuner.scripts.train:main",
            "embedding-eval=embedding_finetuner.scripts.evaluate:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="embedding, finetuning, transformers, sentence-transformers, nlp, machine-learning",
)