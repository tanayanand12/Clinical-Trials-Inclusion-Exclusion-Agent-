from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="clinical-trials-rag",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A library for querying clinical trials data using RAG",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/clinical-trials-rag",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "openai>=1.0.0",
        "requests>=2.28.0",
        "numpy>=1.21.0",
        "python-dotenv>=0.19.0",
        "backoff>=2.2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "mypy>=1.0.0",
        ],
        "api": [
            "fastapi>=0.95.0",
            "uvicorn>=0.20.0",
            "pydantic>=2.0.0",
        ],
    },
)