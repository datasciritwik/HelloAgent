from setuptools import setup, find_packages

setup(
    name="HelloAgent",
    version="1.0.0",
    packages=find_packages(),
    py_modules=["main"],
    install_requires=[
    "click>=8.2.1",
    "duckdb>=1.3.1",
    "langchain>=0.3.26",
    "langchain-anthropic>=0.3.16",
    "langchain-google-genai>=2.1.6",
    "langchain-groq>=0.3.4",
    "langchain-openai>=0.3.27",
    "langgraph>=0.5.0",
    "python-dotenv>=1.1.1",
    "requests>=2.32.4",
    "rich>=14.0.0",
    "setuptools>=80.9.0",
],
    entry_points={
        'console_scripts': [
            'helloagent=main:main',
        ],
    },
    author="Ritwik Singh",
    author_email="officialritwik098@gmail.com",
    description="AI Agent CLI with LangGraph",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/datasciritwik/HelloAgent",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)