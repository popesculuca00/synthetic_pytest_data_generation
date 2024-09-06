# Python Unit Test Dataset Generation

This project generates a dataset of Python functions and their **synthetic** corresponding unit tests, designed for training and evaluating language models in automated test generation tasks.

## Process Overview

1. **Main Generation** (`dataset_generation.ipynb`): 
   - Generates the core dataset using an OpenAI-compatible API (vLLM recommended).
   - Saves data in chunks of 1000 samples to prevent data loss from potential failures.

2. **Incomplete Test Generation** (`incomplete_test_generation.ipynb`):
   - Augments the dataset with incomplete test cases for more diverse training scenarios.

3. **Bug Insertion** (`bug_insertion.ipynb`):
   - Further augments the dataset by inserting random bugs into test cases.

4. **Data Aggregation and Analysis** (`eda_and_upload.ipynb`):
   - Combines all generated data.
   - Uploads the final dataset to HuggingFace.
   - Includes a brief exploratory data analysis.

## Utility Scripts

The project includes several shared utility scripts that facilitate various operations:

- **Client Interface and Server Detection**: Supports multiple inference server types (e.g., LLaMA.cpp, vLLM) with automatic configuration.
- **Code Extraction and Formatting**: Extracts and formats Python code from various sources.
- **Pytest Formatting and Execution**: Refines raw test code into well-structured pytest scripts and executes them.
- **Supporting Utilities**: Includes timeout mechanisms, AST manipulation, and ANSI escape sequence handling.

## Requirements

- Python 3.11
- vLLM (recommended) or other OpenAI-compatible API
- Jupyter Notebook
- Additional dependencies listed in `requirements.txt`

## Usage

1. Set up your OpenAI-compatible API (vLLM recommended).
2. Run the notebooks in the order listed above.
3. Adjust parameters as needed in each notebook.
4. Use the generated dataset for training and evaluating language models on unit test generation tasks.
