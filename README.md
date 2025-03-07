# engineering-base
This is a repository for engineers to create and run engineering workflows.

## Pre-requisites
- Python 3.12+
- Git
- Github desktop
- Github extension for VS Code

## Recommended extensions
- Github
- Gitlens
- Pylance
- Black
- Supermaven / Github Copilot

## Installation
1. Clone this repository
Contact Jerry Lei (leiweikuang@gamil.com) to gain access to the Github private repo and get a link.

2. Create a virtual environment
Open a terminal and run the following command:
``` bash
python -m venv .venv
```

A new folder called .venv will be created in the root directory of the project. Activate the virtual environment by running the following command:
``` bash
.venv/scripts/activate
```
You should now see (.venv) in the beginning of the terminal prompt.

3. Install requirements
Run the following command in the terminal:
``` bash
pip install -r requirements.txt
```
This line installs all necessary packages for the project.

## General workflow
1. Add any reuseable function to the functions folder.
2. Any scripts that are reuseable should be added to the scripts folder.
If your function uses other functions in your code base, make sure you use relative imports.
For example, instead of doing absolute imports like these:
``` python
from engineering_lib.functions.example_function import calc_bearing_pressure
```
Do relative imports likethis:
``` python
from .example_function import calc_bearing_pressure
```

3. For every function and scripts written, add a test file in the tests folder. 
Since tests sit outside your engineering_lib module, use absolute imports for your tests.
For example, do absolute imports like these:
``` python
from engineering_lib.functions.example_function import calc_bearing_pressure
```

## How to run tests
Run the following command in the terminal:
``` bash
pytest
```
This runs all .py files that starts with "test_".
If you only want to run a specific test file, or tests within a folder, run the following command:

``` bash
pytest engineering_lib/test_folder_name/test_file_name.py
```

``` bash
pytest engineering_lib/test_folder_name
```
