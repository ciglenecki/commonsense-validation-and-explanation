# 🤔 Commonsense Validation and Explanation (ComVE)

Competition website: https://competitions.codalab.org/competitions/21080

Docs notes: https://docs.google.com/document/d/1nUB-Nw2rtMeh8_mq2UHAL_kMZck2WKo8MhI1Jexj5tk/edit

## Introduction
The task is to directly test whether a system can differentiate natural language statements that make sense from those that do not make sense. There exist three subtasks:
1. The first task is to choose from two natural language statements with similar wordings which one makes sense and which one does not make sense
	```
	Task: Which statement of the two is against common sense?
	Statement1: He put a turkey into the fridge.
	Statement2: He put an elephant into the fridge.
	```
2. The second task is to find the key reason from three options why a given statement does not make sense
	```
	Task: Select the most corresponding reason why this statement is against common sense.
	Statement: He put an elephant into the fridge.
	A: An elephant is much bigger than a fridge.
	B: Elephants are usually white while fridges are usually white.
	C: An elephant cannot eat a fridge.
	```
3. The third task asks machine to generate the reasons and we use BLEU to evaluate them. Examples of all tasks can be found on the competition website.
	```
	Task: Generate the reason why this statement is against common sense and we will use BELU to evaluate it.
	Statement: He put an elephant into the fridge.
	Referential Reasons:
	1. An elephant is much bigger than a fridge.
	2. A fridge is much smaller than an elephant.
	3. Most of the fridges aren’t large enough to contain an elephant.
	```

## Setup

### Python Virtual Environment

Create and populate the [virtual environment](https://docs.python.org/3/library/venv.html#:~:text=A%20virtual%20environment%20is%20a,part%20of%20your%20operating%20system). Simply put, the virtual environment allows you to install Python packages for this project only (which you can easily delete later). This way, we won't clutter your global Python packages.

**Step 1: Create virtual environment and install the packages**

```bash
sudo apt-get install python3-venv
python3 -m venv venv
source venv/bin/activate
sleep 2
pip install -r requirements-dev.txt
pip install -r requirements.txt
# python -m spacy download en_core_web_sm
# python -m spacy download en_core_web_trf # more accurate!
```

**Step 2: Install current directory as a editable Python module:**

This step is important if we use modules in our project (sub directories with `__init__.py`). This step allows you to import variables and functions from files which are a part of another module. For example if you want to import the eval function: `from src.scoring import evaluate` in file `src/train/train.py`.

```bash
pip install -e .
```

**(optional) Step 3: Activate pre-commit hook**

Pre-commit, defined in `.pre-commit-config.yaml` will fix your imports will make sure the code follows Python standards. You can't commit code until you resolve the issues.

```
pre-commit install
```

To remove pre-commit run: `rm -rf .git/hooks`


## 📁 Directory structure

| Directory                 | Description                                         |
| ------------------------- | --------------------------------------------------- |
| [data](data/)             | datasets                                            |
| [docs](docs/)             | documentation                                       |
| [figures](figures/)       | figures                                             |
| [models](models/)         | model checkpoints, model metadata, training reports |
| [references](references/) | research papers and competition guidelines          |
| [src](src/)               | python source code                                  |

## 📋 Notes

empty for now


## 🏆 Team members

<table>
  <tr>
    <td align="center"><a href="https://github.com/mirtamoslavac"><img src="https://avatars.githubusercontent.com/u/72082543?v=4" width="100px;" alt=""/><br /><sub><b>Mirta Moslavac</b></sub><br /></td>
   <td align="center"><a href="https://github.com/rokogrbelja"><img src="https://avatars.githubusercontent.com/u/54799615?v=4" width="100px;" alt=""/><br /><sub><b>Roko Grbelja</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/ciglenecki"><img src="https://avatars.githubusercontent.com/u/12819849?v=4" width="100px;" alt=""/><br /><sub><b>Matej Ciglenečki</b></sub></a><br /></td>
</table>
