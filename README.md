# DAP Lab AI Benchmarking Eval Script

## How to run

- create virtual env
```bash
python -m venv .venv
source .venv/bin/activate
```

- install dependencies
```bash
pip install -r requirements.txt
```

- run the script
```bash
bash eval.sh
```

## Official Test Set

The official test dataset is created with the command

```bash
python generate_test_cases.py --output test_cases_official --limit 200
```