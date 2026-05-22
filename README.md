# Deterministic Python Environment Setup

This project uses a deterministic local editable installation for verl so that all contributors use the exact same source tree layout and package resolution.

Repository structure:

```
my_project/
├── src/
│   ├── verl/
│   └── other_dirs/
├── requirements.txt
└── README.md
```

---

# 1. Create Virtual Environment

From the repository root, run:

```
bash python -m venv .venv source .venv/bin/activate pip install --upgrade pip pip install -r requirements.txt
```

---

# 2. Verify Installation

Run:

```
bash pip show verl
```

Expected output should include:

```
text Editable project location: /path/to/my_project/src/verl
```

This confirms that Python imports directly from the repository source tree.

---

# 3. Recreating Environment From Scratch

To fully recreate the environment:

```
bash rm -rf .venv
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

# 4. Optional: Freeze Installed Versions

For stronger reproducibility:

bash pip freeze > requirements.lock.txt

Then recreate exactly:

bash pip install -r requirements.lock.txt

Note:
requirements.lock.txt captures exact resolved package versions for the current platform and Python version.
