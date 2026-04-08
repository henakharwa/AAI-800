# Fix: Install Python 3.12 for PyTorch Compatibility

PyTorch supports Python 3.8–3.12. Your system has Python 3.15 (alpha) which is not yet supported.

## Step 1 — Install Python 3.12

Download from the official Python website:
https://www.python.org/downloads/release/python-3129/

- Choose: **Windows installer (64-bit)**
- During install, check "Add Python to PATH"
- Install for "All Users" if possible

## Step 2 — Verify Python 3.12 is available

Open a new terminal and run:
```
py -0
```
You should now see `-V:3.12` listed.

## Step 3 — Install all packages on Python 3.12

```
py -3.12 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
py -3.12 -m pip install -r requirements.txt
```

> If you don't have a CUDA GPU, use the CPU-only torch build:
> ```
> py -3.12 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> ```

## Step 4 — Verify environment

```
py -3.12 setup_env.py
```
