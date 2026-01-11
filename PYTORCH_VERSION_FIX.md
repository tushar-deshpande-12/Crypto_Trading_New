# PyTorch Lightning Version Compatibility Fix

## Problem
You have PyTorch Lightning 2.6.0 installed, but pytorch-forecasting requires PyTorch Lightning < 2.0.0

## Solution
Downgrade PyTorch Lightning to a compatible version:

```bash
pip install pytorch-lightning==1.9.5
```

## Full Command (if needed)
```bash
pip uninstall pytorch-lightning
pip install pytorch-lightning==1.9.5
```

## Why This Happens
- pytorch-forecasting was built for PyTorch Lightning 1.x
- PyTorch Lightning 2.x has stricter type checking that breaks compatibility
- The model IS a LightningModule, but version mismatch causes isinstance() to fail

## After Downgrading
1. Restart the application
2. Try training again - it should work now

## Alternative (if downgrade doesn't work)
Install pytorch-forecasting from a newer source:
```bash
pip install git+https://github.com/jdb78/pytorch-forecasting.git
```
