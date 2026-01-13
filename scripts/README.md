# Scripts Directory

This directory contains utility scripts, test files, and batch commands that are not part of the main application.

## Structure

```
scripts/
├── batch_files/          # Windows batch scripts
│   ├── run.bat          # Legacy GUI launcher
│   ├── run_crypto_tracker.bat
│   ├── run_crypto_tracker_debug.bat
│   └── run_data_fetch.bat
├── debug_check.py       # Debug validation script
├── diagnose_nan.py      # NaN diagnostic tool
├── example_validate.py  # Example validation
├── fetch_and_predict.py # CLI prediction tool
├── fix_unicode.py       # Unicode encoding fix utility
├── quick_check.py       # Quick validation check
├── run_check.py         # Run checks
├── test_validation.py   # Validation tests
├── output.txt           # Test output logs
├── validation_output.txt
└── validation_results.json
```

## Usage

### Main Application
To run the main application, use the launcher in the root directory:
```bash
# From project root
python main.py
# or
run.bat
```

### Utility Scripts

**Fetch and Predict (CLI)**
```bash
python scripts/fetch_and_predict.py
```

**Fix Unicode Issues**
```bash
python scripts/fix_unicode.py
```

**Run Validation Tests**
```bash
python scripts/test_validation.py
```

## Notes

- These scripts are for testing and debugging purposes
- The main application should be run from `main.py` in the root directory
- Batch files are Windows-specific utilities
