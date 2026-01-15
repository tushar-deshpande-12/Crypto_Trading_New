# AI DEVELOPMENT GUIDELINES - READ THIS FIRST

**IMPORTANT: AI assistants must read this document before making any code changes.**

This document provides comprehensive guidelines for developing and maintaining the Crypto AI Trading System. Follow these guidelines to ensure code quality, test coverage, and proper integration.

---

## TABLE OF CONTENTS

1. [Development Workflow](#1-development-workflow)
2. [Code Change Requirements](#2-code-change-requirements)
3. [Testing Pipeline](#3-testing-pipeline)
4. [Debug Logging System](#4-debug-logging-system)
5. [GUI Integration](#5-gui-integration)
6. [Setup for New PC](#6-setup-for-new-pc)
7. [Model Accuracy Improvements](#7-model-accuracy-improvements)
8. [Professional Code Standards](#8-professional-code-standards)
9. [Modularity Guidelines](#9-modularity-guidelines)
10. [Feature Integration Checklist](#10-feature-integration-checklist)
11. [Centralized Debug System](#11-centralized-debug-system)
12. [AI Development Loop](#12-ai-development-loop)

---

## 1. DEVELOPMENT WORKFLOW

For EVERY code change, follow this workflow:

```
1. READ this document first
2. Make feature changes
3. Update related tests in test_scripts/
4. Add debug logging points
5. Run test suite: python src/test_scripts/test_full_pipeline.py
6. Check debug logs in debug_logs/
7. Fix any issues found
8. Verify GUI integration
9. Ensure model accuracy is not degraded
```

**Command to run tests:**
```bash
python src/test_scripts/test_full_pipeline.py
```

---

## 2. CODE CHANGE REQUIREMENTS

### For EVERY change made, ensure:

1. **Cross-Feature Consistency**
   - Changes in one module must be reflected in dependent modules
   - Update all related files (preprocessor, dataset, trainer, predictor, backtester)
   - Keep constants synchronized (PREDICTION_HORIZON, context_length, etc.)

2. **Professional Codebase Standards**
   - Clear, descriptive variable and function names
   - Comprehensive docstrings
   - No hardcoded values (use config files)
   - Proper error handling with informative messages
   - No Unicode characters in output (Windows compatibility)

3. **Testing Updates**
   - Add/update tests for new features
   - Ensure test coverage for edge cases
   - Update test_full_pipeline.py if needed

4. **Debug Logging**
   - Add debug_log() calls at critical points
   - Log input/output shapes and statistics
   - Log errors with full context

5. **GUI Integration**
   - Ensure changes work with GUI panels
   - Test GUI callbacks and progress updates
   - Maintain thread safety

---

## 3. TESTING PIPELINE

### Test Files Location
```
src/test_scripts/
├── __init__.py
├── test_full_pipeline.py    # Main test suite
└── PIPELINE_REFERENCE.md    # Technical reference
```

### Running Tests
```bash
# Full test suite
python src/test_scripts/test_full_pipeline.py

# Individual tests (from Python)
from src.test_scripts.test_full_pipeline import test_preprocessing
train_df, val_df, test_df, preprocessor = test_preprocessing(["BTCUSDT"])
```

### Test Components
| Test | Purpose | Expected Outcome |
|------|---------|------------------|
| test_preprocessing | Data loading and feature engineering | [OK] PASSED |
| test_dataset_creation | DataLoader creation | [OK] PASSED |
| test_model_training | Short training run | [OK] PASSED |
| test_prediction | Model inference | [OK] PASSED |
| test_backtesting | Trading simulation | [OK] PASSED |

---

## 4. DEBUG LOGGING SYSTEM

### Location
```
debug_logs/
├── debug_YYYYMMDD_HHMMSS.json   # Session logs
└── summary_YYYYMMDD_HHMMSS.json # Session summary
```

### Usage in Code
```python
from src.utils.debug_logger import debug_log, get_debug_logger

# Initialize logger
debug = get_debug_logger()

# Log events
debug_log("preprocessing", "load_data", {"shape": df.shape}, "info")
debug_log("training", "epoch_1", {"loss": 0.5}, "success")
debug_log("prediction", "error", {"message": str(e)}, "error")

# Log data statistics
debug.log_data_stats("preprocessing", dataframe, "train_data")

# Log training progress
debug.log_training_step(epoch=1, train_loss=0.5, val_loss=0.6)
```

### Debug Log Structure
```json
{
  "timestamp": "2026-01-15T14:00:00",
  "stage": "preprocessing",
  "step": "load_data",
  "status": "info|warning|error|success",
  "data": {...}
}
```

---

## 5. GUI INTEGRATION

### GUI Components Location
```
src/gui/
├── app.py                  # Main application
├── styles.py               # Theming
└── components/
    ├── symbol_table.py     # Crypto list
    ├── chart_panel.py      # Charts
    ├── data_panel.py       # Data fetching
    ├── ml_panel.py         # Training/Prediction
    └── backtest_panel.py   # Backtesting
```

### When modifying backend code, check:
1. GUI callbacks in `src/ml/training/gui_callback.py`
2. Thread safety (use `root.after()` for GUI updates)
3. Progress bar updates
4. Status message updates

---

## 6. SETUP FOR NEW PC

### Required Dependencies
Run this command to install all dependencies:
```bash
pip install -r requirements.txt
```

### Key Dependencies
```
pytorch>=2.0.0
pytorch-lightning>=2.0.0
pytorch-forecasting>=1.0.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.0.0
matplotlib>=3.7.0
tensorboard>=2.12.0
```

### GPU Setup (Optional)
```bash
# For CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Verify Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

### First Run
```bash
# 1. Run tests to verify setup
python src/test_scripts/test_full_pipeline.py

# 2. Start GUI application
python main.py
```

---

## 7. MODEL ACCURACY IMPROVEMENTS

### Every Change Should Improve (or Maintain):
1. **Validation Loss** - Should decrease with training
2. **Direction Accuracy** - % of correct up/down predictions
3. **MAE/RMSE** - Prediction error metrics
4. **R2 Score** - Model fit quality

### Current Baseline Metrics
```
Target: val_loss < 1.0 (normalized scale)
Direction Accuracy: > 55%
R2 Score: > 0.1
```

### Techniques to Improve Accuracy
1. **Feature Engineering**
   - Add more technical indicators
   - Create interaction features
   - Improve temporal features

2. **Model Architecture**
   - Increase hidden_size (64 -> 128)
   - Add LSTM layers (2 -> 3)
   - Tune attention heads

3. **Training**
   - Longer training (more epochs)
   - Learning rate scheduling
   - Early stopping patience

4. **Data**
   - More training data
   - Multiple symbols
   - Data augmentation

---

## 8. PROFESSIONAL CODE STANDARDS

### File Organization
```
src/
├── core/           # Configuration
├── api/            # External APIs
├── data/           # Data pipeline
├── ml/             # Machine learning
│   ├── models/     # Model definitions
│   ├── training/   # Training logic
│   ├── preprocessing/  # Feature engineering
│   ├── inference/  # Prediction
│   └── backtest/   # Backtesting
├── gui/            # User interface
├── utils/          # Utilities
└── test_scripts/   # Tests
```

### Naming Conventions
- Classes: `CamelCase` (e.g., `CryptoPreprocessor`)
- Functions: `snake_case` (e.g., `load_data`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `PREDICTION_HORIZON`)
- Files: `snake_case.py` (e.g., `data_loader.py`)

### Documentation
Every function must have:
```python
def example_function(param1: type, param2: type) -> return_type:
    """
    Brief description of function.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value
    """
```

---

## 9. MODULARITY GUIDELINES

### Dependency Flow
```
api/ -> data/ -> ml/preprocessing/ -> ml/training/ -> ml/inference/
                                  └-> ml/backtest/
```

### Module Independence
- Each module should have clear inputs/outputs
- Use configuration objects, not hardcoded values
- Avoid circular imports
- Use dependency injection where possible

### Configuration Management
```python
# Good: Use config objects
config = TFTConfig(hidden_size=64, ...)
trainer = TFTTrainer(config=config)

# Bad: Hardcoded values
trainer = TFTTrainer(hidden_size=64)  # Don't do this
```

---

## 10. FEATURE INTEGRATION CHECKLIST

When adding a new feature, complete this checklist:

- [ ] Feature code implemented
- [ ] Unit tests added to test_scripts/
- [ ] Debug logging added
- [ ] GUI integration (if applicable)
- [ ] Documentation updated
- [ ] Full test suite passes
- [ ] No regression in model accuracy
- [ ] Code reviewed for Windows compatibility (no Unicode)
- [ ] Config files updated (if new parameters)
- [ ] README updated (if user-facing change)

---

## 11. CENTRALIZED DEBUG SYSTEM

### Debug Log Location
All debug logs are centralized in:
```
debug_logs/
├── debug_YYYYMMDD_HHMMSS.json   # Full session log
└── summary_YYYYMMDD_HHMMSS.json # Quick summary
```

### Checking Debug Logs
```bash
# View latest debug log
python -c "
import json
from pathlib import Path
logs = sorted(Path('debug_logs').glob('debug_*.json'))
if logs:
    with open(logs[-1]) as f:
        data = json.load(f)
        errors = [e for e in data if e['status'] == 'error']
        print(f'Total entries: {len(data)}')
        print(f'Errors: {len(errors)}')
        for e in errors:
            print(f\"  - {e['stage']}/{e['step']}: {e['data']}\")
"
```

### Debug Points by Stage
| Stage | What to Check |
|-------|---------------|
| preprocessing | Data shape, NaN counts, feature stats |
| dataset | Batch shapes, time_idx continuity |
| training | Loss values, gradient norms |
| prediction | Output shapes, value ranges |
| backtesting | Trade signals, portfolio values |

---

## 12. AI DEVELOPMENT LOOP

**The AI should follow this exact loop for every change:**

```
STEP 1: READ DOCUMENTATION
├── Read this file (AI_DEVELOPMENT_GUIDELINES.md)
├── Read PIPELINE_REFERENCE.md for technical details
└── Understand current codebase state

STEP 2: MAKE CHANGES
├── Implement feature/fix
├── Update all dependent files
├── Add debug logging at key points
└── Update tests if needed

STEP 3: RUN TESTS
├── python src/test_scripts/test_full_pipeline.py
├── Wait for all tests to complete
└── Note any failures

STEP 4: CHECK DEBUG LOGS
├── Open debug_logs/debug_*.json
├── Look for errors and warnings
├── Analyze failure context
└── Check data statistics

STEP 5: FIX ISSUES
├── Address errors found in debug logs
├── Re-run tests
└── Repeat until all tests pass

STEP 6: VERIFY INTEGRATION
├── Test GUI if applicable
├── Check model accuracy not degraded
└── Verify Windows compatibility
```

---

## QUICK REFERENCE

### Critical Files
| Purpose | File |
|---------|------|
| Main app | `main.py` |
| Config | `src/core/config.py` |
| Preprocessing | `src/ml/preprocessing/preprocessor.py` |
| Dataset | `src/ml/training/dataset.py` |
| Trainer | `src/ml/training/trainer.py` |
| Predictor | `src/ml/inference/predictor.py` |
| Backtester | `src/ml/backtest/backtester.py` |
| Debug Logger | `src/utils/debug_logger.py` |
| Test Suite | `src/test_scripts/test_full_pipeline.py` |

### Critical Constants (Must Match)
| Constant | Value | Location |
|----------|-------|----------|
| PREDICTION_HORIZON | 24 | preprocessor.py |
| max_prediction_length | 24 | model_config.py |
| prediction_length | 24 | dataset.py |
| max_encoder_length | 168 | model_config.py |
| context_length | 168 | dataset.py |
| target_column | 'target_return' | dataset.py |

### Commands
```bash
# Run tests
python src/test_scripts/test_full_pipeline.py

# Start GUI
python main.py

# Install dependencies
pip install -r requirements.txt

# Check GPU
python -c "import torch; print(torch.cuda.is_available())"
```

---

## REMEMBER

1. **Always run tests after changes**
2. **Check debug logs for errors**
3. **Keep constants synchronized**
4. **No Unicode characters in output**
5. **Test GUI integration**
6. **Improve or maintain model accuracy**
7. **Update documentation**

---

*Last Updated: 2026-01-15*
*Version: 1.0*
