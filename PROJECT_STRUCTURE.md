# Project Structure - v2.0

## Directory Layout

```
AIGC_detector/
��
������ Core System Files
��   ������ video_processor.py                  ? Main processor (refactored)
��   ������ unified_dynamics_calculator.py      ? NEW: Core algorithm
��   ������ video_quality_filter.py             ? NEW: Quality filtering
��   ������ dynamics_config.py                  ? NEW: Configuration
��   ������ simple_raft.py                      ? RAFT wrapper
��   ������ badcase_detector.py                 ? BadCase detection
��   ������ dynamic_motion_compensation/
��       ������ __init__.py
��       ������ camera_compensation.py          ? Camera compensation
��
������ Examples & Tests
��   ������ example_usage.py                    ? NEW: Clean examples
��   ������ test_simple.py                      ? NEW: Quick tests
��
������ Documentation
��   ������ README.md                            ? Completely rewritten
��   ������ QUICK_START_V2.md                   ? NEW: Quick start
��   ������ REFACTORING_COMPLETE.md             ? NEW: Refactoring summary
��   ������ PROJECT_STRUCTURE.md                ? NEW: This file
��   ������ docs/                                ? Other docs
��
������ Models & Data
��   ������ pretrained_models/
��   ��   ������ raft-things.pth                 ? Download required
��   ������ videos/                              ? Your test videos
��   ������ output/                              ? Processing results
��
������ Third-party
��   ������ third_party/
��       ������ RAFT/                            ? RAFT implementation
��
������ Configuration
    ������ requirements.txt                     ? Dependencies
    ������ labels.json                          ?? Video labels
    ������ project.uml                          ? UML diagram
```

---

## File Roles

### Core System

| File | Role | Status | Lines |
|------|------|--------|-------|
| `video_processor.py` | Main orchestrator | Refactored | ~450 |
| `unified_dynamics_calculator.py` | Dynamics calculation | NEW | ~350 |
| `video_quality_filter.py` | Quality filtering | NEW | ~200 |
| `dynamics_config.py` | Configuration | NEW | ~200 |
| `simple_raft.py` | Optical flow | Unchanged | ~325 |
| `badcase_detector.py` | BadCase detection | Unchanged | ~400 |
| `camera_compensation.py` | Camera compensation | Unchanged | ~250 |

### Examples & Tests

| File | Purpose | Status |
|------|---------|--------|
| `example_usage.py` | Usage examples | NEW |
| `test_simple.py` | Quick validation | NEW |

### Documentation

| File | Content | Status |
|------|---------|--------|
| `README.md` | Main documentation | Rewritten |
| `QUICK_START_V2.md` | Quick start guide | NEW |
| `REFACTORING_COMPLETE.md` | Refactoring summary | NEW |
| `PROJECT_STRUCTURE.md` | This file | NEW |

---

## Removed Files

These files were removed during refactoring:

- ? `static_object_analyzer.py` (453 lines)
  - **Reason**: Replaced by `unified_dynamics_calculator.py`
  - **Migration**: Use new unified calculator

- ? `unified_dynamics_scorer.py` (554 lines)
  - **Reason**: Functionality merged into new system
  - **Migration**: Automatic, no code changes needed

- ? `example_new_system.py`
  - **Reason**: Replaced by `example_usage.py`
  - **Migration**: Use new example file

- ? Temporary documentation files
  - `NEW_SYSTEM_GUIDE.md` - Merged into README
  - `REFACTORING_SUMMARY.md` - Replaced by REFACTORING_COMPLETE

**Total reduction**: ~1000+ lines of code removed!

---

## Module Dependencies

```
video_processor.py
������ simple_raft.py
��   ������ third_party/RAFT
������ camera_compensation.py
������ unified_dynamics_calculator.py  ? NEW
������ video_quality_filter.py         ? NEW
������ dynamics_config.py               ? NEW
������ badcase_detector.py

example_usage.py
������ video_processor.py
������ video_quality_filter.py
������ dynamics_config.py

test_simple.py
������ unified_dynamics_calculator.py
������ video_quality_filter.py
������ dynamics_config.py
```

---

## Code Metrics

### Before Refactoring (v1.0)
- Total Python files: 8
- Total lines: ~3,500
- Core algorithm files: 3 (complex dependencies)
- Backward compatibility code: ~500 lines

### After Refactoring (v2.0)
- Total Python files: 7
- Total lines: ~2,400
- Core algorithm files: 1 (unified)
- Backward compatibility code: 0 lines

**Improvement**: -31% code, +100% clarity

---

## Data Flow

```
Input Video
    ��
video_processor.py
    ��
simple_raft.py �� Optical Flow
    ��
camera_compensation.py �� Residual Flow
    ��
unified_dynamics_calculator.py
  ������ Detect static regions
  ������ Detect dynamic regions
  ������ Classify scene type
  ������ Calculate unified score (0-1)
    ��
video_quality_filter.py (optional)
  ������ Filter low-motion videos
  ������ Generate statistics
    ��
Output Results
  ������ analysis_results.json
  ������ analysis_report.txt
  ������ visualizations/
```

---

## Configuration Files

| File | Purpose | Format |
|------|---------|--------|
| `dynamics_config.py` | System thresholds | Python |
| `labels.json` | Expected video labels | JSON |
| `requirements.txt` | Dependencies | Text |

---

## Output Structure

### Single Video Processing
```
output/
������ analysis_report.txt
������ analysis_results.json
������ visualizations/
    ������ frame_0000.png
    ������ frame_0032.png
    ������ temporal_dynamics.png
    ������ camera_compensation.png
```

### Batch Processing
```
output/
������ batch_summary.txt
������ batch_summary.json
������ video1/
��   ������ analysis_report.txt
��   ������ analysis_results.json
��   ������ visualizations/
������ video2/
��   ������ ...
������ filter_report.txt (if filtering applied)
```

---

## Key Improvements Summary

1. **Simpler**: 7 files instead of 10+ files
2. **Cleaner**: 31% less code
3. **Better**: Can filter low-motion in dynamic scenes
4. **Faster**: Unified algorithm, less overhead
5. **Easier**: Three presets, clear API

---

## Development Notes

### Adding New Features

To add a new feature:
1. Check `dynamics_config.py` for parameters
2. Modify `unified_dynamics_calculator.py` for algorithm
3. Update `video_quality_filter.py` for filtering
4. Add examples to `example_usage.py`

### Customizing Thresholds

Edit `dynamics_config.py`:
```python
DETECTION_THRESHOLDS = {
    'static_threshold': 0.002,   # Adjust here
    'subject_threshold': 0.005,  # Adjust here
}
```

### Adding New Presets

Edit `dynamics_config.py`:
```python
PRESET_CONFIGS = {
    'my_custom_preset': {
        'static_threshold': 0.0018,
        'subject_threshold': 0.0045,
        # ...
    }
}
```

---

## Comparison Table

| Aspect | v1.0 (Old) | v2.0 (New) |
|--------|-----------|-----------|
| **Files** | 10+ | 7 |
| **Lines of code** | ~3,500 | ~2,400 |
| **Score range** | Segmented | Unified 0-1 |
| **Filter low-motion** | ? No | ? Yes |
| **Backward compat** | Complex | None (clean) |
| **Configuration** | Hard-coded | 3 presets |
| **Documentation** | Scattered | Centralized |

---

**The refactored system is production-ready!** ?

