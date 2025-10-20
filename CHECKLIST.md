# System Usage Checklist

## Pre-flight Checklist

Before using the system, ensure:

- [ ] Python 3.8+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] RAFT model downloaded to `pretrained_models/raft-things.pth`
- [ ] CUDA available (optional, for faster processing)
- [ ] Video files ready in a directory

---

## First-Time Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Fix NumPy version if needed
pip install "numpy<2"

# 3. Verify installation
python -c "import torch; import cv2; print('OK')"

# 4. Download RAFT model
# Place raft-things.pth in pretrained_models/
```

---

## Quick Test Checklist

Run these commands to verify everything works:

```bash
# Test 1: Check imports
python -c "from unified_dynamics_calculator import UnifiedDynamicsCalculator; print('Import OK')"

# Test 2: Run quick test
python test_simple.py

# Test 3: Check configuration
python dynamics_config.py
```

---

## Processing Checklist

### Single Video Processing

```bash
# Basic command
python video_processor.py -i your_video.mp4 -o output/

# Check output files:
- [ ] output/analysis_results.json exists
- [ ] output/analysis_report.txt exists
- [ ] output/visualizations/ directory created
```

### Batch Processing

```bash
# Batch command
python video_processor.py -i videos/ -o batch_output/ --batch

# Check output files:
- [ ] batch_output/batch_summary.txt exists
- [ ] batch_output/batch_summary.json exists
- [ ] Each video has its own subdirectory
```

---

## Quality Filtering Checklist

After batch processing, apply filters:

```python
from video_quality_filter import VideoQualityFilter

filter = VideoQualityFilter()

# 1. Get statistics
stats = filter.get_quality_statistics(results)
print(stats)

# 2. Filter low-motion videos
low_motion = filter.filter_low_dynamics_in_dynamic_scenes(results, 0.3)

# 3. Review filtered videos
for video in low_motion:
    print(f"{video['video_name']}: {video['score']:.3f}")
    # Decide: re-shoot or accept?
```

Checklist:
- [ ] Reviewed statistics
- [ ] Applied appropriate filters
- [ ] Saved filtered video list
- [ ] Made decisions on flagged videos

---

## Result Interpretation Checklist

For each processed video:

### Check 1: Score Value
- [ ] Score is between 0.0 and 1.0
- [ ] Score makes sense for the video content

### Check 2: Scene Type
- [ ] `scene_type` is 'static' or 'dynamic'
- [ ] Scene type matches video content
  - Buildings/landscapes ¡ú should be 'static'
  - People/animals ¡ú should be 'dynamic'

### Check 3: Classification
- [ ] Category makes sense
  - pure_static / low_dynamic / medium_dynamic / high_dynamic / extreme_dynamic
- [ ] Examples match video content

### Check 4: Confidence
- [ ] Confidence > 0.8 (high reliability)
- [ ] If confidence < 0.6, review video manually

---

## Troubleshooting Checklist

### If scores seem wrong:

- [ ] Check scene type classification
  - Is it correctly identified as static/dynamic?
  
- [ ] Check camera compensation
  - Is it enabled for static scenes?
  - Are there enough feature matches?
  
- [ ] Try different config presets
  - [ ] Tried `strict`
  - [ ] Tried `balanced`
  - [ ] Tried `lenient`
  
- [ ] Check video quality
  - Is the video too blurry?
  - Is there motion blur?
  - Is the resolution very low?

### If processing fails:

- [ ] Check video file format (MP4, AVI, MOV supported)
- [ ] Check if video is corrupted
- [ ] Try with `--device cpu` if CUDA errors
- [ ] Reduce frames with `--max_frames 50`

---

## Production Deployment Checklist

Before deploying to production:

### System Check
- [ ] All tests pass (`python test_simple.py`)
- [ ] Examples run successfully (`python example_usage.py`)
- [ ] Batch processing works on sample dataset

### Configuration
- [ ] Choose appropriate preset (strict/balanced/lenient)
- [ ] Test thresholds on representative sample
- [ ] Document chosen configuration

### Quality Control
- [ ] Define quality thresholds
  - What score is "too low" for dynamic scenes?
  - What score is "too high" for static scenes?
- [ ] Set up filtering pipeline
- [ ] Define review process for filtered videos

### Documentation
- [ ] Team trained on system usage
- [ ] Quality standards documented
- [ ] Filtering criteria defined
- [ ] Review process established

---

## Batch Processing Workflow

### Step 1: Prepare
```bash
# Organize videos
videos/
©À©¤©¤ video1.mp4
©À©¤©¤ video2.mp4
©¸©¤©¤ ...
```

### Step 2: Process
```bash
python video_processor.py -i videos/ -o results/ --batch --config balanced
```

### Step 3: Review Summary
```bash
# Check summary
cat results/batch_summary.txt

# Or view JSON
python -m json.tool results/batch_summary.json
```

### Step 4: Filter
```python
from video_quality_filter import VideoQualityFilter
import json

# Load results
with open('results/batch_summary.json') as f:
    results = json.load(f)

# Filter
filter = VideoQualityFilter()
low_motion = filter.filter_low_dynamics_in_dynamic_scenes(results, 0.3)

# Export
with open('filtered_videos.txt', 'w') as f:
    for v in low_motion:
        f.write(f"{v['video_name']}\n")
```

### Step 5: Take Action
- [ ] Review filtered videos manually
- [ ] Decide which to re-shoot
- [ ] Update dataset
- [ ] Re-process if needed

---

## Monthly Maintenance Checklist

- [ ] Review processing logs
- [ ] Check average processing time
- [ ] Update thresholds if dataset changes
- [ ] Clean up old output directories
- [ ] Backup configuration files

---

## Quick Decision Matrix

### "Should I filter this video?"

For **dynamic scenes** (people/animals):

| Score | Motion Level | Filter? | Action |
|-------|-------------|---------|--------|
| < 0.20 | Almost none | ? Yes | Re-shoot |
| 0.20-0.30 | Very low | ?? Maybe | Review manually |
| 0.30-0.40 | Low-medium | ? No | Acceptable |
| > 0.40 | Normal+ | ? No | Good |

For **static scenes** (buildings):

| Score | Stabilization | Filter? | Action |
|-------|--------------|---------|--------|
| < 0.30 | Excellent | ? No | Good |
| 0.30-0.50 | Good | ? No | Acceptable |
| 0.50-0.70 | Fair | ?? Maybe | Review |
| > 0.70 | Poor | ? Yes | Fix camera |

---

## Complete Example

```python
# 1. Import
from video_processor import VideoProcessor, batch_process_videos
from video_quality_filter import VideoQualityFilter

# 2. Create processor
processor = VideoProcessor(
    raft_model_path="pretrained_models/raft-things.pth",
    device='cuda',
    config_preset='balanced'
)

# 3. Batch process
results = batch_process_videos(
    processor,
    input_dir='videos/',
    output_dir='output/',
    camera_fov=60.0
)

# 4. Filter
filter = VideoQualityFilter()
low_motion = filter.filter_low_dynamics_in_dynamic_scenes(results, 0.3)

# 5. Report
print(f"Processed: {len(results)} videos")
print(f"Low motion: {len(low_motion)} videos")

# 6. Export
with open('low_motion_list.txt', 'w') as f:
    for v in low_motion:
        f.write(f"{v['video_name']}: {v['score']:.3f} - {v['reason']}\n")

# 7. Review
for video in low_motion[:5]:  # First 5
    print(f"\nReview: {video['video_name']}")
    print(f"  Score: {video['score']:.3f}")
    print(f"  Recommendation: {video['recommendation']}")
```

---

## Success Criteria

Your system is working correctly if:

- ? Videos of still people get low scores (< 0.3)
- ? Videos of walking people get medium scores (0.4-0.6)
- ? Videos of dancing people get high scores (> 0.6)
- ? Static building videos get very low scores (< 0.15)
- ? Scene type detection is accurate (> 90%)

---

## Getting Help

If stuck:
1. Read `README.md` - Comprehensive documentation
2. Check `QUICK_START_V2.md` - Quick start guide
3. Review `SYSTEM_OVERVIEW.md` - System explanation
4. Run `python example_usage.py` - See examples
5. Check `REFACTORING_COMPLETE.md` - Technical details

---

**Ready to use the system? Start with Quick Start!** ?

