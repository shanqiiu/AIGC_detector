# AIGC Video Dynamics Assessment System - Overview

## What is This?

A production-ready system that:
- Analyzes video dynamics using optical flow
- Outputs a single 0-1 score indicating motion level
- Automatically adapts to different scene types
- Can filter low-quality videos

## Core Question It Answers

**"How dynamic is this video?"**

Answer: A score from 0 (completely static) to 1 (extremely dynamic)

---

## Two Analysis Modes

### Mode 1: Static Scenes (Buildings, Landscapes)
**Goal**: Check if camera stabilization worked

- Detects static regions in the frame
- Calculates residual flow after camera compensation
- Low scores = good stabilization

**Example**: Building shot with moving camera
- Score 0.08 = Perfect stabilization ?
- Score 0.65 = Poor stabilization ?

### Mode 2: Dynamic Scenes (People, Animals)
**Goal**: Check if subject moved enough

- Detects dynamic regions (subjects)
- Calculates subject motion intensity
- Low scores = insufficient motion

**Example**: Person in video
- Score 0.15 = Person barely moved ? (can filter!)
- Score 0.55 = Person moved normally ?

**The system automatically detects which mode to use!**

---

## Main Use Case: Filter Low-Motion Videos

Problem: You have 100 videos of people, but in 20 videos the person barely moved.

Solution:
```python
from video_processor import batch_process_videos
from video_quality_filter import VideoQualityFilter

# Process all videos
processor = VideoProcessor()
results = batch_process_videos(processor, 'videos/', 'output/', 60.0)

# Filter low-motion videos
filter = VideoQualityFilter()
low_motion = filter.filter_low_dynamics_in_dynamic_scenes(
    results,
    threshold=0.3  # Videos with score < 0.3 are filtered
)

print(f"Found {len(low_motion)} videos where person didn't move enough")

# Save list
with open('low_motion_list.txt', 'w') as f:
    for v in low_motion:
        f.write(f"{v['video_name']}: {v['score']:.3f}\n")
```

---

## Quick Commands

### Process one video
```bash
python video_processor.py -i video.mp4 -o output/
```

### Process many videos
```bash
python video_processor.py -i videos/ -o output/ --batch
```

### Use strict quality control
```bash
python video_processor.py -i videos/ -o output/ --batch --config strict
```

---

## Understanding Your Score

### If scene_type = "dynamic" (person/animal)

| Your Score | What It Means | Action |
|------------|---------------|--------|
| 0.00-0.15 | Subject almost didn't move | ? Filter out |
| 0.15-0.35 | Subject moved slightly | ?? Maybe filter |
| 0.35-0.60 | Subject moved normally | ? Keep |
| 0.60-0.85 | Subject moved actively | ? Keep |
| 0.85-1.00 | Subject moved intensely | ? Keep |

### If scene_type = "static" (building/landscape)

| Your Score | What It Means | Action |
|------------|---------------|--------|
| 0.00-0.15 | Perfect camera stabilization | ? Keep |
| 0.15-0.35 | Good stabilization | ? Keep |
| 0.35-0.60 | Noticeable camera shake | ?? Check |
| 0.60-0.85 | Poor stabilization | ? Review |
| 0.85-1.00 | Severe camera issues | ? Reject |

---

## Configuration Presets

| Preset | When to Use | Threshold | Filters |
|--------|-------------|-----------|---------|
| `strict` | Production quality control | Low | More videos |
| `balanced` | General use (default) | Medium | Moderate |
| `lenient` | Exploration, research | High | Fewer videos |

---

## Output Files

Every processed video generates:

1. **analysis_results.json**
   - Machine-readable data
   - Complete metrics
   
2. **analysis_report.txt**
   - Human-readable report
   - Score interpretation
   
3. **visualizations/** (optional)
   - Frame-by-frame analysis
   - Temporal curves
   - Camera compensation comparison

---

## System Requirements

- **Python**: 3.8+
- **GPU**: Recommended (CUDA 10.2+)
- **RAM**: 8GB+
- **GPU Memory**: 2GB+ (for 720p videos)

---

## Processing Speed

| Resolution | GPU (RTX 3090) | CPU |
|-----------|----------------|-----|
| 720p | ~15 FPS | ~2 FPS |
| 1080p | ~8 FPS | ~1 FPS |
| 4K | ~3 FPS | ~0.3 FPS |

---

## API Quick Reference

### Python API
```python
from video_processor import VideoProcessor, batch_process_videos
from video_quality_filter import VideoQualityFilter

# Process single video
processor = VideoProcessor(config_preset='balanced')
frames = processor.load_video("video.mp4")
result = processor.process_video(frames)

score = result['unified_dynamics']['unified_dynamics_score']
scene = result['unified_dynamics']['scene_type']

# Batch process
results = batch_process_videos(processor, 'videos/', 'output/', 60.0)

# Filter
filter = VideoQualityFilter()
low_motion = filter.filter_low_dynamics_in_dynamic_scenes(results, 0.3)
```

### Command Line
```bash
# Basic
python video_processor.py -i video.mp4 -o output/

# Batch
python video_processor.py -i videos/ -o output/ --batch

# With options
python video_processor.py \
    -i videos/ \
    -o output/ \
    --batch \
    --config strict \
    --device cuda \
    --max_frames 100
```

---

## Common Workflows

### Workflow 1: Quality Control for Dynamic Videos
```
1. Batch process all videos
2. Filter videos with score < 0.3
3. Review filtered videos
4. Request re-shooting if needed
```

### Workflow 2: Assess Static Scene Quality
```
1. Process static scene videos
2. Check if scores < 0.35 (good stabilization)
3. Identify high-score videos (poor stabilization)
4. Adjust camera settings or re-shoot
```

### Workflow 3: Dataset Analysis
```
1. Batch process entire dataset
2. Get quality statistics
3. Visualize score distribution
4. Set quality thresholds based on distribution
```

---

## Integration Examples

### With Video Database
```python
import sqlite3

# Process and store in database
for video_path in video_list:
    result = processor.process_video(frames)
    
    conn.execute("""
        INSERT INTO videos (name, score, scene_type, category)
        VALUES (?, ?, ?, ?)
    """, (
        video_name,
        result['unified_dynamics']['unified_dynamics_score'],
        result['unified_dynamics']['scene_type'],
        result['dynamics_classification']['category']
    ))
```

### With ML Pipeline
```python
# Use as feature for downstream tasks
features = {
    'dynamics_score': result['unified_dynamics']['unified_dynamics_score'],
    'scene_type': result['unified_dynamics']['scene_type'],
    'temporal_stability': result['unified_dynamics']['confidence'],
    # ... other features
}

model.predict(features)
```

---

## Troubleshooting Quick Guide

| Issue | Solution |
|-------|----------|
| NumPy version error | `pip install "numpy<2"` |
| CUDA out of memory | Use `--device cpu` or `--max_frames 100` |
| Processing too slow | Use `--no-viz` flag |
| Model file not found | Download raft-things.pth to pretrained_models/ |
| All videos same score | Adjust config preset or thresholds |

---

## Support & Resources

- **Quick Start**: `QUICK_START_V2.md`
- **Full Documentation**: `README.md`
- **Refactoring Details**: `REFACTORING_COMPLETE.md`
- **Examples**: Run `python example_usage.py`
- **Tests**: Run `python test_simple.py`
- **Configuration**: Edit `dynamics_config.py`

---

## Summary

This system helps you:
1. ? **Assess video dynamics** with a single 0-1 score
2. ? **Filter low-quality videos** automatically
3. ? **Process videos at scale** with batch mode
4. ? **Customize behavior** with three presets
5. ? **Integrate easily** into your pipeline

**The system is production-ready and well-documented!** ?

