# AIGC Video Dynamics Assessment System v2.0

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-2.0-green.svg)](WHATS_NEW.md)

> **v2.0 New**: Unified 0-1 scoring, dual-mode analysis, quality filtering, simplified architecture

> Unified video dynamics assessment system for AIGC (AI-Generated Content) videos

## ? What's New in v2.0

- ? **Unified 0-1 Scoring**: No more forced score segmentation
- ? **Can Filter Low-Motion Videos**: Identify videos where subjects barely moved
- ? **Dual-Mode Analysis**: Auto-adapts to static scenes (buildings) and dynamic scenes (people)
- ? **Quality Filtering Tools**: Built-in filters for quality control
- ? **Simplified API**: 30% less code, easier to use
- ? **Configuration Presets**: Three presets for different use cases

[Read full changelog ¡ú](WHATS_NEW.md)

## Overview

A comprehensive system for automatically assessing video dynamics using optical flow analysis. The system intelligently adapts to different scene types and outputs unified 0-1 dynamics scores.

### Key Features

- **Unified Scoring**: 0-1 standardized dynamics score for all videos
- **Dual-Mode Analysis**: Adapts to static scenes (buildings) and dynamic scenes (people/animals)
- **Automatic Scene Detection**: Intelligently classifies scene types
- **Quality Filtering**: Filter low-motion videos in dynamic scenes
- **Camera Compensation**: Removes camera motion to focus on subject dynamics
- **Resolution Normalization**: Fair comparison across different resolutions
- **Batch Processing**: Efficient multi-video processing
- **Rich Visualizations**: Detailed analysis plots and reports

---

## Quick Start

### Prerequisites

- Python 3.8+
- CUDA 10.2+ (recommended for GPU acceleration)
- 8GB+ RAM
- 2GB+ GPU memory (when using GPU)

### Installation

#### 1. Clone repository

```bash
git clone <repository_url>
cd AIGC_detector
```

#### 2. Create virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

#### 3. Install dependencies

```bash
pip install -r requirements.txt
```

#### 4. Download RAFT pretrained model

Download [raft-things.pth](https://drive.google.com/file/d/1x1FLCHaGFn_Tr4wMo5f9NLPwKKGDtDa7/view?usp=sharing) and place it in `pretrained_models/` directory.

---

## Basic Usage

### Single Video Processing

```bash
python video_processor.py \
    -i video.mp4 \
    -o output/ \
    --config balanced
```

**Output**:
```
output/
©À©¤©¤ analysis_report.txt       # Detailed text report
©À©¤©¤ analysis_results.json     # Structured JSON results
©¸©¤©¤ visualizations/
    ©À©¤©¤ frame_0000.png
    ©À©¤©¤ temporal_dynamics.png
    ©¸©¤©¤ camera_compensation.png
```

### Batch Processing

```bash
python video_processor.py \
    -i videos/ \
    -o batch_output/ \
    --batch \
    --config balanced
```

**Output**:
```
batch_output/
©À©¤©¤ batch_summary.txt         # Summary report
©À©¤©¤ batch_summary.json        # JSON summary
©À©¤©¤ video1/
©¦   ©À©¤©¤ analysis_report.txt
©¦   ©¸©¤©¤ analysis_results.json
©¸©¤©¤ video2/
    ©¸©¤©¤ ...
```

---

## Understanding the Scores

### Unified Dynamics Score (0-1)

The system outputs a **single unified score** that reflects the overall motion intensity:

#### For Static Scenes (Buildings/Still Objects)

| Score Range | Level | Meaning | Examples |
|-------------|-------|---------|----------|
| 0.00-0.15 | Pure Static | Perfectly still | Buildings, sculptures |
| 0.15-0.35 | Low Dynamic | Slight vibration | Flags, leaves in breeze |
| 0.35-0.60 | Medium Dynamic | Noticeable motion | Swaying objects |
| 0.60-0.85 | High Dynamic | Abnormal motion | Strong wind, vibration |
| 0.85-1.00 | Extreme Dynamic | Severe anomaly | Equipment failure |

#### For Dynamic Scenes (People/Animals)

| Score Range | Level | Meaning | Examples |
|-------------|-------|---------|----------|
| 0.00-0.15 | Pure Static | Almost no motion | Standing still, sitting |
| 0.15-0.35 | Low Dynamic | Slight motion | Slow movement, gestures |
| 0.35-0.60 | Medium Dynamic | Normal motion | Walking, daily activities |
| 0.60-0.85 | High Dynamic | Active motion | Running, dancing |
| 0.85-1.00 | Extreme Dynamic | Intense motion | Fast dance, sports |

### Scene Type Detection

The system automatically detects:
- **Static Scene**: Main subject is static (buildings, landscapes)
- **Dynamic Scene**: Main subject is dynamic (people, animals)

---

## Advanced Usage

### Configuration Presets

Three built-in presets for different use cases:

```bash
# Strict mode - Higher quality requirements
python video_processor.py -i videos/ -o output/ --batch --config strict

# Balanced mode - Default, recommended
python video_processor.py -i videos/ -o output/ --batch --config balanced

# Lenient mode - More permissive
python video_processor.py -i videos/ -o output/ --batch --config lenient
```

| Preset | Use Case | Characteristics |
|--------|----------|-----------------|
| `strict` | High quality control | Stricter thresholds, filters more videos |
| `balanced` | General purpose | Balanced accuracy and inclusivity |
| `lenient` | Accept more videos | Looser thresholds, filters fewer videos |

### Quality Filtering

#### Example: Filter low-motion videos in dynamic scenes

```python
from video_processor import batch_process_videos
from video_quality_filter import VideoQualityFilter

# Batch process
processor = VideoProcessor(config_preset='balanced')
results = batch_process_videos(processor, 'videos/', 'output/', 60.0)

# Filter low-dynamic videos in dynamic scenes
quality_filter = VideoQualityFilter()
low_dynamic_videos = quality_filter.filter_low_dynamics_in_dynamic_scenes(
    results,
    threshold=0.3  # Videos with score < 0.3 will be filtered
)

print(f"Found {len(low_dynamic_videos)} low-motion videos:")
for video in low_dynamic_videos:
    print(f"  {video['video_name']}: score={video['score']:.3f}")
    print(f"    {video['reason']}")
```

#### Example: Get quality statistics

```python
from video_quality_filter import VideoQualityFilter

quality_filter = VideoQualityFilter()
stats = quality_filter.get_quality_statistics(results)

print(f"Total videos: {stats['total_videos']}")
print(f"Mean score: {stats['score_statistics']['mean']:.3f}")
print(f"Scene distribution: {stats['scene_type_distribution']}")
print(f"Category distribution: {stats['category_distribution']}")
```

---

## Command Line Arguments

### Basic Parameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `-i, --input` | string | **Required** | Input video file or directory |
| `-o, --output` | string | `output` | Output directory |
| `-m, --raft_model` | string | `pretrained_models/raft-things.pth` | RAFT model path |
| `--device` | string | `cuda` | Computing device (cuda/cpu) |
| `--batch` | flag | False | Batch processing mode |
| `--config` | string | `balanced` | Preset config (strict/balanced/lenient) |

### Advanced Parameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--max_frames` | int | None | Max frames to process |
| `--frame_skip` | int | 1 | Frame skip interval |
| `--fov` | float | 60.0 | Camera field of view (degrees) |
| `--no-viz` | flag | False | Disable visualization |
| `--no-camera-comp` | flag | False | Disable camera compensation |

---

## Python API

### Basic Usage

```python
from video_processor import VideoProcessor

# Create processor
processor = VideoProcessor(
    raft_model_path="pretrained_models/raft-things.pth",
    device='cuda',
    enable_camera_compensation=True,
    config_preset='balanced'
)

# Load and process video
frames = processor.load_video("video.mp4")
result = processor.process_video(frames, output_dir='output/')

# Access results
print(f"Score: {result['unified_dynamics']['unified_dynamics_score']:.3f}")
print(f"Scene: {result['unified_dynamics']['scene_type']}")
print(f"Category: {result['dynamics_classification']['category']}")
```

### Batch Processing with Filtering

```python
from video_processor import batch_process_videos
from video_quality_filter import VideoQualityFilter

# Batch process
processor = VideoProcessor(config_preset='balanced')
results = batch_process_videos(processor, 'videos/', 'output/', 60.0)

# Filter videos
quality_filter = VideoQualityFilter()

# Filter 1: Low-motion in dynamic scenes
low_videos = quality_filter.filter_low_dynamics_in_dynamic_scenes(
    results, threshold=0.3
)

# Filter 2: High-anomaly in static scenes
anomaly_videos = quality_filter.filter_high_static_anomaly(
    results, threshold=0.5
)

# Filter 3: By score range
range_videos = quality_filter.filter_by_score_range(
    results, min_score=0.2, max_score=0.4
)
```

### Custom Configuration

```python
from dynamics_config import get_config

# Load and modify config
config = get_config('balanced')
config['detection']['static_threshold'] = 0.0015  # More strict

# Create processor with custom settings
processor = VideoProcessor(config_preset='balanced')
processor.dynamics_calculator.static_threshold = 0.0015
```

---

## Output Format

### JSON Results Structure

```json
{
  "unified_dynamics_score": 0.52,
  "scene_type": "dynamic",
  "classification": {
    "category": "medium_dynamic",
    "description": "Medium Dynamic",
    "typical_examples": ["Walking", "Daily activities"]
  },
  "temporal_stats": {
    "mean_score": 0.025,
    "temporal_stability": 0.95,
    "mean_static_ratio": 0.65,
    "mean_dynamic_ratio": 0.35
  },
  "confidence": 0.95
}
```

---

## System Architecture

### Core Components

```
video_processor.py              # Main processor
©À©¤©¤ Coordinates all modules
©À©¤©¤ Handles batch processing
©¸©¤©¤ Generates reports

unified_dynamics_calculator.py  # Core algorithm
©À©¤©¤ Detects static/dynamic regions
©À©¤©¤ Auto-classifies scene type
©¸©¤©¤ Outputs unified 0-1 score

video_quality_filter.py         # Quality control
©À©¤©¤ Filters low-motion videos
©À©¤©¤ Filters high-anomaly videos
©¸©¤©¤ Generates statistics

dynamics_config.py              # Configuration
©À©¤©¤ Three presets
©À©¤©¤ Threshold parameters
©¸©¤©¤ Easy customization
```

### Processing Pipeline

```
Video Frames
    ¡ý
Optical Flow (RAFT)
    ¡ý
Camera Motion Compensation
    ¡ý
Residual Flow
    ¡ý
Unified Dynamics Calculator
  ©À¡ú Detect static regions
  ©À¡ú Detect dynamic regions
  ©À¡ú Classify scene type
  ©¸¡ú Calculate unified score
    ¡ý
Quality Filter (Optional)
    ¡ý
Results & Reports
```

---

## Use Cases

### Use Case 1: Filter Low-Motion Videos

**Scenario**: You have a batch of videos with people, but some actors barely moved.

```python
from video_processor import batch_process_videos
from video_quality_filter import VideoQualityFilter

processor = VideoProcessor(config_preset='strict')
results = batch_process_videos(processor, 'videos/', 'output/', 60.0)

quality_filter = VideoQualityFilter()
low_motion = quality_filter.filter_low_dynamics_in_dynamic_scenes(
    results, threshold=0.3
)

print(f"Found {len(low_motion)} videos with insufficient motion")
```

### Use Case 2: Assess Static Scene Quality

**Scenario**: You have videos of buildings shot with a moving camera, and want to check if camera stabilization worked.

```python
processor = VideoProcessor(
    enable_camera_compensation=True,
    config_preset='balanced'
)

frames = processor.load_video("building.mp4")
result = processor.process_video(frames)

if result['unified_dynamics']['scene_type'] == 'static':
    score = result['unified_dynamics']['unified_dynamics_score']
    if score < 0.15:
        print("Perfect stabilization!")
    elif score < 0.35:
        print("Good quality with minor residuals")
    else:
        print("Poor stabilization, check camera setup")
```

### Use Case 3: Quality Control Pipeline

```python
from video_quality_filter import VideoQualityFilter

processor = VideoProcessor(config_preset='strict')
results = batch_process_videos(processor, 'videos/', 'output/', 60.0)

quality_filter = VideoQualityFilter()

# Get statistics
stats = quality_filter.get_quality_statistics(results)
print(f"Mean score: {stats['score_statistics']['mean']:.3f}")
print(f"Scene distribution: {stats['scene_type_distribution']}")

# Filter problematic videos
low_dynamic = quality_filter.filter_low_dynamics_in_dynamic_scenes(results, 0.3)
high_anomaly = quality_filter.filter_high_static_anomaly(results, 0.5)

print(f"\nProblematic videos:")
print(f"  Low motion in dynamic scenes: {len(low_dynamic)}")
print(f"  High anomaly in static scenes: {len(high_anomaly)}")

# Export filtered list
with open('low_motion_videos.txt', 'w') as f:
    for v in low_dynamic:
        f.write(f"{v['video_name']}: {v['score']:.3f}\n")
```

---

## Configuration Guide

### Preset Comparison

| Parameter | Strict | Balanced | Lenient |
|-----------|--------|----------|---------|
| Static threshold | 0.0015 | 0.002 | 0.003 |
| Subject threshold | 0.004 | 0.005 | 0.008 |
| Low dynamic filter | 0.35 | 0.30 | 0.20 |
| High anomaly filter | 0.40 | 0.50 | 0.60 |

### When to Use Each Preset

- **Strict**: Quality control for production, want to filter more videos
- **Balanced**: General purpose, good for most scenarios (default)
- **Lenient**: Accept more videos, exploratory analysis

### Customizing Thresholds

```python
from dynamics_config import get_config

config = get_config('balanced')

# View current settings
print(config['detection']['static_threshold'])    # 0.002
print(config['detection']['subject_threshold'])   # 0.005

# Modify as needed
processor = VideoProcessor(config_preset='balanced')
processor.dynamics_calculator.static_threshold = 0.0015  # More strict
```

---

## Understanding the Results

### Key Metrics

1. **unified_dynamics_score** (0-1)
   - The main output score
   - 0 = completely static
   - 1 = extremely dynamic

2. **scene_type** ('static' or 'dynamic')
   - Auto-detected scene classification
   - 'static': Buildings, landscapes, still objects
   - 'dynamic': People, animals, moving subjects

3. **classification**
   - Category: pure_static, low_dynamic, medium_dynamic, high_dynamic, extreme_dynamic
   - Description and typical examples

4. **confidence** (0-1)
   - Reliability of the assessment
   - Based on temporal stability

### Example Results

#### Static Scene (Building)
```json
{
  "unified_dynamics_score": 0.08,
  "scene_type": "static",
  "classification": {
    "category": "pure_static",
    "description": "Pure Static"
  },
  "confidence": 0.95
}
```

#### Dynamic Scene (Person Walking)
```json
{
  "unified_dynamics_score": 0.52,
  "scene_type": "dynamic",
  "classification": {
    "category": "medium_dynamic",
    "description": "Medium Dynamic"
  },
  "confidence": 0.92
}
```

#### Dynamic Scene (Person Almost Still) - Can be filtered!
```json
{
  "unified_dynamics_score": 0.18,
  "scene_type": "dynamic",
  "classification": {
    "category": "low_dynamic",
    "description": "Low Dynamic"
  },
  "confidence": 0.88
}
```

---

## Quality Filtering

### Filter Low-Motion Videos

Identify videos where subjects have insufficient motion:

```python
from video_quality_filter import VideoQualityFilter

quality_filter = VideoQualityFilter()

# Filter dynamic scenes with score < 0.3
low_motion_videos = quality_filter.filter_low_dynamics_in_dynamic_scenes(
    results,
    threshold=0.3
)

# Generate report
report = quality_filter.generate_filter_report(
    results,
    low_motion_videos,
    "Low Motion Filter"
)
```

### Filter Anomaly Videos

Identify static scenes with excessive residual motion:

```python
# Filter static scenes with score > 0.5
anomaly_videos = quality_filter.filter_high_static_anomaly(
    results,
    threshold=0.5
)
```

---

## Project Structure

```
AIGC_detector/
©À©¤©¤ video_processor.py              # Main processor (refactored)
©À©¤©¤ unified_dynamics_calculator.py  # Core dynamics calculator
©À©¤©¤ video_quality_filter.py         # Quality filtering tools
©À©¤©¤ dynamics_config.py              # Configuration management
©À©¤©¤ simple_raft.py                  # RAFT optical flow wrapper
©À©¤©¤ badcase_detector.py             # BadCase detection
©À©¤©¤ dynamic_motion_compensation/
©¦   ©¸©¤©¤ camera_compensation.py      # Camera motion compensation
©À©¤©¤ pretrained_models/
©¦   ©¸©¤©¤ raft-things.pth            # RAFT model
©À©¤©¤ example_new_system.py          # Usage examples
©À©¤©¤ test_simple.py                 # Quick tests
©À©¤©¤ NEW_SYSTEM_GUIDE.md            # Detailed guide
©¸©¤©¤ README.md                      # This file
```

---

## Examples

See `example_new_system.py` for comprehensive examples:

```bash
python example_new_system.py
```

Examples include:
1. Single video processing
2. Batch processing with filtering
3. Configuration comparison
4. Quality statistics

---

## Frequently Asked Questions

### Q: How to filter "dynamic scenes with low motion"?

```python
low_videos = quality_filter.filter_low_dynamics_in_dynamic_scenes(
    results, 
    threshold=0.3  # Adjust as needed
)
```

### Q: What's the difference between scene types?

- **Static scene**: Camera moves around static objects (buildings)
  - Focuses on residual flow after camera compensation
  - Low scores indicate good stabilization
  
- **Dynamic scene**: Subject moves (people/animals)
  - Focuses on subject motion intensity
  - Low scores indicate insufficient motion

### Q: How to adjust system sensitivity?

Use different presets or modify thresholds:
- More strict: `--config strict` or lower thresholds
- More lenient: `--config lenient` or raise thresholds

### Q: Can I use this without camera compensation?

Yes! Use `--no-camera-comp` flag:
```bash
python video_processor.py -i video.mp4 -o output/ --no-camera-comp
```

---

## Technical Details

### Camera Compensation

The system uses homography-based camera motion compensation:
1. Detect feature points in consecutive frames
2. Estimate camera motion via RANSAC homography
3. Subtract camera motion from optical flow
4. Analyze residual flow for subject dynamics

### Resolution Normalization

All flow magnitudes are normalized by image diagonal length:
```python
diagonal = sqrt(height^2 + width^2)
normalized_flow = flow_magnitude / diagonal
```

This ensures fair comparison across different video resolutions.

### Scene Type Detection

Automatic scene classification based on:
- Dynamic region ratio
- Motion intensity
- Static region residual
- Multi-factor weighted decision

---

## Performance

### Typical Processing Speed

| Resolution | GPU (RTX 3090) | CPU (i7-10700K) |
|-----------|----------------|-----------------|
| 720p | ~15 FPS | ~2 FPS |
| 1080p | ~8 FPS | ~1 FPS |
| 4K | ~3 FPS | ~0.3 FPS |

### Memory Requirements

| Resolution | GPU Memory | RAM |
|-----------|------------|-----|
| 720p | ~2GB | ~4GB |
| 1080p | ~3GB | ~6GB |
| 4K | ~6GB | ~12GB |

---

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce max frames
python video_processor.py -i video.mp4 --max_frames 100

# Use CPU
python video_processor.py -i video.mp4 --device cpu
```

### Low Confidence Scores

- Check if video has motion blur
- Verify camera compensation is working
- Try different config presets

### All Videos Classified as Same Type

- Adjust detection thresholds in config
- Check if videos are actually similar
- Review camera compensation effectiveness

---

## Citation

If you use this system in your research, please cite:

```bibtex
@software{aigc_dynamics_2024,
  title={AIGC Video Dynamics Assessment System},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo}
}
```

---

## License

MIT License - See LICENSE file for details

---

## Changelog

### Version 2.0 (Current)
- Complete system refactoring
- Unified 0-1 scoring standard
- Dual-mode analysis (static + dynamic)
- Quality filtering capabilities
- Simplified architecture
- Removed backward compatibility code

### Version 1.0
- Initial release
- Static object dynamics analysis
- Camera compensation
- Basic visualization

---

## Support

For help and documentation:
- Read `NEW_SYSTEM_GUIDE.md` for detailed guide
- Run `python test_simple.py` for quick validation
- Check `example_new_system.py` for usage examples
- Run `python dynamics_config.py` for configuration options

---

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## Acknowledgments

- RAFT optical flow: [Princeton-VL/RAFT](https://github.com/princeton-vl/RAFT)
- Camera compensation techniques from computer vision literature

---

## Documentation Navigation

### Quick Access
- ? **[Quick Start Guide](QUICK_START_V2.md)** - Get started in 5 minutes
- ? **[System Overview](SYSTEM_OVERVIEW.md)** - Understand how it works
- ? **[Usage Checklist](CHECKLIST.md)** - Step-by-step workflow
- ? **[What's New in v2.0](WHATS_NEW.md)** - See latest changes

### For Developers
- ?? **[Project Structure](PROJECT_STRUCTURE.md)** - File organization
- ? **[Refactoring Details](REFACTORING_COMPLETE.md)** - Technical changes
- ? **[File Changes](FILE_CHANGES.md)** - Detailed changelog

### Lost? Start Here
- ?? **[Documentation Index](DOCS_INDEX.md)** - Find what you need

---

**Note**: This is the refactored v2.0 system with simplified architecture and enhanced capabilities. All backward compatibility code has been removed for a cleaner codebase.
