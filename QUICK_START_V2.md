# Quick Start Guide - v2.0

## Installation (5 minutes)

### 1. Install dependencies
```bash
pip install torch torchvision numpy opencv-python matplotlib tqdm scipy
```

### 2. Download RAFT model
Download [raft-things.pth](https://drive.google.com/file/d/1x1FLCHaGFn_Tr4wMo5f9NLPwKKGDtDa7/view?usp=sharing) to `pretrained_models/`

---

## Basic Usage (3 commands)

### Single Video
```bash
python video_processor.py -i video.mp4 -o output/
```

### Batch Processing
```bash
python video_processor.py -i videos/ -o output/ --batch
```

### With Filtering
```python
from video_processor import batch_process_videos
from video_quality_filter import VideoQualityFilter

processor = VideoProcessor()
results = batch_process_videos(processor, 'videos/', 'output/', 60.0)

filter = VideoQualityFilter()
low_motion = filter.filter_low_dynamics_in_dynamic_scenes(results, 0.3)

print(f"Found {len(low_motion)} low-motion videos")
```

---

## Understanding Your Results

### The Main Score
- **unified_dynamics_score**: 0-1 value
  - 0.0 = completely static
  - 1.0 = extremely dynamic

### For Dynamic Scenes (People/Animals)
- 0.00-0.15: Almost no motion (?? can filter)
- 0.15-0.35: Slight motion (?? can filter)
- 0.35-0.60: Normal motion (? good)
- 0.60-0.85: Active motion (? good)
- 0.85-1.00: Intense motion (? good)

### For Static Scenes (Buildings)
- 0.00-0.15: Perfect stabilization (? good)
- 0.15-0.35: Good quality (? good)
- 0.35-0.60: Noticeable residual (?? check)
- 0.60-0.85: Poor stabilization (? bad)
- 0.85-1.00: Severe anomaly (? bad)

---

## Common Tasks

### Task 1: Find videos where person didn't move enough
```python
from video_quality_filter import VideoQualityFilter

filter = VideoQualityFilter()
low_motion = filter.filter_low_dynamics_in_dynamic_scenes(
    results,
    threshold=0.3  # Adjust as needed
)
```

### Task 2: Check if camera stabilization worked
```python
processor = VideoProcessor(enable_camera_compensation=True)
result = processor.process_video(frames)

if result['unified_dynamics']['scene_type'] == 'static':
    score = result['unified_dynamics']['unified_dynamics_score']
    print(f"Stabilization quality: {score:.3f}")
    # Lower is better for static scenes
```

### Task 3: Batch process with strict quality control
```bash
python video_processor.py \
    -i videos/ \
    -o output/ \
    --batch \
    --config strict
```

---

## Configuration

Three presets available:
- `strict`: Filter more videos (quality control)
- `balanced`: Default (recommended)
- `lenient`: Accept more videos (exploratory)

Use with `--config` flag or in Python:
```python
processor = VideoProcessor(config_preset='strict')
```

---

## Output Files

After processing, you get:
```
output/
©À©¤©¤ analysis_report.txt       # Human-readable report
©À©¤©¤ analysis_results.json     # Machine-readable data
©¸©¤©¤ visualizations/
    ©À©¤©¤ frame_0000.png       # Key frame analysis
    ©À©¤©¤ temporal_dynamics.png # Score over time
    ©¸©¤©¤ camera_compensation.png
```

---

## Troubleshooting

### NumPy version error
```bash
pip install "numpy<2"
```

### CUDA out of memory
```bash
python video_processor.py -i video.mp4 --device cpu
```

### Need faster processing
```bash
python video_processor.py -i video.mp4 --no-viz --max_frames 100
```

---

## Next Steps

1. **Read full documentation**: `README.md`
2. **Try examples**: `python example_usage.py`
3. **Customize config**: Edit `dynamics_config.py`
4. **Check refactoring details**: `REFACTORING_COMPLETE.md`

---

**Ready to use!** The system is production-ready with enhanced capabilities. ?

