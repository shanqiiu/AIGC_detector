# Documentation Index - Where to Find What

## ? I Want to Get Started Quickly
�� Read **[QUICK_START_V2.md](QUICK_START_V2.md)** (5 minutes)

---

## ? I Want to Understand the System
�� Read **[SYSTEM_OVERVIEW.md](SYSTEM_OVERVIEW.md)** (10 minutes)

---

## ? I Want Complete Documentation
�� Read **[README.md](README.md)** (20 minutes)

---

## ? I Want to...

### Process a Single Video
```bash
python video_processor.py -i video.mp4 -o output/
```
�� See [README.md - Basic Usage](README.md#basic-usage)

### Process Multiple Videos
```bash
python video_processor.py -i videos/ -o output/ --batch
```
�� See [README.md - Batch Processing](README.md#batch-processing)

### Filter Low-Motion Videos
```python
from video_quality_filter import VideoQualityFilter
filter = VideoQualityFilter()
low_motion = filter.filter_low_dynamics_in_dynamic_scenes(results, 0.3)
```
�� See [README.md - Quality Filtering](README.md#quality-filtering)

### Understand My Scores
�� Read [README.md - Understanding the Scores](README.md#understanding-the-scores)
�� Read [SYSTEM_OVERVIEW.md - Score Interpretation](SYSTEM_OVERVIEW.md#understanding-your-score)

### Customize Configuration
�� Edit [dynamics_config.py](dynamics_config.py)
�� See [README.md - Configuration Guide](README.md#configuration-guide)

### Learn What Changed in v2.0
�� Read **[WHATS_NEW.md](WHATS_NEW.md)**

### See Code Examples
�� Run `python example_usage.py`
�� See [example_usage.py](example_usage.py)

### Troubleshoot Issues
�� Check **[CHECKLIST.md - Troubleshooting](CHECKLIST.md#troubleshooting-checklist)**

---

## ? By Document Type

### Getting Started
1. **[QUICK_START_V2.md](QUICK_START_V2.md)** - 5-minute quick start
2. **[SYSTEM_OVERVIEW.md](SYSTEM_OVERVIEW.md)** - System explanation
3. **[CHECKLIST.md](CHECKLIST.md)** - Step-by-step checklist

### Complete Reference
1. **[README.md](README.md)** - Main documentation
2. **[WHATS_NEW.md](WHATS_NEW.md)** - Version 2.0 changelog
3. **[REFACTORING_COMPLETE.md](REFACTORING_COMPLETE.md)** - Technical details

### Project Information
1. **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - File organization
2. **[FILE_CHANGES.md](FILE_CHANGES.md)** - Detailed change log
3. **[project.uml](project.uml)** - UML class diagram

---

## ? Learning Path

### Beginner (Never used the system)
1. Read [QUICK_START_V2.md](QUICK_START_V2.md)
2. Run `python test_simple.py`
3. Try single video: `python video_processor.py -i video.mp4 -o output/`
4. Read [SYSTEM_OVERVIEW.md](SYSTEM_OVERVIEW.md)

### Intermediate (Used v1.0 before)
1. Read [WHATS_NEW.md](WHATS_NEW.md)
2. Check [FILE_CHANGES.md](FILE_CHANGES.md)
3. Update your code (very simple changes)
4. Read [README.md - Advanced Usage](README.md#advanced-usage)

### Advanced (Want to customize)
1. Read [REFACTORING_COMPLETE.md](REFACTORING_COMPLETE.md)
2. Study `unified_dynamics_calculator.py` source code
3. Edit [dynamics_config.py](dynamics_config.py)
4. Review `video_quality_filter.py` for filtering logic

---

## ? Find Answers to Specific Questions

### Q: How does the scoring work?
�� [README.md - Understanding the Scores](README.md#understanding-the-scores)
�� [SYSTEM_OVERVIEW.md - Understanding Your Score](SYSTEM_OVERVIEW.md#understanding-your-score)

### Q: How to filter videos?
�� [README.md - Quality Filtering](README.md#quality-filtering)
�� [SYSTEM_OVERVIEW.md - Main Use Case](SYSTEM_OVERVIEW.md#main-use-case-filter-low-motion-videos)

### Q: What changed in v2.0?
�� [WHATS_NEW.md](WHATS_NEW.md)
�� [FILE_CHANGES.md](FILE_CHANGES.md)

### Q: Which config preset should I use?
�� [README.md - Configuration Presets](README.md#configuration-presets)
�� [dynamics_config.py](dynamics_config.py) - See actual values

### Q: How to customize thresholds?
�� [README.md - Custom Configuration](README.md#custom-configuration)
�� [REFACTORING_COMPLETE.md - Technical Details](REFACTORING_COMPLETE.md#technical-details)

### Q: What are all the files for?
�� [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
�� [FILE_CHANGES.md](FILE_CHANGES.md)

### Q: How to migrate from v1.0?
�� [WHATS_NEW.md - Migration Guide](WHATS_NEW.md#migration-guide)

### Q: System not working?
�� [CHECKLIST.md - Troubleshooting](CHECKLIST.md#troubleshooting-checklist)

---

## ? Document Purpose Summary

| Document | Purpose | Read Time | When to Read |
|----------|---------|-----------|--------------|
| **QUICK_START_V2.md** | Get started fast | 5 min | First time user |
| **SYSTEM_OVERVIEW.md** | Understand concepts | 10 min | Want overview |
| **README.md** | Complete reference | 20 min | Need details |
| **WHATS_NEW.md** | Version changes | 5 min | Upgrading from v1.0 |
| **CHECKLIST.md** | Step-by-step guide | 10 min | Following workflow |
| **REFACTORING_COMPLETE.md** | Technical details | 15 min | Developer/advanced |
| **PROJECT_STRUCTURE.md** | File organization | 5 min | Understanding layout |
| **FILE_CHANGES.md** | Detailed changelog | 10 min | Want specifics |
| **DOCS_INDEX.md** | This file | 2 min | Lost/confused |

---

## ? Smart Reading Strategy

### If you have 5 minutes
1. Read [QUICK_START_V2.md](QUICK_START_V2.md)
2. Run your first command

### If you have 15 minutes
1. Read [QUICK_START_V2.md](QUICK_START_V2.md)
2. Read [SYSTEM_OVERVIEW.md](SYSTEM_OVERVIEW.md)
3. Run `python example_usage.py`

### If you have 30 minutes
1. Read [QUICK_START_V2.md](QUICK_START_V2.md)
2. Read [SYSTEM_OVERVIEW.md](SYSTEM_OVERVIEW.md)
3. Read [README.md](README.md)
4. Try all examples
5. Process your own videos

### If you're a developer
1. Read [REFACTORING_COMPLETE.md](REFACTORING_COMPLETE.md)
2. Study source code:
   - `unified_dynamics_calculator.py`
   - `video_quality_filter.py`
3. Read [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
4. Customize [dynamics_config.py](dynamics_config.py)

---

## ?? Documentation Map

```
Start Here
    ��
DOCS_INDEX.md (you are here)
    ��
[Choose your path]
    ��
�����������������������������Щ������������������������������������Щ�����������������������������������
��             ��                  ��                 ��
Quick Start   System Overview   Complete Docs    What's New
    ��              ��                 ��                ��
Examples      Concepts          API Reference    Migration
    ��              ��                 ��                ��
Usage         Workflow          Customization    Changes
```

---

## ? Still Need Help?

### Common Paths

**"I'm new and confused"**
�� Start with [QUICK_START_V2.md](QUICK_START_V2.md)

**"I used v1.0 before"**
�� Read [WHATS_NEW.md](WHATS_NEW.md) first

**"I want to customize"**
�� Check [dynamics_config.py](dynamics_config.py) and [README.md - Custom Configuration](README.md#custom-configuration)

**"Something's not working"**
�� See [CHECKLIST.md - Troubleshooting](CHECKLIST.md#troubleshooting-checklist)

**"I need the full API"**
�� Read [README.md - Python API](README.md#python-api)

---

## ? Quick Links

- [Installation](QUICK_START_V2.md#installation-5-minutes)
- [Basic Usage](README.md#basic-usage)
- [Score Interpretation](README.md#understanding-the-scores)
- [Quality Filtering](README.md#quality-filtering)
- [Configuration Presets](README.md#configuration-presets)
- [API Reference](README.md#python-api)
- [Troubleshooting](CHECKLIST.md#troubleshooting-checklist)
- [Examples](example_usage.py)

---

**Start your journey here, find what you need quickly!** ?

