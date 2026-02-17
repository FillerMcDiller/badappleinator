# Bad Apple Image Transformer

Transform **any image into a Bad Apple video** by applying the iconic animation style to it. The script uses extracted Bad Apple video frames to animate your image.

## Features

- 🎬 **Convert images to MP4 videos** with Bad Apple animation
- 🎨 **Apply Bad Apple silhouettes** from the original video to any image
- ⚙️ **Customizable parameters**: blur strength and threshold
- 🖼️ **Single frame transformation** mode for testing
- 📊 **Progress tracking** during video generation

## How It Works

1. Uses extracted Bad Apple video frames (480x360, 6572 frames @ 30 FPS)
2. For each frame, extracts the black and white silhouette
3. Applies that silhouette to your input image
4. Compiles all frames into an MP4 video

Result: Your image animated with the Bad Apple video's iconic silhouettes!

## Setup

### Requirements

- Python 3.6+
- Video frames extracted to `frames/` folder (6572 PNG frames at 480x360)

### Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic: Create Bad Apple video from image

```bash
python main.py image.jpg -o output.mp4
```

This creates a video where your image is animated with Bad Apple's silhouettes!

### Customize blur and threshold

```bash
# Smoother, more artistic silhouettes
python main.py image.jpg -o output.mp4 --blur 20

# More detailed, sharper silhouettes
python main.py image.jpg -o output.mp4 --blur 10

# More white areas
python main.py image.jpg -o output.mp4 --threshold 100

# More black areas
python main.py image.jpg -o output.mp4 --threshold 150
```

### Single frame test

Transform image using only one Bad Apple frame (for testing):

```bash
# Use frame number 3286 (middle of video)
python main.py image.jpg --frame 3286 -o test_frame.png

# Use first frame
python main.py image.jpg --frame 0 -o test_frame_0.png
```

### Advanced parameters

```bash
# Full example
python main.py photo.png -o video.mp4 \
  --blur 16 \
  --threshold 120 \
  --fps 30 \
  -d ./output

# Hide progress bar
python main.py image.jpg -o video.mp4 --no-progress
```

## Command-line Arguments

| Argument | Short | Type | Default | Description |
|----------|-------|------|---------|-------------|
| `image` | - | str | - | Input image path (required) |
| `--output` | `-o` | str | - | Output filename (required, e.g., `video.mp4`) |
| `--frame` | - | int | - | Transform single frame only (frame number) |
| `--blur` | - | int | 15 | Blur strength for silhouette (1-31, must be odd) |
| `--threshold` | - | int | 127 | Threshold value 0-255 (lower=more white) |
| `--fps` | - | int | 30 | Video frame rate |
| `--dir` | `-d` | str | output | Output directory |
| `--no-progress` | - | flag | - | Hide progress bar |

## Parameters Guide

### Blur (1-31, odd numbers only)

- **Lower (5-9)**: Sharper, more detailed silhouettes
- **Medium (13-17)**: Balanced, default Bad Apple look (15)
- **Higher (21-29)**: Smoother, more stylized, less detail

*Default: 15*

### Threshold (0-255)

Controls which pixels become white vs black:

- **Lower (50-100)**: More white areas, emphasizes bright parts
- **Medium (120-135)**: Balanced, matches Bad Apple reference (~85% black)
- **Higher (150-200)**: More black areas, stronger silhouettes

*Default: 127*

## Examples

```bash
# Transform a portrait
python main.py portrait.jpg -o portrait_badapple.mp4

# Smoothed, artistic version
python main.py photo.jpg -o artistic.mp4 --blur 25 --threshold 100

# Sharp, detailed version
python main.py photo.jpg -o detailed.mp4 --blur 7 --threshold 150

# Test with single frame
python main.py image.jpg --frame 1000 -o test.png

# High FPS smooth video
python main.py image.jpg -o video.mp4 --fps 60
```

## Output

- **Format**: MP4 video (H.264 codec)
- **Resolution**: 480x360 (Bad Apple resolution)
- **Duration**: 219 seconds (~3.6 minutes) @ 30 FPS
- **Colors**: Pure black and white with input image colors applied
- **Location**: `output/` directory (or custom `-d` directory)

## Troubleshooting

### "Frames directory not found"

Make sure `frames/` folder exists with extracted video frames. Frames should be named `frame_000000.png`, `frame_000001.png`, etc.

### Video codec issues

The script uses 'mp4v' codec. If this doesn't work on your system, you may need to install ffmpeg or use a different codec.

### Memory issues with large images

The script resizes input images to 480x360. Very large input images shouldn't cause memory issues.

## Performance

- ~6500 frames to process
- Speed depends on your CPU/GPU
- With progress bar, you can see estimated time remaining

## Requirements

- Python 3.6+
- OpenCV (`opencv-python>=4.10.0`)
- NumPy (`numpy>=1.24.3,<2.0`)
- tqdm (`tqdm>=4.60.0`)

## File Structure

```
badappleinator/
  main.py                 # Main script
  frames/                 # Extracted Bad Apple frames (6572 files)
    frame_000000.png
    frame_000001.png
    ...
    frame_006571.png
  output/                 # Generated videos
    video.mp4
    ...
```

## License

Free to use and modify!
