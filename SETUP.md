# Bad Apple Image Transformer - Setup Complete!

## ✓ What Was Done

1. **Created main.py** - Complete Python script that:
   - Loads any image and resizes it to 480x360 (Bad Apple resolution)
   - Processes all 6572 video frames from the `frames/` directory
   - Extracts silhouettes from each Bad Apple frame
   - Applies those silhouettes to your input image
   - Compiles all processed frames into an MP4 video

2. **Extracted Video Frames** - All 6572 frames from badapple.mp4:
   - Extracted to `frames/` directory
   - Each frame is a PNG file
   - Resolution: 480x360 @ 30 FPS

3. **Updated Documentation**:
   - README.md with complete usage guide
   - examples.py with 8 different usage examples
   - requirements.txt with necessary dependencies

## 🚀 Quick Start

### Option 1: Create video from command line

```bash
# Basic usage (uses default blur=15, threshold=127)
python main.py your_image.jpg -o output.mp4

# Customized version
python main.py portrait.jpg -o portrait_badapple.mp4 --blur 20 --threshold 100

# Test single frame first
python main.py image.jpg --frame 3286 -o test_frame.png
```

### Option 2: Use Python code

```python
from main import BadAppleTransformer

transformer = BadAppleTransformer()
transformer.transform_to_video(
    image_path="image.jpg",
    output_filename="video.mp4",
    blur=15,
    threshold=127
)
```

## 📊 How It Works

1. **Input Image** → Resized to 480x360
2. **Each Frame** of Bad Apple:
   - Extracted as silhouette (black & white)
   - Applied as mask to input image
   - Compiled into video
3. **Output Video** → MP4 (3.6 minutes @ 30 FPS)

## 🎛️ Parameters

### --blur (default: 15)
- **Lower (5-9)**: Sharper silhouettes, more details
- **Higher (21-29)**: Smoother, more artistic

### --threshold (default: 127)
- **Lower (50-100)**: More white, emphasize bright areas
- **Higher (150-200)**: More black, darker silhouettes

### --fps (default: 30)
- Video frame rate
- 60 for smoother playback

## 📁 File Structure

```
badappleinator/
├── main.py              ← Main script
├── frames/              ← 6572 extracted frames
│   ├── frame_000000.png
│   ├── frame_000001.png
│   ├── ...
│   └── frame_006571.png
├── output/              ← Generated videos (created on first run)
│   └── output.mp4
├── README.md            ← Full documentation
├── examples.py          ← Usage examples
├── requirements.txt     ← Dependencies
└── badapple.mp4        ← Original video
```

## 🔧 Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python main.py --help
```

## 💡 Example Usage

### Basic transformation
```bash
python main.py photo.jpg -o badapple.mp4
```

### Smooth, artistic version
```bash
python main.py photo.jpg -o artistic.mp4 --blur 25
```

### Sharp, detailed version
```bash
python main.py photo.jpg -o detailed.mp4 --blur 5
```

### High contrast silhouettes
```bash
python main.py photo.jpg -o highcontrast.mp4 --threshold 50
```

### High FPS for smooth playback
```bash
python main.py photo.jpg -o smooth.mp4 --fps 60
```

### Test before creating full video
```bash
# This creates one PNG file instead of full video
python main.py photo.jpg --frame 1000 -o test.png
```

## ⚙️ Full Command Reference

```bash
python main.py IMAGE -o OUTPUT [OPTIONS]

Positional:
  IMAGE              Input image file (required)

Options:
  -o, --output       Output filename (required, e.g., video.mp4)
  --frame NUM        Transform only single frame (outputs PNG)
  --blur BLUR        Blur strength 1-31 odd (default: 15)
  --threshold THRESH Threshold 0-255 (default: 127)
  --fps FPS          Video FPS (default: 30)
  -d, --dir DIR      Output directory (default: output)
  --no-progress      Hide progress bar
  -h, --help         Show help message
```

## 📈 Performance

- **Processing time**: Depends on your CPU
- **Output size**: ~50-100MB MP4 video (depends on image)
- **Memory**: Minimal (images resized to 480x360)

## 🎬 Output Details

- **Format**: MP4 video
- **Codec**: H.264
- **Resolution**: 480x360 (Bad Apple native)
- **Duration**: ~3.6 minutes (6572 frames @ 30 fps)
- **Colors**: Your image animated with Bad Apple silhouettes

## 🐛 Troubleshooting

### "Frames directory not found"
- Ensure `frames/` folder exists
- Should contain 6572 PNG files named `frame_000000.png` through `frame_006571.png`

### "Could not open video writer"
- May need ffmpeg installed for MP4 writing
- Try installing: `pip install opencv-contrib-python`

### Video is too slow
- Use `--fps 60` for smoother playback
- Or reduce resolution in source image

## 📝 Notes

- Input images are automatically resized to 480x360
- Video will be ~219 seconds (3.6 minutes) long at 30 FPS
- The script preserves image colors - silhouettes are applied as masks
- Each frame processing is very fast (~0.1 seconds)

## 🎯 Next Steps

1. Find an image you want to transform
2. Run: `python main.py your_image.jpg -o output.mp4`
3. Wait for processing (10-15 minutes typical)
4. Watch the result: `output/output.mp4`
5. Adjust blur/threshold parameters if needed

Enjoy your Bad Apple videos! 🎨✨
