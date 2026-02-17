#!/usr/bin/env python3
"""
Bad Apple Image Transformer
Converts any image into Bad Apple video by applying the video's animation to the image
"""

import cv2
import numpy as np
import argparse
import os
from pathlib import Path
from tqdm import tqdm


class BadAppleTransformer:
    """Transform images into Bad Apple video style using video frames"""
    
    def __init__(self, output_dir="output", frames_dir="frames"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.frames_dir = Path(frames_dir)
        if not self.frames_dir.exists():
            raise ValueError(f"Frames directory not found: {frames_dir}")
        
        # Get list of frame files
        self.frames = sorted(self.frames_dir.glob("frame_*.png"))
        if not self.frames:
            raise ValueError(f"No frames found in {frames_dir}")
        
        # Get video properties from first frame
        first_frame = cv2.imread(str(self.frames[0]))
        self.height, self.width = first_frame.shape[:2]
        
        print(f"Found {len(self.frames)} frames ({self.width}x{self.height})")
    
    def load_image(self, image_path):
        """Load and resize image to match frame dimensions"""
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Resize to frame dimensions
        image = cv2.resize(image, (self.width, self.height))
        return image
    
    def extract_silhouette(self, frame, blur=15, threshold=127):
        """
        Extract a foreground mask from a Bad Apple frame.
        The returned mask always represents the characters/foreground as white.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (blur, blur), 0)
        
        # Apply threshold to get pure black and white
        _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
        
        # Ensure the foreground is white by choosing the smaller area as foreground
        white_pixels = np.count_nonzero(binary == 255)
        total_pixels = binary.size
        if white_pixels > (total_pixels / 2):
            binary = cv2.bitwise_not(binary)
        
        return binary

    def group_by_shade(self, image, bins=4):
        """Group colors by shade while preserving hue/saturation."""
        bins = max(2, min(16, bins))
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        v = hsv[:, :, 2]
        step = 255.0 / (bins - 1)
        v_grouped = np.round(v / step) * step
        hsv[:, :, 2] = np.clip(v_grouped, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def compute_rotation_angle(self, silhouette):
        """Compute a stable rotation angle from the silhouette using contours/ellipse."""
        contours, _ = cv2.findContours(silhouette, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        largest = max(contours, key=cv2.contourArea)
        if len(largest) >= 5:
            ellipse = cv2.fitEllipse(largest)
            angle = ellipse[2]
            return angle - 90.0
        moments = cv2.moments(largest)
        if abs(moments["mu20"] - moments["mu02"]) < 1e-5:
            return 0.0
        angle = 0.5 * np.degrees(np.arctan2(2 * moments["mu11"], moments["mu20"] - moments["mu02"]))
        return angle

    def rotate_image(self, image, angle):
        """Rotate image around its center."""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    def fit_to_mask(self, image, silhouette):
        """Resize and place the image so it fills the silhouette bounds."""
        ys, xs = np.where(silhouette > 0)
        if len(xs) == 0 or len(ys) == 0:
            return image
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        target_w = max(1, x_max - x_min + 1)
        target_h = max(1, y_max - y_min + 1)
        
        # Scale to cover the bounding box
        h, w = image.shape[:2]
        scale = max(target_w / w, target_h / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Center-crop to target size
        x_start = max(0, (new_w - target_w) // 2)
        y_start = max(0, (new_h - target_h) // 2)
        cropped = resized[y_start:y_start + target_h, x_start:x_start + target_w]
        
        # Place into a black canvas
        canvas = np.zeros_like(image)
        canvas[y_min:y_max + 1, x_min:x_max + 1] = cropped
        return canvas

    def ensure_frame_size(self, frame):
        """Ensure the frame matches the target 4:3 size (width x height)."""
        if frame.shape[1] == self.width and frame.shape[0] == self.height:
            return frame
        return cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
    
    def warp_to_silhouette(self, image, ba_frame):
        """
        Warp input image to match the Bad Apple character shape
        Uses optical flow to find motion/deformation between frames
        Preserves full color throughout
        """
        # Convert to grayscale ONLY for optical flow calculation
        gray_input = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_ba = cv2.cvtColor(ba_frame, cv2.COLOR_BGR2GRAY)
        
        # Use optical flow to compute the deformation field
        flow = cv2.calcOpticalFlowFarneback(
            gray_input, gray_ba, 
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        
        # Create coordinate maps
        h, w = image.shape[:2]
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Apply flow to warp coordinates
        map_x = (x + flow[..., 0]).astype(np.float32)
        map_y = (y + flow[..., 1]).astype(np.float32)
        
        # Warp the FULL COLOR image using the flow field
        warped = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        return warped
    
    
    def apply_silhouette(self, warped_image, silhouette):
        """
        Apply Bad Apple silhouette to warped image while preserving colors.
        White areas show image colors, black areas show black background.
        """
        mask = silhouette.astype(np.float32) / 255.0
        result = np.zeros_like(warped_image, dtype=np.float32)
        for i in range(3):
            result[:, :, i] = warped_image[:, :, i].astype(np.float32) * mask
        return result.astype(np.uint8)

    def apply_shade_separation(self, image, silhouette, fg_scale=0.8, bg_scale=1.1):
        """Separate character/background by adjusting brightness while keeping colors."""
        mask = silhouette.astype(np.float32) / 255.0
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        fg = hsv.copy()
        bg = hsv.copy()
        fg[:, :, 2] = np.clip(fg[:, :, 2] * fg_scale, 0, 255)
        bg[:, :, 2] = np.clip(bg[:, :, 2] * bg_scale, 0, 255)
        fg_bgr = cv2.cvtColor(fg.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
        bg_bgr = cv2.cvtColor(bg.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
        
        result = np.zeros_like(image, dtype=np.float32)
        for i in range(3):
            result[:, :, i] = (fg_bgr[:, :, i] * mask) + (bg_bgr[:, :, i] * (1.0 - mask))
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def transform_to_video(self, image_path, output_filename, fps=30, 
                          blur=15, threshold=127, show_progress=True, full_warp=False,
                          frame_limit=None, rotate=True, shade_grouping=True, shade_bins=4):
        """
        Transform image into Bad Apple video with warping
        
        Parameters:
        - image_path: Input image path
        - output_filename: Output video filename  
        - fps: Video frame rate (default: 30, matches Bad Apple)
        - blur: Blur strength for silhouette extraction
        - threshold: Threshold for silhouette extraction
        - show_progress: Show progress bar
        - full_warp: If True, warp the entire image to the video motion; if False, warp only character silhouettes
        - frame_limit: If set, process only the first N frames (useful for testing)
        - rotate: If True, rotate frames to match silhouette orientation
        - shade_grouping: If True, group colors by shade while preserving hues
        - shade_bins: Number of shade bins to group into
        """
        # Load and resize input image
        input_image = self.load_image(image_path)
        
        # Setup video writer
        output_path = self.output_dir / output_filename
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            str(output_path), 
            fourcc, 
            fps, 
            (self.width, self.height)
        )
        
        if not video_writer.isOpened():
            raise ValueError(f"Could not open video writer for {output_path}")
        
        mode_name = "Full Video Warping" if full_warp else "Character Warping"
        print(f"Transforming image into Bad Apple video with {mode_name}...")
        print(f"  Input image: {image_path}")
        print(f"  Output video: {output_path}")
        total_frames = len(self.frames)
        if frame_limit is not None:
            total_frames = min(total_frames, frame_limit)
        print(f"  Processing {total_frames} frames...")
        
        # Process each frame
        frame_list = self.frames[:total_frames]
        iterator = tqdm(frame_list, disable=not show_progress, desc="Processing frames")
        smoothed_angle = 0.0
        
        for frame_path in iterator:
            # Load Bad Apple frame
            ba_frame = cv2.imread(str(frame_path))
            silhouette = self.extract_silhouette(ba_frame, blur, threshold)
            angle = self.compute_rotation_angle(silhouette) if rotate else 0.0
            smoothed_angle = (0.8 * smoothed_angle) + (0.2 * angle)
            
            # Warp input image to match Bad Apple motion
            warped_image = self.warp_to_silhouette(input_image, ba_frame)
            if rotate:
                warped_image = self.rotate_image(warped_image, smoothed_angle)
            if shade_grouping:
                warped_image = self.group_by_shade(warped_image, bins=shade_bins)
            if full_warp:
                # Keep full frame but separate character/background shades
                output_frame = self.apply_shade_separation(warped_image, silhouette)
            else:
                # Mask to character silhouettes only
                fitted = self.fit_to_mask(warped_image, silhouette)
                output_frame = self.apply_silhouette(fitted, silhouette)
            
            # Write to video
            video_writer.write(self.ensure_frame_size(output_frame))
        
        video_writer.release()
        print(f"\n✓ Done! Video saved to: {output_path}")
        return output_path
    
    def transform_single_frame(self, image_path, frame_number, output_filename, 
                              blur=15, threshold=127, full_warp=False,
                              rotate=True, shade_grouping=True, shade_bins=4):
        """Transform image using a single Bad Apple frame with warping"""
        input_image = self.load_image(image_path)
        
        if frame_number >= len(self.frames):
            raise ValueError(f"Frame {frame_number} out of range (max: {len(self.frames)-1})")
        
        # Load Bad Apple frame
        ba_frame = cv2.imread(str(self.frames[frame_number]))
        silhouette = self.extract_silhouette(ba_frame, blur, threshold)
        
        # Warp input image to match Bad Apple motion
        warped_image = self.warp_to_silhouette(input_image, ba_frame)
        if rotate:
            angle = self.compute_rotation_angle(silhouette)
            warped_image = self.rotate_image(warped_image, angle)
        if shade_grouping:
            warped_image = self.group_by_shade(warped_image, bins=shade_bins)
        if full_warp:
            # Keep full frame but separate character/background shades
            output_frame = self.apply_shade_separation(warped_image, silhouette)
        else:
            # Mask to character silhouettes only
            fitted = self.fit_to_mask(warped_image, silhouette)
            output_frame = self.apply_silhouette(fitted, silhouette)
        
        # Save
        output_path = self.output_dir / output_filename
        cv2.imwrite(str(output_path), self.ensure_frame_size(output_frame))
        print(f"Saved to: {output_path}")
        return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Transform any image into Bad Apple video',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create Bad Apple video from image (character warping)
  python main.py image.jpg -o video.mp4
  
  # Full video warping (entire frame warped)
  python main.py image.jpg -o video.mp4 --full-warp
  
  # Test single frame
  python main.py image.jpg --frame 1000 -o test.png
  
  # Test single frame with full warp
  python main.py image.jpg --frame 1000 -o test.png --full-warp
  
  # Customize parameters
  python main.py image.jpg -o video.mp4 --blur 20 --threshold 100 --full-warp
        """
    )
    
    parser.add_argument('image', help='Path to input image')
    parser.add_argument(
        '-o', '--output',
        help='Output filename (required for video mode)'
    )
    parser.add_argument(
        '--frame',
        type=int,
        help='Transform single frame only (specify frame number)'
    )
    parser.add_argument(
        '--blur',
        type=int,
        default=15,
        help='Blur strength for silhouette extraction (default: 15)'
    )
    parser.add_argument(
        '--threshold',
        type=int,
        default=127,
        help='Threshold for silhouette extraction 0-255 (default: 127)'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Video FPS (default: 30)'
    )
    parser.add_argument(
        '-d', '--dir',
        default='output',
        help='Output directory (default: output)'
    )
    parser.add_argument(
        '--no-progress',
        action='store_true',
        help='Hide progress bar'
    )
    parser.add_argument(
        '--full-warp',
        action='store_true',
        help='Warp entire image to match the video motion (no silhouette masking)'
    )
    parser.add_argument(
        '--no-rotate',
        action='store_true',
        help='Disable auto-rotation based on silhouette orientation'
    )
    parser.add_argument(
        '--no-shade-group',
        action='store_true',
        help='Disable shade grouping (use original colors)'
    )
    parser.add_argument(
        '--shade-bins',
        type=int,
        default=4,
        help='Number of shade bins for grouping colors (default: 4)'
    )
    parser.add_argument(
        '--frame-limit',
        type=int,
        help='Only process the first N frames (useful for testing)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return
    
    # Ensure blur is odd
    if args.blur % 2 == 0:
        args.blur += 1
    
    # Validate threshold
    if not 0 <= args.threshold <= 255:
        print("Error: Threshold must be between 0 and 255")
        return
    
    # Initialize transformer
    try:
        transformer = BadAppleTransformer(args.dir)
    except ValueError as e:
        print(f"Error: {e}")
        print("Make sure frames/ directory exists with extracted video frames.")
        print("You may need to run frame extraction first.")
        return
    
    # Single frame mode
    if args.frame is not None:
        if not args.output:
            print("Error: --output required for frame mode")
            return
        try:
            transformer.transform_single_frame(
                args.image, 
                args.frame, 
                args.output,
                args.blur,
                args.threshold,
                full_warp=args.full_warp,
                rotate=not args.no_rotate,
                shade_grouping=not args.no_shade_group,
                shade_bins=args.shade_bins
            )
        except ValueError as e:
            print(f"Error: {e}")
        return
    
    # Video mode
    if not args.output:
        print("Error: --output is required")
        print("Example: python main.py image.jpg -o output.mp4")
        return
    
    try:
        transformer.transform_to_video(
            args.image,
            args.output,
            fps=args.fps,
            blur=args.blur,
            threshold=args.threshold,
            show_progress=not args.no_progress,
            full_warp=args.full_warp,
            frame_limit=args.frame_limit,
            rotate=not args.no_rotate,
            shade_grouping=not args.no_shade_group,
            shade_bins=args.shade_bins
        )
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
