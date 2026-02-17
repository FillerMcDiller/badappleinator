#!/usr/bin/env python3
"""
Examples of using BadAppleTransformer to create Bad Apple videos
"""

from main import BadAppleTransformer
from pathlib import Path


def example_1_basic_video():
    """Example 1: Create a basic Bad Apple video from image"""
    print("Example 1: Basic video transformation")
    
    transformer = BadAppleTransformer()
    
    # Transform image to video
    transformer.transform_to_video(
        image_path="image.jpg",
        output_filename="output.mp4"
    )


def example_2_single_frame_test():
    """Example 2: Test with a single frame before creating full video"""
    print("\nExample 2: Test single frame")
    
    transformer = BadAppleTransformer()
    
    # Transform using only one frame to test settings
    transformer.transform_single_frame(
        image_path="image.jpg",
        frame_number=3286,  # Middle of video
        output_filename="test_frame.png"
    )


def example_3_custom_blur():
    """Example 3: Create videos with different blur values"""
    print("\nExample 3: Different blur strengths")
    
    transformer = BadAppleTransformer()
    
    blur_values = [5, 9, 15, 21, 27]
    
    for blur in blur_values:
        print(f"\n  Creating video with blur={blur}...")
        transformer.transform_to_video(
            image_path="image.jpg",
            output_filename=f"badapple_blur_{blur}.mp4",
            blur=blur
        )


def example_4_custom_threshold():
    """Example 4: Create videos with different threshold values"""
    print("\nExample 4: Different thresholds")
    
    transformer = BadAppleTransformer()
    
    thresholds = [50, 100, 127, 150, 200]
    
    for threshold in thresholds:
        print(f"\n  Creating video with threshold={threshold}...")
        transformer.transform_to_video(
            image_path="image.jpg",
            output_filename=f"badapple_threshold_{threshold}.mp4",
            threshold=threshold
        )


def example_5_combined_parameters():
    """Example 5: Combine blur and threshold for different looks"""
    print("\nExample 5: Combined parameter tuning")
    
    transformer = BadAppleTransformer()
    
    styles = {
        "smooth_artistic": {"blur": 25, "threshold": 127},
        "detailed_sharp": {"blur": 7, "threshold": 127},
        "emphasize_dark": {"blur": 15, "threshold": 150},
        "emphasize_light": {"blur": 15, "threshold": 100},
        "high_contrast": {"blur": 9, "threshold": 50},
    }
    
    for style_name, params in styles.items():
        print(f"\n  Creating {style_name}...")
        transformer.transform_to_video(
            image_path="image.jpg",
            output_filename=f"badapple_{style_name}.mp4",
            **params
        )


def example_6_batch_processing():
    """Example 6: Process multiple images"""
    print("\nExample 6: Batch processing")
    
    transformer = BadAppleTransformer()
    
    # Create output subdirectories
    images_dir = Path("images")
    if images_dir.exists():
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        
        for image_path in image_files:
            try:
                print(f"  Processing {image_path.name}...")
                transformer.transform_to_video(
                    image_path=str(image_path),
                    output_filename=f"badapple_{image_path.stem}.mp4"
                )
            except Exception as e:
                print(f"    Error: {e}")


def example_7_high_fps_video():
    """Example 7: Create smoother video with higher FPS"""
    print("\nExample 7: High FPS video")
    
    transformer = BadAppleTransformer()
    
    for fps in [30, 60]:
        print(f"\n  Creating video with {fps} FPS...")
        transformer.transform_to_video(
            image_path="image.jpg",
            output_filename=f"badapple_{fps}fps.mp4",
            fps=fps
        )


def example_8_all_together():
    """Example 8: Complete workflow - test frame then create optimized video"""
    print("\nExample 8: Complete workflow")
    
    transformer = BadAppleTransformer()
    
    print("\n  Step 1: Test with middle frame...")
    transformer.transform_single_frame(
        image_path="image.jpg",
        frame_number=3286,
        output_filename="preview.png",
        blur=15,
        threshold=127
    )
    print("  ✓ Preview saved")
    
    print("\n  Step 2: Create optimized video...")
    transformer.transform_to_video(
        image_path="image.jpg",
        output_filename="badapple_final.mp4",
        blur=15,
        threshold=127,
        fps=30
    )
    print("  ✓ Video created")


if __name__ == '__main__':
    print("Bad Apple Transformer Examples")
    print("=" * 40)
    print("\nUncomment the examples you want to run!")
    print("\nAvailable examples:")
    print("  1. example_1_basic_video() - Create basic video")
    print("  2. example_2_single_frame_test() - Test single frame")
    print("  3. example_3_custom_blur() - Try different blur values")
    print("  4. example_4_custom_threshold() - Try different thresholds")
    print("  5. example_5_combined_parameters() - Test parameter combinations")
    print("  6. example_6_batch_processing() - Process multiple images")
    print("  7. example_7_high_fps_video() - Create 60 FPS videos")
    print("  8. example_8_all_together() - Complete workflow")
    
    # Uncomment one of these to run:
    # example_1_basic_video()
    # example_2_single_frame_test()
    # example_3_custom_blur()
    # example_4_custom_threshold()
    # example_5_combined_parameters()
    # example_6_batch_processing()
    # example_7_high_fps_video()
    # example_8_all_together()
