-#!/usr/bin/env python3
import cv2
import numpy as np
import matplotlib.pyplot as plt

def analyze_image_structure():
    """Analyze image structure to understand character arrangement"""
    # Read original ROI image
    roi = cv2.imread('output/2_roi.png')
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Read binary character image
    binary = cv2.imread('output/6_binary_chars.png', cv2.IMREAD_GRAYSCALE)
    
    print("=== Image Analysis ===")
    print(f"ROI Dimensions: {roi.shape}")
    print(f"Binary Image Dimensions: {binary.shape}")
    
    # Display a portion of the image
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Portion of original ROI
    axes[0, 0].imshow(roi_gray[1900:2100, 900:1100], cmap='gray')
    axes[0, 0].set_title("ROI Region (200x200)")
    axes[0, 0].axis('off')
    
    # 2. Corresponding portion of binary image
    axes[0, 1].imshow(binary[1900:2100, 900:1100], cmap='gray')
    axes[0, 1].set_title("Binary Region")
    axes[0, 1].axis('off')
    
    # 3. Zoom in on a small region
    small_region = binary[1950:2000, 980:1020]
    axes[0, 2].imshow(small_region, cmap='gray')
    axes[0, 2].set_title("Zoom Region (50x40)")
    axes[0, 2].axis('off')
    
    # 4. Horizontal projection
    horizontal_proj = np.sum(binary > 0, axis=1)
    axes[1, 0].plot(horizontal_proj)
    axes[1, 0].set_xlabel("Row Index")
    axes[1, 0].set_ylabel("White Pixel Count")
    axes[1, 0].set_title("Horizontal Projection")
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Vertical projection
    vertical_proj = np.sum(binary > 0, axis=0)
    axes[1, 1].plot(vertical_proj)
    axes[1, 1].set_xlabel("Column Index")
    axes[1, 1].set_ylabel("White Pixel Count")
    axes[1, 1].set_title("Vertical Projection")
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Connected component analysis
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    print(f"Number of connected components: {num_labels-1}")  # Subtract background
    
    # Filter small components
    min_area = 10
    large_components = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            large_components.append(i)
    
    print(f"Components with area >{min_area}: {len(large_components)}")
    
    # Display connected components
    colored_labels = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
    for i in large_components[:10]:  # Only show first 10
        colored_labels[labels == i] = np.random.randint(0, 255, 3)
    
    axes[1, 2].imshow(colored_labels[1900:2100, 900:1100])
    axes[1, 2].set_title("Connected Components (Colored)")
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('image_analysis.png', dpi=150)
    plt.show()
    
    # Analyze connected component statistics
    if len(large_components) > 0:
        print("\n=== Connected Component Statistics ===")
        areas = [stats[i, cv2.CC_STAT_AREA] for i in large_components]
        widths = [stats[i, cv2.CC_STAT_WIDTH] for i in large_components]
        heights = [stats[i, cv2.CC_STAT_HEIGHT] for i in large_components]
        
        print(f"Area: min={min(areas)}, max={max(areas)}, avg={np.mean(areas):.1f}")
        print(f"Width: min={min(widths)}, max={max(widths)}, avg={np.mean(widths):.1f}")
        print(f"Height: min={min(heights)}, max={max(heights)}, avg={np.mean(heights):.1f}")
        
        # Aspect ratio distribution
        aspect_ratios = [w/h for w, h in zip(widths, heights)]
        print(f"Aspect Ratio: min={min(aspect_ratios):.3f}, max={max(aspect_ratios):.3f}, avg={np.mean(aspect_ratios):.3f}")
        
        # Check for obvious character features
        print("\n=== Character Feature Analysis ===")
        # Dot-matrix characters are usually composed of multiple small connected components
        # Check spatial distribution of components
        centroids_x = [centroids[i][0] for i in large_components]
        centroids_y = [centroids[i][1] for i in large_components]
        
        # Check clustering in horizontal direction
        from collections import Counter
        # Discretize x-coordinates to find columns
        x_bins = np.digitize(centroids_x, bins=np.arange(0, binary.shape[1], 50))
        common_x = Counter(x_bins).most_common(5)
        print(f"Most common X positions: {common_x}")
        
        # Check clustering in vertical direction
        y_bins = np.digitize(centroids_y, bins=np.arange(0, binary.shape[0], 50))
        common_y = Counter(y_bins).most_common(5)
        print(f"Most common Y positions: {common_y}")

if __name__ == "__main__":
    analyze_image_structure()