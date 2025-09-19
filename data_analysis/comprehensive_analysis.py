#!/usr/bin/env python3
"""
Comprehensive dataset analysis matching the original visualization format.

Creates detailed dashboards with 12-panel and 6-panel layouts matching
the original analysis visualizations.
"""

import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
import re
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ComprehensiveAnalyzer:
    """Comprehensive analyzer matching original visualization format."""
    
    def __init__(self, labels_path: str, images_root: str):
        """
        Initialize analyzer.
        
        Args:
            labels_path: Path to probe_labels.json
            images_root: Path to probe_images directory
        """
        self.labels_path = Path(labels_path)
        self.images_root = Path(images_root)
        self.labels_data = None
        self.images_data = None
        
        # Filename pattern for parsing metadata
        self.filename_pattern = re.compile(r"([A-Z0-9]+)_(\d+)_(\d+)_1flight_(\d+)_([02])\.jpg")
        
    def load_data(self):
        """Load labels and images data."""
        print("Loading dataset...")
        
        # Load labels
        with open(self.labels_path, 'r') as f:
            self.labels_data = json.load(f)
        
        # Load images metadata
        self.images_data = self.labels_data['images']
        
        print(f"✅ Loaded {len(self.images_data)} images")
        print(f"✅ Loaded {len(self.labels_data['annotations'])} annotations")
        
    def parse_filenames(self):
        """Parse filename patterns and extract metadata."""
        parsed_data = []
        
        for img in self.images_data:
            filename = img['file_name']
            match = self.filename_pattern.match(filename)
            
            if match:
                mission, orbit, pass_, time, variant = match.groups()
                parsed_data.append({
                    'filename': filename,
                    'mission': mission,
                    'orbit': int(orbit),
                    'pass_': int(pass_),
                    'time': int(time),
                    'variant': int(variant),
                    'width': img['width'],
                    'height': img['height']
                })
        
        return pd.DataFrame(parsed_data)
        
    def analyze_basic_stats(self):
        """Analyze basic dataset statistics."""
        # Image statistics
        widths = [img['width'] for img in self.images_data]
        heights = [img['height'] for img in self.images_data]
        
        # Annotation statistics
        bbox_areas = []
        bbox_widths = []
        bbox_heights = []
        bbox_aspect_ratios = []
        relative_areas = []
        
        for ann in self.labels_data['annotations']:
            bbox = ann['bbox']  # [x, y, w, h]
            w, h = bbox[2], bbox[3]
            area = w * h
            aspect_ratio = w / h if h > 0 else 0
            
            bbox_areas.append(area)
            bbox_widths.append(w)
            bbox_heights.append(h)
            bbox_aspect_ratios.append(aspect_ratio)
            
            # Relative area
            img_id = ann['image_id']
            img = next(img for img in self.images_data if img['id'] == img_id)
            img_area = img['width'] * img['height']
            relative_area = area / img_area
            relative_areas.append(relative_area)
        
        return {
            'bbox_areas': bbox_areas,
            'bbox_widths': bbox_widths,
            'bbox_heights': bbox_heights,
            'bbox_aspect_ratios': bbox_aspect_ratios,
            'relative_areas': relative_areas,
            'image_widths': widths,
            'image_heights': heights
        }
        
    def analyze_spatial_distribution(self):
        """Analyze spatial distribution of bounding boxes."""
        relative_x_centers = []
        relative_y_centers = []
        edge_distances = []
        
        for ann in self.labels_data['annotations']:
            img_id = ann['image_id']
            img = next(img for img in self.images_data if img['id'] == img_id)
            bbox = ann['bbox']  # [x, y, w, h]
            
            # Calculate center
            x_center = bbox[0] + bbox[2] / 2
            y_center = bbox[1] + bbox[3] / 2
            
            # Relative positions
            rel_x = x_center / img['width']
            rel_y = y_center / img['height']
            
            relative_x_centers.append(rel_x)
            relative_y_centers.append(rel_y)
            
            # Distance to edges
            dist_to_left = bbox[0]
            dist_to_right = img['width'] - (bbox[0] + bbox[2])
            dist_to_top = bbox[1]
            dist_to_bottom = img['height'] - (bbox[1] + bbox[3])
            
            min_edge_dist = min(dist_to_left, dist_to_right, dist_to_top, dist_to_bottom)
            edge_distances.append(min_edge_dist)
        
        # Quadrant analysis
        quadrants = []
        for rel_x, rel_y in zip(relative_x_centers, relative_y_centers):
            if rel_x < 0.5 and rel_y < 0.5:
                quadrants.append('top_left')
            elif rel_x >= 0.5 and rel_y < 0.5:
                quadrants.append('top_right')
            elif rel_x < 0.5 and rel_y >= 0.5:
                quadrants.append('bottom_left')
            else:
                quadrants.append('bottom_right')
        
        return {
            'relative_x_centers': relative_x_centers,
            'relative_y_centers': relative_y_centers,
            'edge_distances': edge_distances,
            'quadrants': quadrants
        }
        
    def analyze_image_characteristics(self):
        """Analyze image characteristics like brightness, contrast, blur."""
        brightness_values = []
        contrast_values = []
        blur_values = []
        
        print("Analyzing image characteristics...")
        for i, img in enumerate(self.images_data):
            if i % 10 == 0:
                print(f"  Processing image {i+1}/{len(self.images_data)}")
            
            img_path = self.images_root / img['file_name']
            if not img_path.exists():
                continue
            
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate brightness (mean pixel value)
            brightness = np.mean(gray)
            brightness_values.append(brightness)
            
            # Calculate contrast (standard deviation)
            contrast = np.std(gray)
            contrast_values.append(contrast)
            
            # Calculate blur (Laplacian variance)
            blur = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_values.append(blur)
        
        return {
            'brightness': brightness_values,
            'contrast': contrast_values,
            'blur': blur_values
        }
        
    def create_comprehensive_dashboard(self, df, basic_stats, spatial_stats, img_chars):
        """Create the 12-panel comprehensive dashboard."""
        print("\n📈 CREATING 12-PANEL COMPREHENSIVE DASHBOARD")
        print("=" * 60)
        
        output_dir = Path("data_analysis")
        output_dir.mkdir(exist_ok=True)
        
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle('Probe Detection Dataset Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Bounding Box Area Distribution
        axes[0, 0].hist(basic_stats['bbox_areas'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Bounding Box Area Distribution', fontweight='bold')
        axes[0, 0].set_xlabel('Area (pixels²)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Aspect Ratio Distribution
        axes[0, 1].hist(basic_stats['bbox_aspect_ratios'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].set_title('Bounding Box Aspect Ratio Distribution', fontweight='bold')
        axes[0, 1].set_xlabel('Width/Height Ratio')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Probe Position Heatmap
        x_coords = []
        y_coords = []
        for ann in self.labels_data['annotations']:
            bbox = ann['bbox']
            x_center = bbox[0] + bbox[2] / 2
            y_center = bbox[1] + bbox[3] / 2
            x_coords.append(x_center)
            y_coords.append(y_center)
        
        axes[0, 2].hexbin(x_coords, y_coords, gridsize=20, cmap='Blues', alpha=0.8)
        axes[0, 2].set_title('Probe Position Heatmap', fontweight='bold')
        axes[0, 2].set_xlabel('X Coordinate')
        axes[0, 2].set_ylabel('Y Coordinate')
        
        # 4. Relative Area Distribution
        axes[0, 3].hist(basic_stats['relative_areas'], bins=30, alpha=0.7, color='salmon', edgecolor='black')
        axes[0, 3].set_title('Relative Area Distribution', fontweight='bold')
        axes[0, 3].set_xlabel('Bbox Area / Image Area')
        axes[0, 3].set_ylabel('Frequency')
        axes[0, 3].grid(True, alpha=0.3)
        
        # 5. Image Brightness Distribution
        axes[1, 0].hist(img_chars['brightness'], bins=30, alpha=0.7, color='yellow', edgecolor='black')
        axes[1, 0].set_title('Image Brightness Distribution', fontweight='bold')
        axes[1, 0].set_xlabel('Mean Brightness')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 6. Image Contrast Distribution
        axes[1, 1].hist(img_chars['contrast'], bins=30, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 1].set_title('Image Contrast Distribution', fontweight='bold')
        axes[1, 1].set_xlabel('Contrast (Std Dev)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 7. Image Blur Score Distribution
        axes[1, 2].hist(img_chars['blur'], bins=30, alpha=0.7, color='cyan', edgecolor='black')
        axes[1, 2].set_title('Image Blur Score Distribution', fontweight='bold')
        axes[1, 2].set_xlabel('Laplacian Variance')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].grid(True, alpha=0.3)
        
        # 8. Probe Position Distribution (Pie Chart)
        quadrant_counts = Counter(spatial_stats['quadrants'])
        labels = list(quadrant_counts.keys())
        sizes = list(quadrant_counts.values())
        colors = ['lightcoral', 'lightblue', 'lightgreen', 'gold']
        axes[1, 3].pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        axes[1, 3].set_title('Probe Position Distribution', fontweight='bold')
        
        # 9. Bbox Width vs Height
        axes[2, 0].scatter(basic_stats['bbox_widths'], basic_stats['bbox_heights'], 
                          alpha=0.6, color='purple', s=50)
        axes[2, 0].set_title('Bbox Width vs Height', fontweight='bold')
        axes[2, 0].set_xlabel('Width (pixels)')
        axes[2, 0].set_ylabel('Height (pixels)')
        axes[2, 0].grid(True, alpha=0.3)
        
        # 10. File Size Distribution (simulated based on image dimensions)
        file_sizes = []
        for img in self.images_data:
            # Estimate file size based on image dimensions and typical compression
            estimated_size = (img['width'] * img['height'] * 3) / 1000  # Rough estimate in KB
            file_sizes.append(estimated_size)
        
        axes[2, 1].hist(file_sizes, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[2, 1].set_title('File Size Distribution', fontweight='bold')
        axes[2, 1].set_xlabel('File Size (KB)')
        axes[2, 1].set_ylabel('Frequency')
        axes[2, 1].grid(True, alpha=0.3)
        
        # 11. Images per Mission
        mission_counts = df['mission'].value_counts()
        mission_labels = [f"M{i+1}" for i in range(len(mission_counts))]
        axes[2, 2].bar(mission_labels, mission_counts.values, 
                      color='lightblue', alpha=0.7)
        axes[2, 2].set_title('Images per Mission', fontweight='bold')
        axes[2, 2].set_xlabel('Mission ID')
        axes[2, 2].set_ylabel('Image Count')
        axes[2, 2].grid(True, alpha=0.3)
        
        # 12. Min Distance to Image Edge
        axes[2, 3].hist(spatial_stats['edge_distances'], bins=30, alpha=0.7, 
                       color='lightgreen', edgecolor='black')
        axes[2, 3].set_title('Min Distance to Image Edge', fontweight='bold')
        axes[2, 3].set_xlabel('Distance (pixels)')
        axes[2, 3].set_ylabel('Frequency')
        axes[2, 3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'comprehensive_analysis_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 12-panel dashboard saved to {output_dir}/comprehensive_analysis_dashboard.png")
        
    def create_mission_operational_dashboard(self, df):
        """Create the 6-panel mission and operational analysis dashboard."""
        print("\n📈 CREATING 6-PANEL MISSION & OPERATIONAL DASHBOARD")
        print("=" * 60)
        
        output_dir = Path("data_analysis")
        output_dir.mkdir(exist_ok=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Mission and Operational Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # Time Distribution by Mission
        mission_times = {}
        for mission in df['mission'].unique():
            mission_data = df[df['mission'] == mission]
            mission_times[mission] = mission_data['time'].tolist()
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(mission_times)))
        for i, (mission, times) in enumerate(mission_times.items()):
            axes[0, 0].hist(times, bins=20, alpha=0.7, label=mission, color=colors[i])
        axes[0, 0].set_title('Time Distribution by Mission', fontweight='bold')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Orbit vs Pass (colored by variant)
        scatter = axes[0, 1].scatter(df['orbit'], df['pass_'], c=df['variant'], 
                                   cmap='viridis', alpha=0.7, s=50)
        axes[0, 1].set_title('Orbit vs Pass (colored by variant)', fontweight='bold')
        axes[0, 1].set_xlabel('Orbit')
        axes[0, 1].set_ylabel('Pass Number')
        plt.colorbar(scatter, ax=axes[0, 1], label='Variant')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Time vs Camera Variant
        variant_0_times = df[df['variant'] == 0]['time'].tolist()
        variant_2_times = df[df['variant'] == 2]['time'].tolist()
        axes[0, 2].scatter(variant_0_times, [0]*len(variant_0_times), 
                          alpha=0.7, color='blue', label='Variant 0', s=50)
        axes[0, 2].scatter(variant_2_times, [2]*len(variant_2_times), 
                          alpha=0.7, color='orange', label='Variant 2', s=50)
        axes[0, 2].set_title('Time vs Camera Variant', fontweight='bold')
        axes[0, 2].set_xlabel('Time')
        axes[0, 2].set_ylabel('Variant')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Mission Timeline
        for i, (mission, times) in enumerate(mission_times.items()):
            axes[1, 0].plot(times, [i]*len(times), 'o-', alpha=0.7, 
                           label=mission, markersize=4, color=colors[i])
        axes[1, 0].set_title('Mission Timeline', fontweight='bold')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Mission')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Orbit-Pass Coverage Heatmap
        orbit_pass_data = list(zip(df['orbit'], df['pass_']))
        orbit_pass_counts = {}
        for orbit, pass_ in orbit_pass_data:
            key = (orbit, pass_)
            orbit_pass_counts[key] = orbit_pass_counts.get(key, 0) + 1
        
        if orbit_pass_counts:
            orbits, passes, counts = zip(*[(k[0], k[1], v) for k, v in orbit_pass_counts.items()])
            scatter = axes[1, 1].scatter(orbits, passes, c=counts, cmap='Blues', 
                                       s=100, alpha=0.7)
            axes[1, 1].set_title('Orbit-Pass Coverage Heatmap', fontweight='bold')
            axes[1, 1].set_xlabel('Orbit')
            axes[1, 1].set_ylabel('Pass Number')
            plt.colorbar(scatter, ax=axes[1, 1], label='Coverage')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Time Intervals Between Samples
        time_intervals = []
        for mission, times in mission_times.items():
            if len(times) > 1:
                intervals = np.diff(sorted(times))
                time_intervals.extend(intervals)
        
        if time_intervals:
            axes[1, 2].hist(time_intervals, bins=20, alpha=0.7, color='purple', edgecolor='black')
            axes[1, 2].set_title('Time Intervals Between Samples', fontweight='bold')
            axes[1, 2].set_xlabel('Time Interval')
            axes[1, 2].set_ylabel('Frequency')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'mission_operational_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 6-panel dashboard saved to {output_dir}/mission_operational_analysis.png")
        
    def run_comprehensive_analysis(self):
        """Run complete comprehensive analysis."""
        print("🔍 COMPREHENSIVE DATASET ANALYSIS")
        print("=" * 60)
        
        self.load_data()
        df = self.parse_filenames()
        
        print(f"\n📊 Analyzing {len(df)} images...")
        
        # Run analyses
        basic_stats = self.analyze_basic_stats()
        spatial_stats = self.analyze_spatial_distribution()
        img_chars = self.analyze_image_characteristics()
        
        # Create dashboards
        self.create_comprehensive_dashboard(df, basic_stats, spatial_stats, img_chars)
        self.create_mission_operational_dashboard(df)
        
        print("\n🎉 COMPREHENSIVE ANALYSIS COMPLETE!")
        print("=" * 60)
        print("Generated dashboards:")
        print("  📊 comprehensive_analysis_dashboard.png - 12-panel detailed analysis")
        print("  📈 mission_operational_analysis.png - 6-panel operational analysis")


def main():
    """Main analysis function."""
    # Configuration
    labels_path = "data/probe_labels.json"
    images_root = "data/probe_images"
    
    # Check if files exist
    if not Path(labels_path).exists():
        print(f"❌ Labels file not found: {labels_path}")
        return
    
    if not Path(images_root).exists():
        print(f"❌ Images directory not found: {images_root}")
        return
    
    # Run analysis
    analyzer = ComprehensiveAnalyzer(labels_path, images_root)
    analyzer.run_comprehensive_analysis()


if __name__ == "__main__":
    main()