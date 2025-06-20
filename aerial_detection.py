#!/usr/bin/env python3
"""
Aerial Object Detection System
A Python implementation of volumetric reconstruction for detecting small aerial objects 
using 3D motion reconstruction from multiple camera views with OpenCV edge detection.
"""

import numpy as np
import cv2
import json
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

@dataclass
class CameraInfo:
    """Camera metadata structure"""
    camera_id: int
    position: np.ndarray  # [x, y, z] in world coordinates
    rotation: np.ndarray  # [roll, pitch, yaw] in radians
    fov_horizontal: float  # Field of view in radians
    fov_vertical: float
    image_width: int
    image_height: int

@dataclass
class FrameInfo:
    """Frame metadata structure"""
    frame_id: int
    camera_id: int
    timestamp: float
    image_path: str

class MotionDetector:
    """Handles motion detection between consecutive frames with edge enhancement"""
    
    def __init__(self, threshold: float = 30.0, use_edge_enhancement: bool = True):
        self.threshold = threshold
        self.use_edge_enhancement = use_edge_enhancement
        # Background subtractor for better motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
    
    def detect_motion(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect motion between two frames with optional edge enhancement
        Returns: (motion_mask, motion_magnitude)
        """
        if prev_frame.shape != curr_frame.shape:
            raise ValueError("Frame dimensions must match")
        
        # Convert to grayscale if needed
        if len(prev_frame.shape) == 3:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        else:
            prev_gray = prev_frame
            curr_gray = curr_frame
        
        # Method 1: Traditional frame differencing
        diff = cv2.absdiff(prev_gray, curr_gray)
        
        # Method 2: Background subtraction (more robust)
        fg_mask = self.bg_subtractor.apply(curr_frame)
        
        if self.use_edge_enhancement:
            # Edge detection on current frame
            edges_curr = cv2.Canny(curr_gray, 50, 150)
            edges_prev = cv2.Canny(prev_gray, 50, 150)
            
            # Edge-based motion detection
            edge_diff = cv2.absdiff(edges_curr, edges_prev)
            
            # Combine edge motion with regular motion
            # This helps detect small objects with distinct edges
            combined_diff = cv2.addWeighted(diff, 0.7, edge_diff, 0.3, 0)
        else:
            combined_diff = diff
        
        # Apply morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined_diff = cv2.morphologyEx(combined_diff, cv2.MORPH_OPEN, kernel)
        
        # Apply Gaussian blur to reduce noise
        diff_blurred = cv2.GaussianBlur(combined_diff, (5, 5), 0)
        
        # Combine traditional differencing with background subtraction
        motion_mask = (diff_blurred > self.threshold) | (fg_mask == 255)
        
        # Remove small noise blobs
        motion_mask = cv2.morphologyEx(motion_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        
        return motion_mask.astype(bool), diff_blurred.astype(np.float32)

class VoxelGrid:
    """3D voxel grid for accumulating motion data"""
    
    def __init__(self, size: int = 200, voxel_size: float = 1.0, center: np.ndarray = None):
        self.size = size
        self.voxel_size = voxel_size
        self.center = center if center is not None else np.array([0.0, 0.0, 100.0])
        self.grid = np.zeros((size, size, size), dtype=np.float32)
        
        # Calculate grid bounds
        half_extent = (size * voxel_size) / 2.0
        self.min_bounds = self.center - half_extent
        self.max_bounds = self.center + half_extent
    
    def world_to_voxel(self, world_pos: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """Convert world coordinates to voxel indices"""
        if not self._is_inside_bounds(world_pos):
            return None
        
        relative_pos = world_pos - self.min_bounds
        voxel_indices = (relative_pos / self.voxel_size).astype(int)
        
        # Clamp to valid range
        voxel_indices = np.clip(voxel_indices, 0, self.size - 1)
        return tuple(voxel_indices)
    
    def _is_inside_bounds(self, world_pos: np.ndarray) -> bool:
        """Check if world position is inside grid bounds"""
        return np.all(world_pos >= self.min_bounds) and np.all(world_pos <= self.max_bounds)
    
    def add_motion(self, world_pos: np.ndarray, intensity: float):
        """Add motion intensity at world position"""
        voxel_idx = self.world_to_voxel(world_pos)
        if voxel_idx is not None:
            self.grid[voxel_idx] += intensity

class RayCaster:
    """Handles ray casting from camera pixels into 3D space"""
    
    @staticmethod
    def pixel_to_ray_direction(pixel_x: int, pixel_y: int, camera: CameraInfo) -> np.ndarray:
        """Convert pixel coordinates to 3D ray direction in camera space"""
        # Normalize pixel coordinates to [-1, 1]
        norm_x = (2.0 * pixel_x / camera.image_width) - 1.0
        norm_y = (2.0 * pixel_y / camera.image_height) - 1.0
        
        # Calculate ray direction based on FOV
        ray_x = norm_x * np.tan(camera.fov_horizontal / 2.0)
        ray_y = norm_y * np.tan(camera.fov_vertical / 2.0)
        ray_z = 1.0  # Forward direction
        
        # Create direction vector and normalize
        direction = np.array([ray_x, ray_y, ray_z])
        return direction / np.linalg.norm(direction)
    
    @staticmethod
    def camera_to_world_direction(camera_direction: np.ndarray, camera: CameraInfo) -> np.ndarray:
        """Transform camera space direction to world space"""
        # Create rotation matrix from camera orientation
        roll, pitch, yaw = camera.rotation
        
        # Rotation matrices
        R_x = np.array([[1, 0, 0],
                       [0, np.cos(roll), -np.sin(roll)],
                       [0, np.sin(roll), np.cos(roll)]])
        
        R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                       [0, 1, 0],
                       [-np.sin(pitch), 0, np.cos(pitch)]])
        
        R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                       [np.sin(yaw), np.cos(yaw), 0],
                       [0, 0, 1]])
        
        # Combined rotation matrix
        R = R_z @ R_y @ R_x
        
        return R @ camera_direction
    
    def cast_ray_into_voxel_grid(self, camera_pos: np.ndarray, ray_direction: np.ndarray, 
                                voxel_grid: VoxelGrid, max_distance: float = 1000.0, 
                                step_size: float = 0.5) -> List[Tuple[int, int, int]]:
        """Cast ray and return all voxel indices it passes through"""
        voxel_indices = []
        num_steps = int(max_distance / step_size)
        
        for i in range(num_steps):
            distance = i * step_size
            world_pos = camera_pos + distance * ray_direction
            
            voxel_idx = voxel_grid.world_to_voxel(world_pos)
            if voxel_idx is not None:
                voxel_indices.append(voxel_idx)
            elif distance > 50.0:  # Stop if we've gone far past the grid
                break
        
        return voxel_indices

class AerialObjectDetector:
    """Main class for aerial object detection"""
    
    def __init__(self, motion_threshold: float = 25.0, voxel_size: float = 2.0):
        self.motion_detector = MotionDetector(motion_threshold)
        self.ray_caster = RayCaster()
        self.voxel_grid = VoxelGrid(size=150, voxel_size=voxel_size)
        self.cameras = {}
        self.frame_history = {}
        
    def add_camera(self, camera: CameraInfo):
        """Add a camera to the system"""
        self.cameras[camera.camera_id] = camera
        self.frame_history[camera.camera_id] = []
    
    def process_frame(self, frame_info: FrameInfo, image: np.ndarray):
        """Process a single frame for motion detection and voxel updates"""
        camera = self.cameras.get(frame_info.camera_id)
        if camera is None:
            raise ValueError(f"Camera {frame_info.camera_id} not found")
        
        # Get previous frame for this camera
        prev_frames = self.frame_history[frame_info.camera_id]
        
        if len(prev_frames) > 0:
            prev_image = prev_frames[-1]['image']
            
            # Detect motion
            motion_mask, motion_magnitude = self.motion_detector.detect_motion(prev_image, image)
            
            # Process motion pixels
            motion_pixels = np.where(motion_mask)
            
            print(f"Camera {frame_info.camera_id}: Found {len(motion_pixels[0])} motion pixels")
            
            # Cast rays for motion pixels
            for i in range(len(motion_pixels[0])):
                pixel_y, pixel_x = motion_pixels[0][i], motion_pixels[1][i]
                intensity = motion_magnitude[pixel_y, pixel_x]
                
                # Skip weak motion
                if intensity < self.motion_detector.threshold * 1.5:
                    continue
                
                # Get ray direction
                camera_ray = self.ray_caster.pixel_to_ray_direction(pixel_x, pixel_y, camera)
                world_ray = self.ray_caster.camera_to_world_direction(camera_ray, camera)
                
                # Cast ray into voxel grid
                voxel_indices = self.ray_caster.cast_ray_into_voxel_grid(
                    camera.position, world_ray, self.voxel_grid, max_distance=200.0
                )
                
                # Add motion to voxels along ray
                for voxel_idx in voxel_indices:
                    self.voxel_grid.grid[voxel_idx] += intensity * 0.1  # Scale down intensity
        
        # Store current frame
        self.frame_history[frame_info.camera_id].append({
            'frame_info': frame_info,
            'image': image.copy()
        })
        
        # Keep only last 2 frames to save memory
        if len(self.frame_history[frame_info.camera_id]) > 2:
            self.frame_history[frame_info.camera_id].pop(0)
    
    def get_detected_objects(self, threshold_percentile: float = 95.0) -> np.ndarray:
        """Extract high-intensity voxels as detected objects"""
        flat_grid = self.voxel_grid.grid.flatten()
        threshold = np.percentile(flat_grid[flat_grid > 0], threshold_percentile)
        
        object_indices = np.where(self.voxel_grid.grid > threshold)
        object_positions = []
        
        for i in range(len(object_indices[0])):
            voxel_idx = (object_indices[0][i], object_indices[1][i], object_indices[2][i])
            world_pos = self.voxel_grid.min_bounds + np.array(voxel_idx) * self.voxel_grid.voxel_size
            intensity = self.voxel_grid.grid[voxel_idx]
            object_positions.append([world_pos[0], world_pos[1], world_pos[2], intensity])
        
        return np.array(object_positions)
    
    def visualize_detections(self):
        """Visualize detected objects in 3D"""
        objects = self.get_detected_objects()
        
        if len(objects) == 0:
            print("No objects detected")
            return
        
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot detected objects
        if len(objects) > 0:
            scatter = ax.scatter(objects[:, 0], objects[:, 1], objects[:, 2], 
                               c=objects[:, 3], cmap='hot', s=50, alpha=0.8)
            plt.colorbar(scatter, label='Motion Intensity')
        
        # Plot camera positions
        for camera in self.cameras.values():
            ax.scatter(camera.position[0], camera.position[1], camera.position[2], 
                      c='blue', s=200, marker='^', label=f'Camera {camera.camera_id}')
        
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_zlabel('Z (meters)')
        ax.set_title('Aerial Object Detection System - 3D Spatial View')
        ax.legend()
        
        plt.tight_layout()
        plt.show()

class RealTimeAerialDetector:
    """Real-time aerial object detection using webcam or video files"""
    
    def __init__(self, motion_threshold: float = 25.0, voxel_size: float = 2.0):
        self.detector = AerialObjectDetector(motion_threshold, voxel_size)
        self.setup_single_camera()
        
    def setup_single_camera(self):
        """Setup a single camera for demonstration"""
        camera = CameraInfo(
            camera_id=0, 
            position=np.array([0, 0, 0]), 
            rotation=np.array([0, 0.1, 0]),  # Slight upward tilt
            fov_horizontal=np.pi/3, 
            fov_vertical=np.pi/4,
            image_width=640, 
            image_height=480
        )
        self.detector.add_camera(camera)
    
    def run_webcam_detection(self, camera_id: int = 0):
        """Run real-time detection using webcam"""
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        frame_count = 0
        print("Starting real-time detection. Press 'q' to quit, 's' to show 3D visualization")
        print("Move objects in front of the camera to see motion detection...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error reading frame")
                break
            
            # Process frame
            frame_info = FrameInfo(
                frame_id=frame_count,
                camera_id=0,
                timestamp=frame_count / 30.0,  # Assume 30 FPS
                image_path=f"webcam_frame_{frame_count}.jpg"
            )
            
            # Get motion detection results for visualization
            if frame_count > 0:
                camera = self.detector.cameras[0]
                prev_frames = self.detector.frame_history[0]
                
                if len(prev_frames) > 0:
                    prev_image = prev_frames[-1]['image']
                    motion_mask, motion_magnitude = self.detector.motion_detector.detect_motion(prev_image, frame)
                    
                    # Create visualization overlay
                    overlay = self.create_motion_overlay(frame, motion_mask, motion_magnitude)
                    cv2.imshow('Aerial Object Detection System - Real Time', overlay)
            
            # Process frame for 3D reconstruction
            self.detector.process_frame(frame_info, frame)
            
            frame_count += 1
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and frame_count > 10:
                print("Showing 3D visualization...")
                self.detector.visualize_detections()
            elif key == ord('r'):
                print("Resetting detection grid...")
                self.detector.voxel_grid.grid.fill(0)
        
        cap.release()
        cv2.destroyAllWindows()
    
    def create_motion_overlay(self, frame: np.ndarray, motion_mask: np.ndarray, 
                            motion_magnitude: np.ndarray) -> np.ndarray:
        """Create visualization overlay showing motion detection"""
        overlay = frame.copy()
        
        # Highlight motion areas in red
        motion_colored = np.zeros_like(frame)
        motion_colored[:, :, 2] = motion_mask * 255  # Red channel
        
        # Blend with original frame
        overlay = cv2.addWeighted(overlay, 0.7, motion_colored, 0.3, 0)
        
        # Draw motion intensity as contours
        motion_uint8 = (motion_magnitude).astype(np.uint8)
        contours, _ = cv2.findContours(motion_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter small contours (noise) and draw larger ones
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:  # Minimum area threshold
                # Draw bounding box
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Draw contour
                cv2.drawContours(overlay, [contour], -1, (255, 255, 0), 2)
                
                # Add area text
                cv2.putText(overlay, f'Area: {int(area)}', (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add status text
        motion_pixels = np.sum(motion_mask)
        cv2.putText(overlay, f'Motion Pixels: {motion_pixels}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(overlay, 'Press: q=quit, s=show 3D, r=reset', (10, 460), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return overlay
    
    def run_video_file_detection(self, video_path: str):
        """Run detection on a video file"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return
        
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Processing video: {total_frames} frames at {fps} FPS")
        print("Press 'q' to quit, 's' to show 3D visualization")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_info = FrameInfo(
                frame_id=frame_count,
                camera_id=0,
                timestamp=frame_count / fps,
                image_path=f"video_frame_{frame_count}.jpg"
            )
            
            # Resize frame if too large
            if frame.shape[1] > 640:
                scale = 640 / frame.shape[1]
                new_width = 640
                new_height = int(frame.shape[0] * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            # Show progress
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
            
            # Process frame (same as webcam)
            if frame_count > 0:
                camera = self.detector.cameras[0]
                prev_frames = self.detector.frame_history[0]
                
                if len(prev_frames) > 0:
                    prev_image = prev_frames[-1]['image']
                    motion_mask, motion_magnitude = self.detector.motion_detector.detect_motion(prev_image, frame)
                    
                    overlay = self.create_motion_overlay(frame, motion_mask, motion_magnitude)
                    cv2.imshow('Aerial Object Detection System - Video Analysis', overlay)
            
            self.detector.process_frame(frame_info, frame)
            frame_count += 1
            
            # Handle key presses (check every few frames to avoid blocking)
            if frame_count % 5 == 0:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    print("Showing 3D visualization...")
                    self.detector.visualize_detections()
        
        cap.release()
        cv2.destroyAllWindows()
        
        print("Video processing complete!")
        self.detector.visualize_detections()

# Demonstration functions
def demo_webcam():
    """Demo using webcam"""
    print("=== Webcam Demo ===")
    detector = RealTimeAerialDetector(motion_threshold=20.0)
    detector.run_webcam_detection()

def demo_video_file():
    """Demo using video file"""
    print("=== Video File Demo ===")
    video_path = input("Enter path to video file (or press Enter for webcam demo): ").strip()
    
    if not video_path:
        demo_webcam()
        return
    
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        print("Falling back to webcam demo...")
        demo_webcam()
        return
    
    detector = RealTimeAerialDetector(motion_threshold=15.0)
    detector.run_video_file_detection(video_path)

def demo_synthetic():
    """Demo with synthetic data (original simulation)"""
    print("=== Synthetic Demo ===")
    simulate_aerial_object()
    """Create a sample camera setup for testing"""
    detector = AerialObjectDetector(motion_threshold=20.0, voxel_size=1.5)
    
    # Add multiple cameras positioned around the detection area
    cameras = [
        CameraInfo(camera_id=0, position=np.array([0, 0, 0]), 
                  rotation=np.array([0, 0.2, 0]), fov_horizontal=np.pi/3, fov_vertical=np.pi/4,
                  image_width=640, image_height=480),
        CameraInfo(camera_id=1, position=np.array([100, 0, 0]), 
                  rotation=np.array([0, 0.2, np.pi]), fov_horizontal=np.pi/3, fov_vertical=np.pi/4,
                  image_width=640, image_height=480),
        CameraInfo(camera_id=2, position=np.array([50, 100, 0]), 
                  rotation=np.array([0, 0.2, -np.pi/2]), fov_horizontal=np.pi/3, fov_vertical=np.pi/4,
                  image_width=640, image_height=480),
    ]
    
    for camera in cameras:
        detector.add_camera(camera)
    
    return detector

def simulate_aerial_object():
    """Simulate detection of a moving aerial object"""
    detector = create_sample_setup()
    
    # Simulate frames with a moving object
    for frame_num in range(10):
        for camera_id in detector.cameras.keys():
            # Create synthetic image with moving object
            image = np.random.randint(0, 50, (480, 640), dtype=np.uint8)  # Background noise
            
            # Add a moving "object" (bright spot that moves across frames)
            object_x = 200 + frame_num * 20 + camera_id * 50
            object_y = 240 + frame_num * 10
            
            if 0 <= object_x < 640 and 0 <= object_y < 480:
                # Add bright moving object
                cv2.circle(image, (object_x, object_y), 15, 255, -1)
                
                # Add some noise around object
                noise_region = image[max(0, object_y-25):min(480, object_y+25), 
                                   max(0, object_x-25):min(640, object_x+25)]
                noise_region += np.random.randint(0, 100, noise_region.shape).astype(np.uint8)
            
            frame_info = FrameInfo(
                frame_id=frame_num,
                camera_id=camera_id,
                timestamp=frame_num * 0.1,  # 10 FPS
                image_path=f"frame_{frame_num:03d}_cam_{camera_id}.jpg"
            )
            
            detector.process_frame(frame_info, image)
    
    print("Processing complete. Visualizing results...")
    detector.visualize_detections()
    
    # Print detection summary
    objects = detector.get_detected_objects()
    print(f"\nDetected {len(objects)} potential aerial objects")
    if len(objects) > 0:
        print("Object positions (X, Y, Z, Intensity):")
        for i, obj in enumerate(objects[:5]):  # Show first 5
            print(f"  Object {i+1}: ({obj[0]:.1f}, {obj[1]:.1f}, {obj[2]:.1f}) - Intensity: {obj[3]:.1f}")

if __name__ == "__main__":
    print("Aerial Object Detection System - Python Implementation")
    print("============================================================")
    print("\nChoose demo mode:")
    print("1. Webcam real-time detection")
    print("2. Video file analysis")
    print("3. Synthetic simulation")
    
    choice = input("\nEnter choice (1-3) or press Enter for webcam: ").strip()
    
    if choice == "2":
        demo_video_file()
    elif choice == "3":
        demo_synthetic()
    else:
        # Default to webcam
        demo_webcam()
