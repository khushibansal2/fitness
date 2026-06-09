import argparse
import importlib
import json
import math
import time
from typing import Dict, List, Tuple

import cv2
import mediapipe as mp
import numpy as np
from scipy import stats

if not hasattr(mp, 'solutions'):
    try:
        mp.solutions = importlib.import_module('mediapipe.solutions')
    except ModuleNotFoundError:
        try:
            mp.solutions = importlib.import_module('mediapipe.python.solutions')
        except ModuleNotFoundError:
            mp.solutions = None

class EnhancedFitnessAnalyzer:
    def __init__(self):
        self.mp_pose = None
        self.mp_drawing = None
        self.mp_drawing_styles = None
        self.pose = None

        if mp.solutions is not None:
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
        
        # Performance distributions for percentile scoring 
        self.performance_distributions = {
            'situps': {
                'teenage': {'male': [15, 18, 22, 25, 30, 35, 40], 'female': [12, 15, 18, 22, 25, 30, 35], 'other': [13, 16, 20, 23, 27, 32, 37]},
                'youth': {'male': [25, 30, 35, 40, 45, 50, 55], 'female': [20, 25, 28, 32, 36, 40, 45], 'other': [22, 27, 31, 36, 40, 45, 50]},
                'adult': {'male': [20, 25, 30, 35, 40, 45, 50], 'female': [15, 20, 25, 30, 35, 40, 45], 'other': [17, 22, 27, 32, 37, 42, 47]}
            },
            'vertical_jump': {
                'teenage': {'male': [35, 40, 45, 50, 55, 60, 65], 'female': [25, 30, 35, 40, 45, 50, 55], 'other': [30, 35, 40, 45, 50, 55, 60]},
                'youth': {'male': [45, 50, 55, 60, 65, 70, 75], 'female': [35, 40, 45, 50, 55, 60, 65], 'other': [40, 45, 50, 55, 60, 65, 70]},
                'adult': {'male': [40, 45, 50, 55, 60, 65, 70], 'female': [30, 35, 40, 45, 50, 55, 60], 'other': [35, 40, 45, 50, 55, 60, 65]}
            },
            'broad_jump': {
                'teenage': {'male': [90, 100, 110, 120, 130, 140, 150], 'female': [80, 90, 100, 110, 120, 130, 140], 'other': [85, 95, 105, 115, 125, 135, 145]},
                'youth': {'male': [180, 190, 200, 210, 220, 230, 240], 'female': [170, 180, 190, 200, 210, 220, 230], 'other': [175, 185, 195, 205, 215, 225, 235]},
                'adult': {'male': [140, 150, 160, 170, 180, 190, 200], 'female': [130, 140, 150, 160, 170, 180, 190], 'other': [135, 145, 155, 165, 175, 185, 195]}
            },
            'flexibility': {
                'teenage': {'male': [20, 22, 25, 28, 30, 32, 35], 'female': [25, 28, 30, 32, 35, 38, 40], 'other': [22, 25, 27, 30, 32, 35, 37]},
                'youth': {'male': [25, 28, 30, 32, 35, 38, 40], 'female': [30, 32, 35, 38, 40, 42, 45], 'other': [27, 30, 32, 35, 37, 40, 42]},
                'adult': {'male': [20, 22, 25, 28, 30, 32, 35], 'female': [25, 28, 30, 32, 35, 38, 40], 'other': [22, 25, 27, 30, 32, 35, 37]}
            }
        }
        
        # Current person bounding box for tracking
        self.person_bbox = None
        self.bbox_history = []
        self.crop_padding = 0.2  # 20% padding around person
        
    def _ensure_pose_available(self) -> None:
        if self.pose is None or self.mp_pose is None:
            raise RuntimeError('MediaPipe pose detection is not available in this environment')

    def get_person_bbox(self, landmarks, frame_width: int, frame_height: int) -> Tuple[int, int, int, int]:
        """Calculate bounding box around the person with padding"""
        if not landmarks:
            return 0, 0, frame_width, frame_height
            
        # Get all landmark coordinates
        x_coords = [lm.x * frame_width for lm in landmarks.landmark]
        y_coords = [lm.y * frame_height for lm in landmarks.landmark]
        
        # Find bounding box
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Add padding
        width = x_max - x_min
        height = y_max - y_min
        padding_x = width * self.crop_padding
        padding_y = height * self.crop_padding
        
        # Apply padding and ensure within frame bounds
        x_min = max(0, int(x_min - padding_x))
        y_min = max(0, int(y_min - padding_y))
        x_max = min(frame_width, int(x_max + padding_x))
        y_max = min(frame_height, int(y_max + padding_y))
        
        return x_min, y_min, x_max, y_max
    
    def smooth_bbox(self, current_bbox: Tuple[int, int, int, int], alpha: float = 0.7) -> Tuple[int, int, int, int]:
        """Smooth bounding box transitions to avoid jittery crops"""
        if not self.bbox_history:
            self.bbox_history.append(current_bbox)
            return current_bbox
            
        # Exponential smoothing
        prev_bbox = self.bbox_history[-1]
        smoothed_bbox = tuple(int(alpha * curr + (1 - alpha) * prev) 
                             for curr, prev in zip(current_bbox, prev_bbox))
        
        self.bbox_history.append(smoothed_bbox)
        # Keep only recent history
        if len(self.bbox_history) > 10:
            self.bbox_history.pop(0)
            
        return smoothed_bbox
    
    def crop_and_resize_frame(self, frame: np.ndarray, landmarks, target_size: Tuple[int, int] = (640, 480)) -> np.ndarray:
        """Crop frame around person and resize"""
        height, width = frame.shape[:2]
        
        if landmarks:
            # Get person bounding box
            bbox = self.get_person_bbox(landmarks, width, height)
            # Smooth the bounding box
            bbox = self.smooth_bbox(bbox)
            x_min, y_min, x_max, y_max = bbox
            
            # Ensure minimum crop size
            crop_width = x_max - x_min
            crop_height = y_max - y_min
            min_size = min(width, height) // 3
            
            if crop_width < min_size or crop_height < min_size:
                # Use full frame if crop would be too small
                cropped = frame
            else:
                # Crop the frame
                cropped = frame[y_min:y_max, x_min:x_max]
        else:
            cropped = frame
            
        # Resize to target size
        resized = cv2.resize(cropped, target_size)
        return resized
    
    def calculate_percentile_score(self, performance_value: float, test_type: str, age: str, gender: str) -> int:
        """Calculate 0-100 score based on performance percentile"""
        if test_type not in self.performance_distributions:
            return 50  # Default score if no distribution available
            
        try:
            distribution = self.performance_distributions[test_type][age][gender]
        except KeyError:
            return 50  # Default score if category not found
            
        # Calculate percentile using the distribution
        percentile = stats.percentileofscore(distribution, performance_value)
        
        # Convert to 0-100 scale
        score = max(0, min(100, int(percentile)))
        return score
    
    def get_performance_feedback(self, score: int, test_type: str, current_value: float, age: str, gender: str) -> Dict[str, str]:
        """Generate personalized feedback based on performance"""
        feedback = {
            'technique_tips': '',
            'improvement_targets': '',
            'overall_rating': ''
        }
        
        # Overall rating based on score
        if score >= 85:
            feedback['overall_rating'] = 'Excellent'
        elif score >= 70:
            feedback['overall_rating'] = 'Good'
        elif score >= 50:
            feedback['overall_rating'] = 'Average'
        else:
            feedback['overall_rating'] = 'Needs Improvement'
        
        # Test-specific feedback
        if test_type == 'situps':
            if score < 50:
                feedback['technique_tips'] = 'Keep your back straight, engage your core, and avoid pulling on your neck'
                feedback['improvement_targets'] = f'Practice daily - aim for {int(current_value + 5)} situps to improve your score'
            elif score < 70:
                feedback['technique_tips'] = 'Focus on controlled movements and full range of motion'
                feedback['improvement_targets'] = f'Increase to {int(current_value + 3)} situps for better rating'
            else:
                feedback['technique_tips'] = 'Maintain excellent form and consider adding variations'
                feedback['improvement_targets'] = 'Great performance! Try advanced variations to challenge yourself'
                
        elif test_type == 'vertical_jump':
            if score < 50:
                feedback['technique_tips'] = 'Use arm swing, bend knees to 90 degrees, and explode upward'
                feedback['improvement_targets'] = f'Work on leg strength - aim for {current_value + 5:.1f}cm jump height'
            elif score < 70:
                feedback['technique_tips'] = 'Focus on explosive power and landing technique'
                feedback['improvement_targets'] = f'Target {current_value + 3:.1f}cm for score improvement'
            else:
                feedback['technique_tips'] = 'Excellent jumping technique! Maintain consistency'
                feedback['improvement_targets'] = 'Outstanding performance! Focus on consistency'
                
        elif test_type == 'broad_jump':
            if score < 50:
                feedback['technique_tips'] = 'Swing arms back, squat deep, and drive forward with arm swing'
                feedback['improvement_targets'] = f'Build leg power - aim for {current_value + 10:.1f}cm distance'
            elif score < 70:
                feedback['technique_tips'] = 'Work on coordination between arm swing and leg drive'
                feedback['improvement_targets'] = f'Target {current_value + 5:.1f}cm for better rating'
            else:
                feedback['technique_tips'] = 'Great technique! Focus on consistency and power'
                feedback['improvement_targets'] = 'Excellent performance! Maintain this level'
                
        elif test_type == 'flexibility':
            if score < 50:
                feedback['technique_tips'] = 'Stretch daily, warm up before testing, reach gradually without bouncing'
                feedback['improvement_targets'] = f'Daily stretching can help you reach {current_value + 3:.1f}cm'
            elif score < 70:
                feedback['technique_tips'] = 'Hold stretches longer and breathe deeply'
                feedback['improvement_targets'] = f'Aim for {current_value + 2:.1f}cm with consistent stretching'
            else:
                feedback['technique_tips'] = 'Excellent flexibility! Maintain with regular stretching'
                feedback['improvement_targets'] = 'Great flexibility! Keep up the regular stretching routine'
        
        return feedback
    
    def draw_pose_overlay(self, frame: np.ndarray, landmarks, connections=None) -> np.ndarray:
        """Draw pose landmarks and connections on frame"""
        if landmarks:
            # Draw landmarks
            self.mp_drawing.draw_landmarks(
                frame,
                landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        return frame
    
    def calculate_angle(self, p1: List[float], p2: List[float], p3: List[float]) -> float:
        """Calculate angle between three points"""
        a = np.array(p1)
        b = np.array(p2)
        c = np.array(p3)
        
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return angle
    
    def calculate_distance_2d(self, p1: List[float], p2: List[float]) -> float:
        """Calculate 2D distance between two points"""
        return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    
    def smooth_signal(self, data: List[float], window_size: int = 5) -> List[float]:
        """Smooth signal using moving average"""
        smoothed = []
        for i in range(len(data)):
            start = max(0, i - window_size // 2)
            end = min(len(data), i + window_size // 2 + 1)
            smoothed.append(sum(data[start:end]) / (end - start))
        return smoothed
    
    def analyze_situps(self, video_path: str, age: str, gender: str, show_overlay: bool = True) -> Dict:
        """Enhanced sit-ups analysis with auto-crop and scoring"""
        self._ensure_pose_available()
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        situp_count = 0
        stage = "down"
        frame_angles = []
        processed_frames = []
        
        print("Analyzing sit-ups with enhanced features...")
        
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Reset bbox history for new video
        self.bbox_history = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                # Auto-crop and resize frame
                cropped_frame = self.crop_and_resize_frame(frame, results.pose_landmarks)
                
                # Re-process cropped frame for more accurate landmarks
                cropped_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
                cropped_results = self.pose.process(cropped_rgb)
                
                if cropped_results.pose_landmarks:
                    landmarks = cropped_results.pose_landmarks.landmark
                    
                    try:
                        # Calculate torso angle with cropped coordinates
                        nose = [landmarks[self.mp_pose.PoseLandmark.NOSE.value].x,
                               landmarks[self.mp_pose.PoseLandmark.NOSE.value].y]
                        left_shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                       landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                        right_shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                        landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                        left_hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                   landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
                        right_hip = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                                    landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                        
                        # Calculate center points
                        shoulder_center = [(left_shoulder[0] + right_shoulder[0])/2, 
                                         (left_shoulder[1] + right_shoulder[1])/2]
                        hip_center = [(left_hip[0] + right_hip[0])/2, 
                                     (left_hip[1] + right_hip[1])/2]
                        
                        # Calculate torso angle
                        torso_angle = abs(math.degrees(math.atan2(
                            shoulder_center[1] - hip_center[1],
                            shoulder_center[0] - hip_center[0]
                        )))
                        
                        if torso_angle > 90:
                            torso_angle = 180 - torso_angle
                        
                        frame_angles.append(torso_angle)
                        
                        # Sit-up detection
                        if torso_angle < 30 and stage == "up":
                            stage = "down"
                        
                        if torso_angle > 60 and stage == "down":
                            stage = "up"
                            situp_count += 1
                            print(f"Sit-up #{situp_count} detected! (Torso angle: {torso_angle:.1f}°)")
                        
                        # Add pose overlay if requested
                        if show_overlay:
                            overlay_frame = self.draw_pose_overlay(cropped_frame.copy(), cropped_results.pose_landmarks)
                            # Add angle text
                            cv2.putText(overlay_frame, f'Angle: {torso_angle:.1f}°', 
                                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            cv2.putText(overlay_frame, f'Count: {situp_count}', 
                                      (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            processed_frames.append(overlay_frame)
                        
                    except (IndexError, AttributeError):
                        continue
        
        cap.release()
        
        # Calculate performance score
        score = self.calculate_percentile_score(situp_count, 'situps', age, gender)
        
        # Get feedback
        feedback = self.get_performance_feedback(score, 'situps', situp_count, age, gender)
        
        # Smooth angles for analysis
        smoothed_angles = self.smooth_signal(frame_angles) if frame_angles else []
        
        result = {
            'test_type': 'situps',
            'raw_count': situp_count,
            'score': score,
            'feedback': feedback,
            'total_frames': frame_count,
            'average_angle': np.mean(smoothed_angles) if smoothed_angles else 0,
            'angle_range': [np.min(smoothed_angles), np.max(smoothed_angles)] if smoothed_angles else [0, 0],
            'fps': fps,
            'processed_frames': len(processed_frames) if show_overlay else 0
        }
        
        print(f"\nSit-ups Analysis Complete:")
        print(f"Raw Count: {situp_count}")
        print(f"Performance Score: {score}/100")
        print(f"Rating: {feedback['overall_rating']}")
        print(f"Technique Tips: {feedback['technique_tips']}")
        print(f"Improvement Target: {feedback['improvement_targets']}")
        
        return result
    
    def analyze_vertical_jump(self, video_path: str, age: str, gender: str, show_overlay: bool = True) -> Dict:
        """Enhanced vertical jump analysis"""
        self._ensure_pose_available()
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        print("Analyzing vertical jump with enhanced features...")
        
        hip_heights = []
        processed_frames = []
        frame_count = 0
        
        # Reset bbox history
        self.bbox_history = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                # Auto-crop and resize
                cropped_frame = self.crop_and_resize_frame(frame, results.pose_landmarks)
                cropped_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
                cropped_results = self.pose.process(cropped_rgb)
                
                if cropped_results.pose_landmarks:
                    landmarks = cropped_results.pose_landmarks.landmark
                    
                    try:
                        left_hip_y = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y
                        right_hip_y = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y
                        avg_hip_y = (left_hip_y + right_hip_y) / 2
                        hip_heights.append(avg_hip_y)
                        
                        if show_overlay:
                            overlay_frame = self.draw_pose_overlay(cropped_frame.copy(), cropped_results.pose_landmarks)
                            cv2.putText(overlay_frame, f'Hip Height: {avg_hip_y:.3f}', 
                                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            processed_frames.append(overlay_frame)
                        
                    except (IndexError, AttributeError):
                        continue
        
        cap.release()
        
        if not hip_heights:
            return {'test_type': 'vertical_jump', 'score': 0, 'error': 'No valid poses detected'}
        
        # Calculate jump height
        smoothed_heights = self.smooth_signal(hip_heights, window_size=3)
        min_height = max(smoothed_heights)  
        max_height = min(smoothed_heights) 
        relative_jump = min_height - max_height
        estimated_jump_cm = relative_jump * 200  # Improved conversion factor
        
        # Calculate score
        score = self.calculate_percentile_score(estimated_jump_cm, 'vertical_jump', age, gender)
        
        # Get feedback
        feedback = self.get_performance_feedback(score, 'vertical_jump', estimated_jump_cm, age, gender)
        
        result = {
            'test_type': 'vertical_jump',
            'raw_height_cm': round(estimated_jump_cm, 1),
            'score': score,
            'feedback': feedback,
            'relative_jump': round(relative_jump, 4),
            'total_frames': frame_count,
            'processed_frames': len(processed_frames) if show_overlay else 0
        }
        
        print(f"\nVertical Jump Analysis Complete:")
        print(f"Jump Height: {estimated_jump_cm:.1f} cm")
        print(f"Performance Score: {score}/100")
        print(f"Rating: {feedback['overall_rating']}")
        
        return result
    
    def analyze_broad_jump(self, video_path: str, age: str, gender: str, show_overlay: bool = True) -> Dict:
        """Enhanced broad jump analysis"""
        self._ensure_pose_available()
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        print("Analyzing broad jump with enhanced features...")
        
        foot_positions = []
        processed_frames = []
        frame_count = 0
        
        self.bbox_history = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                cropped_frame = self.crop_and_resize_frame(frame, results.pose_landmarks)
                cropped_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
                cropped_results = self.pose.process(cropped_rgb)
                
                if cropped_results.pose_landmarks:
                    landmarks = cropped_results.pose_landmarks.landmark
                    
                    try:
                        left_foot = landmarks[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x
                        right_foot = landmarks[self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x
                        avg_foot_x = (left_foot + right_foot) / 2
                        foot_positions.append(avg_foot_x)
                        
                        if show_overlay:
                            overlay_frame = self.draw_pose_overlay(cropped_frame.copy(), cropped_results.pose_landmarks)
                            cv2.putText(overlay_frame, f'Foot Pos: {avg_foot_x:.3f}', 
                                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            processed_frames.append(overlay_frame)
                        
                    except (IndexError, AttributeError):
                        continue
        
        cap.release()
        
        if len(foot_positions) < 10:
            return {'test_type': 'broad_jump', 'score': 0, 'error': 'Insufficient data'}
        
        # Calculate jump distance
        start_pos = np.mean(foot_positions[:10])
        end_pos = np.mean(foot_positions[-10:])
        relative_distance = abs(end_pos - start_pos)
        estimated_distance_cm = relative_distance * 300  # Improved conversion
        
        # Calculate score
        score = self.calculate_percentile_score(estimated_distance_cm, 'broad_jump', age, gender)
        
        # Get feedback
        feedback = self.get_performance_feedback(score, 'broad_jump', estimated_distance_cm, age, gender)
        
        result = {
            'test_type': 'broad_jump',
            'raw_distance_cm': round(estimated_distance_cm, 1),
            'score': score,
            'feedback': feedback,
            'relative_distance': round(relative_distance, 4),
            'total_frames': frame_count,
            'processed_frames': len(processed_frames) if show_overlay else 0
        }
        
        print(f"\nBroad Jump Analysis Complete:")
        print(f"Jump Distance: {estimated_distance_cm:.1f} cm")
        print(f"Performance Score: {score}/100")
        print(f"Rating: {feedback['overall_rating']}")
        
        return result
    
    def analyze_flexibility(self, video_path: str, age: str, gender: str, show_overlay: bool = True) -> Dict:
        """Enhanced flexibility analysis"""
        self._ensure_pose_available()
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        print("Analyzing flexibility with enhanced features...")
        
        reach_distances = []
        processed_frames = []
        frame_count = 0
        
        self.bbox_history = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                cropped_frame = self.crop_and_resize_frame(frame, results.pose_landmarks)
                cropped_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
                cropped_results = self.pose.process(cropped_rgb)
                
                if cropped_results.pose_landmarks:
                    landmarks = cropped_results.pose_landmarks.landmark
                    
                    try:
                        left_hand = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                    landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                        right_hand = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                     landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                        left_hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                   landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
                        right_hip = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                                    landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                        
                        avg_hand = [(left_hand[0] + right_hand[0])/2, (left_hand[1] + right_hand[1])/2]
                        avg_hip = [(left_hip[0] + right_hip[0])/2, (left_hip[1] + right_hip[1])/2]
                        
                        reach_distance = abs(avg_hand[0] - avg_hip[0])
                        reach_distances.append(reach_distance)
                        
                        if show_overlay:
                            overlay_frame = self.draw_pose_overlay(cropped_frame.copy(), cropped_results.pose_landmarks)
                            cv2.putText(overlay_frame, f'Reach: {reach_distance:.3f}', 
                                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            processed_frames.append(overlay_frame)
                        
                    except (IndexError, AttributeError):
                        continue
        
        cap.release()
        
        if not reach_distances:
            return {'test_type': 'flexibility', 'score': 0, 'error': 'No valid poses detected'}
        
        # Calculate maximum reach
        max_reach = max(reach_distances)
        estimated_reach_cm = max_reach * 100  # Convert to cm
        
        # Calculate score
        score = self.calculate_percentile_score(estimated_reach_cm, 'flexibility', age, gender)
        
        # Get feedback
        feedback = self.get_performance_feedback(score, 'flexibility', estimated_reach_cm, age, gender)
        
        result = {
            'test_type': 'flexibility',
            'raw_reach_cm': round(estimated_reach_cm, 1),
            'score': score,
            'feedback': feedback,
            'relative_reach': round(max_reach, 4),
            'total_frames': frame_count,
            'processed_frames': len(processed_frames) if show_overlay else 0
        }
        
        print(f"\nFlexibility Analysis Complete:")
        print(f"Maximum Reach: {estimated_reach_cm:.1f} cm")
        print(f"Performance Score: {score}/100")
        print(f"Rating: {feedback['overall_rating']}")
        
        return result

def main():
    parser = argparse.ArgumentParser(description='Enhanced Fitness Assessment AI Analyzer')
    parser.add_argument('--video', type=str, help='Path to video file for analysis')
    parser.add_argument('--test', type=str, required=True, 
                       choices=['situps', 'vertical_jump', 'broad_jump', 'flexibility'], 
                       help='Type of test to perform')
    parser.add_argument('--age', type=str, required=True,
                       choices=['teenage', 'youth', 'adult'],
                       help='Age category')
    parser.add_argument('--gender', type=str, required=True,
                       choices=['male', 'female', 'other'],
                       help='Gender category')
    parser.add_argument('--show-overlay', action='store_true', default=True,
                       help='Show pose overlay during analysis')
    parser.add_argument('--output', type=str, help='Output file for results (JSON format)')
    
    args = parser.parse_args()
    
    analyzer = EnhancedFitnessAnalyzer()
    
    print("ENHANCED FITNESS ASSESSMENT AI - COMPLETE ANALYSIS")
    print(f"Test Type: {args.test.title()}")
    print(f"Profile: {args.age.title()} {args.gender.title()}")
    print("-" * 60)
    
    if not args.video:
        print("Error: Video file required for analysis")
        return
    
    try:
        if args.test == 'situps':
            result = analyzer.analyze_situps(args.video, args.age, args.gender, args.show_overlay)
            
        elif args.test == 'vertical_jump':
            result = analyzer.analyze_vertical_jump(args.video, args.age, args.gender, args.show_overlay)
            
        elif args.test == 'broad_jump':
            result = analyzer.analyze_broad_jump(args.video, args.age, args.gender, args.show_overlay)
            
        elif args.test == 'flexibility':
            result = analyzer.analyze_flexibility(args.video, args.age, args.gender, args.show_overlay)
        
        # Check for errors
        if 'error' in result:
            print(f"Error: {result['error']}")
            return
        
        # Display comprehensive results
        print(f"\n" + "="*60)
        print("COMPREHENSIVE RESULTS SUMMARY")
        print("="*60)
        print(f"Test Type: {result['test_type'].title()}")
        print(f"Raw Performance: {result.get('raw_count', result.get('raw_height_cm', result.get('raw_distance_cm', result.get('raw_reach_cm', 'N/A'))))}")
        print(f"Performance Score: {result['score']}/100")
        print(f"Overall Rating: {result['feedback']['overall_rating']}")
        print(f"\nTechnique Tips:")
        print(f"  {result['feedback']['technique_tips']}")
        print(f"\nImprovement Targets:")
        print(f"  {result['feedback']['improvement_targets']}")
        print(f"\nTechnical Details:")
        print(f"  Total Frames Processed: {result['total_frames']}")
        if result.get('processed_frames', 0) > 0:
            print(f"  Frames with Pose Overlay: {result['processed_frames']}")
        print("="*60)
        
        # Save results if output file specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nResults saved to: {args.output}")
            
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        print("Make sure your video file exists and is readable")
        return
    
    print(f"\nAnalysis Complete!")
    print(f"Note: Measurements are AI estimations based on pose detection")
    print(f"Score is calculated using percentile-based performance distribution")

if __name__ == "__main__":
    main()
