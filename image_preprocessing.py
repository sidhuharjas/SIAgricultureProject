import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

class ImagePreprocessor:
    """Image preprocessing for AI training - rotates, resizes, and normalizes images."""
    
    def __init__(self, target_size: Tuple[int, int] = (640, 640)):
        """
        Initialize the preprocessor.
        
        Args:
            target_size: Target image size as (width, height). Default (640, 640).
        """
        self.target_size = target_size
    
    def load_image(self, image_path: str) -> np.ndarray:
        """Load image from file path."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate image by specified angle (in degrees).
        
        Args:
            image: Input image as numpy array
            angle: Rotation angle in degrees (positive = counter-clockwise)
        
        Returns:
            Rotated image
        """
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Perform rotation
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
        return rotated
    
    def resize_image(self, image: np.ndarray, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Resize image to target size while maintaining aspect ratio.
        
        Args:
            image: Input image
            size: Target size (width, height). Uses self.target_size if None.
        
        Returns:
            Resized image
        """
        if size is None:
            size = self.target_size
        
        resized = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        return resized
    
    def normalize_image(self, image: np.ndarray, method: str = "minmax") -> np.ndarray:
        """
        Normalize image to 0-1 range or standardize.
        
        Args:
            image: Input image
            method: "minmax" for 0-1 range, "zscore" for standardization
        
        Returns:
            Normalized image as float32
        """
        image = image.astype(np.float32)
        
        if method == "minmax":
            # Normalize to 0-1 range
            normalized = image / 255.0
        elif method == "zscore":
            # Standardize: (x - mean) / std
            mean = np.mean(image)
            std = np.std(image)
            normalized = (image - mean) / (std + 1e-7)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return normalized
    
    def preprocess(
        self, 
        image_path: str, 
        rotation_angle: float = 0.0,
        normalize_method: str = "minmax"
    ) -> np.ndarray:
        """
        Complete preprocessing pipeline: load -> rotate -> resize -> normalize.
        
        Args:
            image_path: Path to input image
            rotation_angle: Rotation angle in degrees
            normalize_method: Normalization method ("minmax" or "zscore")
        
        Returns:
            Preprocessed image as numpy array
        """
        # Load image
        image = self.load_image(image_path)
        
        # Rotate if specified
        if rotation_angle != 0:
            image = self.rotate_image(image, rotation_angle)
        
        # Resize to standard frame
        image = self.resize_image(image)
        
        # Normalize
        image = self.normalize_image(image, method=normalize_method)
        
        return image
    
    def batch_preprocess(self, image_dir: str, rotation_angle: float = 0.0) -> dict:
        """
        Preprocess all images in a directory.
        
        Args:
            image_dir: Directory containing images
            rotation_angle: Rotation angle in degrees
        
        Returns:
            Dictionary with image paths as keys and preprocessed images as values
        """
        image_dir = Path(image_dir)
        preprocessed = {}
        
        for image_path in image_dir.glob("*.jpg"):
            try:
                preprocessed[str(image_path)] = self.preprocess(str(image_path), rotation_angle)
                print(f"✓ Preprocessed: {image_path.name}")
            except Exception as e:
                print(f"✗ Error processing {image_path.name}: {e}")
        
        return preprocessed


# Example usage
if __name__ == "__main__":
    # Initialize preprocessor with 640x640 standard size
    preprocessor = ImagePreprocessor(target_size=(640, 640))
    
    # Example: Preprocess a single image
    # image = preprocessor.preprocess(
    #     "path/to/image.jpg",
    #     rotation_angle=45.0,
    #     normalize_method="minmax"
    # )
    # print(f"Preprocessed image shape: {image.shape}")
    
    print("Image preprocessor ready for use!")
