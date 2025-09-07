# utils/image_processor.py - Universal Image Processor
import io
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import logging
import cv2

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Universal image processor for ML models"""
    
    def __init__(self, target_size=(224, 224), normalize=True):
        self.target_size = target_size
        self.normalize = normalize
        
    def process_image(self, image_file, target_size=None, preprocessing_type='auto'):
        """
        Process image for ML model input
        
        Args:
            image_file: Uploaded file object
            target_size: Tuple of (width, height) for resizing
            preprocessing_type: 'auto', 'classification', 'segmentation', 'detection'
        
        Returns:
            Processed numpy array ready for model input
        """
        try:
            # Use provided target size or default
            size = target_size or self.target_size
            
            # Read image data
            image_data = image_file.read()
            
            # Open image with PIL
            image = Image.open(io.BytesIO(image_data))
            
            # Log original image info
            logger.info(f"Original image: {image.size}, mode: {image.mode}")
            
            # Process based on type
            if preprocessing_type == 'auto':
                processed = self._auto_process(image, size)
            elif preprocessing_type == 'classification':
                processed = self._classification_process(image, size)
            elif preprocessing_type == 'segmentation':
                processed = self._segmentation_process(image, size)
            elif preprocessing_type == 'detection':
                processed = self._detection_process(image, size)
            else:
                processed = self._default_process(image, size)
            
            logger.info(f"Processed image shape: {processed.shape}")
            return processed
            
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            raise Exception(f"Failed to process image: {str(e)}")
    
    def _auto_process(self, image, size):
        """Auto-detect best processing method"""
        # Start with classification processing as default
        return self._classification_process(image, size)
    
    def _classification_process(self, image, size):
        """Process image for classification models"""
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize image maintaining aspect ratio
            image = self._smart_resize(image, size)
            
            # Convert to numpy array
            image_array = np.array(image, dtype=np.float32)
            
            # Normalize to [0, 1] if requested
            if self.normalize:
                image_array = image_array / 255.0
            
            # Try different input formats for compatibility
            
            # Format 1: Flattened (for traditional ML models)
            flattened = image_array.flatten().reshape(1, -1)
            
            # Format 2: 4D tensor (batch, height, width, channels) - for CNN
            tensor_hwc = image_array.reshape(1, *image_array.shape)
            
            # Format 3: 4D tensor (batch, channels, height, width) - for PyTorch
            if len(image_array.shape) == 3:
                tensor_chw = np.transpose(image_array, (2, 0, 1))
                tensor_chw = tensor_chw.reshape(1, *tensor_chw.shape)
            else:
                tensor_chw = tensor_hwc
            
            # Return the most common format first (flattened for sklearn compatibility)
            return flattened
            
        except Exception as e:
            logger.error(f"Classification processing failed: {e}")
            raise
    
    def _segmentation_process(self, image, size):
        """Process image for segmentation models"""
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize without maintaining aspect ratio for segmentation
            image = image.resize(size, Image.LANCZOS)
            
            image_array = np.array(image, dtype=np.float32)
            
            if self.normalize:
                image_array = image_array / 255.0
            
            # Return as 4D tensor (batch, height, width, channels)
            return image_array.reshape(1, *image_array.shape)
            
        except Exception as e:
            logger.error(f"Segmentation processing failed: {e}")
            raise
    
    def _detection_process(self, image, size):
        """Process image for object detection models"""
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize maintaining aspect ratio and pad if needed
            image = self._letterbox_resize(image, size)
            
            image_array = np.array(image, dtype=np.float32)
            
            if self.normalize:
                image_array = image_array / 255.0
            
            return image_array.reshape(1, *image_array.shape)
            
        except Exception as e:
            logger.error(f"Detection processing failed: {e}")
            raise
    
    def _default_process(self, image, size):
        """Default processing method"""
        return self._classification_process(image, size)
    
    def _smart_resize(self, image, target_size):
        """Resize image while maintaining aspect ratio"""
        try:
            # Calculate aspect ratios
            original_width, original_height = image.size
            target_width, target_height = target_size
            
            original_ratio = original_width / original_height
            target_ratio = target_width / target_height
            
            if original_ratio > target_ratio:
                # Original is wider, fit by width
                new_width = target_width
                new_height = int(target_width / original_ratio)
            else:
                # Original is taller, fit by height
                new_height = target_height
                new_width = int(target_height * original_ratio)
            
            # Resize image
            image = image.resize((new_width, new_height), Image.LANCZOS)
            
            # Create new image with target size and paste resized image
            if new_width != target_width or new_height != target_height:
                new_image = Image.new('RGB', target_size, (128, 128, 128))  # Gray background
                paste_x = (target_width - new_width) // 2
                paste_y = (target_height - new_height) // 2
                new_image.paste(image, (paste_x, paste_y))
                image = new_image
            
            return image
            
        except Exception as e:
            logger.warning(f"Smart resize failed, using simple resize: {e}")
            return image.resize(target_size, Image.LANCZOS)
    
    def _letterbox_resize(self, image, target_size):
        """Letterbox resize for object detection"""
        try:
            original_width, original_height = image.size
            target_width, target_height = target_size
            
            # Calculate scale
            scale = min(target_width / original_width, target_height / original_height)
            
            # Calculate new size
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            
            # Resize image
            image = image.resize((new_width, new_height), Image.LANCZOS)
            
            # Create letterbox
            letterbox = Image.new('RGB', target_size, (114, 114, 114))  # Gray padding
            paste_x = (target_width - new_width) // 2
            paste_y = (target_height - new_height) // 2
            letterbox.paste(image, (paste_x, paste_y))
            
            return letterbox
            
        except Exception as e:
            logger.warning(f"Letterbox resize failed: {e}")
            return image.resize(target_size, Image.LANCZOS)
    
    def enhance_image(self, image, brightness=1.0, contrast=1.0, saturation=1.0, sharpness=1.0):
        """Apply image enhancements"""
        try:
            if brightness != 1.0:
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(brightness)
            
            if contrast != 1.0:
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(contrast)
            
            if saturation != 1.0:
                enhancer = ImageEnhance.Color(image)
                image = enhancer.enhance(saturation)
            
            if sharpness != 1.0:
                enhancer = ImageEnhance.Sharpness(image)
                image = enhancer.enhance(sharpness)
            
            return image
            
        except Exception as e:
            logger.warning(f"Image enhancement failed: {e}")
            return image
    
    def preprocess_for_model_type(self, image_file, model_type):
        """Preprocess based on detected model type"""
        try:
            if 'tensorflow' in model_type.lower() or 'keras' in model_type.lower():
                return self.process_image(image_file, preprocessing_type='classification')
            elif 'pytorch' in model_type.lower():
                # PyTorch often uses channels-first format
                processed = self.process_image(image_file, preprocessing_type='classification')
                if len(processed.shape) == 4 and processed.shape[-1] == 3:
                    # Convert from NHWC to NCHW
                    processed = np.transpose(processed, (0, 3, 1, 2))
                return processed
            elif 'sklearn' in model_type.lower() or 'pickle' in model_type.lower():
                # Scikit-learn usually expects flattened input
                return self.process_image(image_file, preprocessing_type='classification')
            else:
                # Default processing
                return self.process_image(image_file, preprocessing_type='auto')
                
        except Exception as e:
            logger.error(f"Model-specific preprocessing failed: {e}")
            return self.process_image(image_file, preprocessing_type='auto')
    
    def get_multiple_formats(self, image_file):
        """Get image in multiple formats for model compatibility"""
        try:
            # Read image
            image_data = image_file.read()
            image = Image.open(io.BytesIO(image_data))
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize
            image = self._smart_resize(image, self.target_size)
            image_array = np.array(image, dtype=np.float32)
            
            if self.normalize:
                image_array = image_array / 255.0
            
            formats = {
                'flattened': image_array.flatten().reshape(1, -1),
                'hwc_4d': image_array.reshape(1, *image_array.shape),
                'chw_4d': np.transpose(image_array, (2, 0, 1)).reshape(1, 3, *self.target_size),
                'hwc_3d': image_array,
                'chw_3d': np.transpose(image_array, (2, 0, 1))
            }
            
            return formats
            
        except Exception as e:
            logger.error(f"Multiple format generation failed: {e}")
            raise