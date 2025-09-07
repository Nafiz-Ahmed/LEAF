# model/model_loader.py - Universal Model Loader
import os
import pickle
import joblib
import logging
import numpy as np
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class ModelLoader:
    """Universal model loader supporting various ML frameworks"""
    
    def __init__(self):
        self.model = None
        self.model_type = None
        self.model_info = {}
        self.class_names = []
        self.input_shape = None
        self.load_time = None
        
    def load_model(self):
        """Auto-detect and load model from various formats"""
        try:
            model_dir = 'model'
            
            # Try to load model config if exists
            config_path = os.path.join(model_dir, 'model_config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    self.class_names = config.get('class_names', [])
                    self.input_shape = config.get('input_shape', None)
                    self.model_info.update(config)
            
            # Try different model formats
            model_loaded = False
            
            # 1. Try Pickle (.pkl, .pickle)
            if not model_loaded:
                model_loaded = self._try_pickle_models(model_dir)
            
            # 2. Try Joblib (.joblib)
            if not model_loaded:
                model_loaded = self._try_joblib_models(model_dir)
            
            # 3. Try TensorFlow/Keras
            if not model_loaded:
                model_loaded = self._try_tensorflow_models(model_dir)
            
            # 4. Try PyTorch
            if not model_loaded:
                model_loaded = self._try_pytorch_models(model_dir)
            
            # 5. Try Scikit-learn specific
            if not model_loaded:
                model_loaded = self._try_sklearn_models(model_dir)
            
            # 6. Try ONNX
            if not model_loaded:
                model_loaded = self._try_onnx_models(model_dir)
            
            if not model_loaded:
                raise Exception("No compatible model found in model directory")
            
            self.load_time = datetime.utcnow()
            logger.info(f"Model loaded successfully: {self.model_type}")
            
            # Set default class names if not provided
            if not self.class_names:
                self._generate_default_class_names()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _try_pickle_models(self, model_dir):
        """Try to load pickle models"""
        pickle_files = [f for f in os.listdir(model_dir) if f.endswith(('.pkl', '.pickle'))]
        
        for file in pickle_files:
            try:
                path = os.path.join(model_dir, file)
                with open(path, 'rb') as f:
                    self.model = pickle.load(f)
                self.model_type = f"Pickle ({file})"
                logger.info(f"Loaded pickle model: {file}")
                return True
            except Exception as e:
                logger.warning(f"Failed to load pickle model {file}: {e}")
                continue
        return False
    
    def _try_joblib_models(self, model_dir):
        """Try to load joblib models"""
        joblib_files = [f for f in os.listdir(model_dir) if f.endswith('.joblib')]
        
        for file in joblib_files:
            try:
                path = os.path.join(model_dir, file)
                self.model = joblib.load(path)
                self.model_type = f"Joblib ({file})"
                logger.info(f"Loaded joblib model: {file}")
                return True
            except Exception as e:
                logger.warning(f"Failed to load joblib model {file}: {e}")
                continue
        return False
    
    def _try_tensorflow_models(self, model_dir):
        """Try to load TensorFlow/Keras models"""
        try:
            import tensorflow as tf
            
            # Try SavedModel format
            if os.path.isdir(os.path.join(model_dir, 'saved_model')):
                self.model = tf.keras.models.load_model(os.path.join(model_dir, 'saved_model'))
                self.model_type = "TensorFlow SavedModel"
                return True
            
            # Try .h5 format
            h5_files = [f for f in os.listdir(model_dir) if f.endswith('.h5')]
            for file in h5_files:
                try:
                    path = os.path.join(model_dir, file)
                    self.model = tf.keras.models.load_model(path)
                    self.model_type = f"Keras H5 ({file})"
                    return True
                except:
                    continue
                    
        except ImportError:
            logger.info("TensorFlow not available, skipping TF model loading")
        except Exception as e:
            logger.warning(f"Failed to load TensorFlow model: {e}")
        return False
    
    def _try_pytorch_models(self, model_dir):
        """Try to load PyTorch models"""
        try:
            import torch
            
            pt_files = [f for f in os.listdir(model_dir) if f.endswith(('.pt', '.pth'))]
            for file in pt_files:
                try:
                    path = os.path.join(model_dir, file)
                    self.model = torch.load(path, map_location='cpu')
                    self.model_type = f"PyTorch ({file})"
                    return True
                except:
                    continue
                    
        except ImportError:
            logger.info("PyTorch not available, skipping PyTorch model loading")
        except Exception as e:
            logger.warning(f"Failed to load PyTorch model: {e}")
        return False
    
    def _try_sklearn_models(self, model_dir):
        """Try to load scikit-learn models"""
        try:
            from sklearn.externals import joblib as sklearn_joblib
            
            sklearn_files = [f for f in os.listdir(model_dir) if 'sklearn' in f.lower()]
            for file in sklearn_files:
                try:
                    path = os.path.join(model_dir, file)
                    self.model = sklearn_joblib.load(path)
                    self.model_type = f"Scikit-learn ({file})"
                    return True
                except:
                    continue
                    
        except Exception as e:
            logger.warning(f"Failed to load scikit-learn model: {e}")
        return False
    
    def _try_onnx_models(self, model_dir):
        """Try to load ONNX models"""
        try:
            import onnxruntime as ort
            
            onnx_files = [f for f in os.listdir(model_dir) if f.endswith('.onnx')]
            for file in onnx_files:
                try:
                    path = os.path.join(model_dir, file)
                    self.model = ort.InferenceSession(path)
                    self.model_type = f"ONNX ({file})"
                    return True
                except:
                    continue
                    
        except ImportError:
            logger.info("ONNX Runtime not available, skipping ONNX model loading")
        except Exception as e:
            logger.warning(f"Failed to load ONNX model: {e}")
        return False
    
    def _generate_default_class_names(self):
        """Generate default class names based on model type"""
        try:
            if hasattr(self.model, 'classes_'):
                # Scikit-learn models
                self.class_names = [str(cls) for cls in self.model.classes_]
            elif hasattr(self.model, 'predict_proba'):
                # Try to infer from a dummy prediction
                dummy_input = np.random.random((1, 10))  # Adjust as needed
                try:
                    proba = self.model.predict_proba(dummy_input)
                    n_classes = proba.shape[1]
                    self.class_names = [f'Class_{i}' for i in range(n_classes)]
                except:
                    self.class_names = ['Class_0', 'Class_1']  # Default binary
            else:
                # Default fallback
                self.class_names = ['Positive', 'Negative']
                
        except Exception as e:
            logger.warning(f"Could not generate class names: {e}")
            self.class_names = ['Class_A', 'Class_B', 'Class_C']
    
    def predict(self, processed_image):
        """Make prediction using the loaded model"""
        try:
            if self.model is None:
                raise Exception("Model not loaded")
            
            # Handle different model types
            if 'tensorflow' in self.model_type.lower() or 'keras' in self.model_type.lower():
                return self._predict_tensorflow(processed_image)
            elif 'pytorch' in self.model_type.lower():
                return self._predict_pytorch(processed_image)
            elif 'onnx' in self.model_type.lower():
                return self._predict_onnx(processed_image)
            else:
                # Default sklearn-like interface
                return self._predict_sklearn(processed_image)
                
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def _predict_sklearn(self, processed_image):
        """Predict using scikit-learn like interface"""
        try:
            # Get prediction
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(processed_image)[0]
                predictions = [
                    {
                        'label': self.class_names[i] if i < len(self.class_names) else f'Class_{i}',
                        'confidence': float(prob)
                    }
                    for i, prob in enumerate(probabilities)
                ]
            else:
                prediction = self.model.predict(processed_image)[0]
                predictions = [
                    {
                        'label': str(prediction),
                        'confidence': 1.0
                    }
                ]
            
            # Sort by confidence
            predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
            return predictions
            
        except Exception as e:
            logger.error(f"Sklearn prediction failed: {e}")
            raise
    
    def _predict_tensorflow(self, processed_image):
        """Predict using TensorFlow/Keras model"""
        try:
            import tensorflow as tf
            
            # Ensure proper input shape
            if len(processed_image.shape) == 3:
                processed_image = np.expand_dims(processed_image, axis=0)
            
            # Make prediction
            prediction = self.model.predict(processed_image)
            
            # Handle different output formats
            if len(prediction.shape) == 2:  # Probabilities
                probabilities = prediction[0]
                predictions = [
                    {
                        'label': self.class_names[i] if i < len(self.class_names) else f'Class_{i}',
                        'confidence': float(prob)
                    }
                    for i, prob in enumerate(probabilities)
                ]
            else:  # Single prediction
                predictions = [
                    {
                        'label': f'Prediction: {prediction[0]}',
                        'confidence': 1.0
                    }
                ]
            
            predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
            return predictions
            
        except Exception as e:
            logger.error(f"TensorFlow prediction failed: {e}")
            raise
    
    def _predict_pytorch(self, processed_image):
        """Predict using PyTorch model"""
        try:
            import torch
            
            # Convert to tensor
            if isinstance(processed_image, np.ndarray):
                tensor = torch.from_numpy(processed_image).float()
            else:
                tensor = processed_image
            
            # Ensure proper dimensions
            if len(tensor.shape) == 3:
                tensor = tensor.unsqueeze(0)
            
            # Make prediction
            with torch.no_grad():
                self.model.eval()
                output = self.model(tensor)
                
                if hasattr(torch, 'softmax'):
                    probabilities = torch.softmax(output, dim=1)[0].numpy()
                else:
                    probabilities = output[0].numpy()
            
            predictions = [
                {
                    'label': self.class_names[i] if i < len(self.class_names) else f'Class_{i}',
                    'confidence': float(prob)
                }
                for i, prob in enumerate(probabilities)
            ]
            
            predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
            return predictions
            
        except Exception as e:
            logger.error(f"PyTorch prediction failed: {e}")
            raise
    
    def _predict_onnx(self, processed_image):
        """Predict using ONNX model"""
        try:
            # Get input name
            input_name = self.model.get_inputs()[0].name
            
            # Make prediction
            result = self.model.run(None, {input_name: processed_image})
            probabilities = result[0][0]  # Assuming single output
            
            predictions = [
                {
                    'label': self.class_names[i] if i < len(self.class_names) else f'Class_{i}',
                    'confidence': float(prob)
                }
                for i, prob in enumerate(probabilities)
            ]
            
            predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
            return predictions
            
        except Exception as e:
            logger.error(f"ONNX prediction failed: {e}")
            raise
    
    def warm_up(self):
        """Warm up the model with a dummy prediction"""
        try:
            # Create dummy data based on expected input shape
            if self.input_shape:
                dummy_input = np.random.random(self.input_shape).astype(np.float32)
            else:
                # Default shape for image models
                dummy_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
                # Try flattened version if 4D fails
                try:
                    self.predict(dummy_input)
                    return "Model warmed up successfully"
                except:
                    dummy_input = np.random.random((1, 224*224*3)).astype(np.float32)
            
            result = self.predict(dummy_input)
            return f"Model warmed up successfully with {len(result)} classes"
            
        except Exception as e:
            logger.warning(f"Model warm-up failed: {e}")
            return "Warm-up skipped"
    
    def get_model_info(self):
        """Get comprehensive model information"""
        info = {
            'model_type': self.model_type,
            'model_loaded': self.model is not None,
            'load_time': self.load_time.isoformat() if self.load_time else None,
            'class_names': self.class_names,
            'number_of_classes': len(self.class_names),
            'input_shape': self.input_shape,
        }
        
        # Add model-specific info
        try:
            if hasattr(self.model, 'get_params'):
                # Scikit-learn model
                info['parameters'] = str(self.model.get_params())
            elif hasattr(self.model, 'summary'):
                # Keras model
                info['model_summary'] = "Available via model.summary()"
            elif hasattr(self.model, '__class__'):
                info['model_class'] = str(self.model.__class__)
        except:
            pass
        
        info.update(self.model_info)
        return info