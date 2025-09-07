# app.py - Main Flask Application
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import io
import logging
import traceback
from datetime import datetime
import hashlib
import json
from functools import lru_cache
import time
import psutil
import gc

# Import your model utilities
from model.model_loader import ModelLoader
from utils.image_processor import ImageProcessor
from utils.cache_manager import CacheManager
from utils.health_checker import HealthChecker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins="*")  # Configure as needed for production

# Configuration
class Config:
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    UPLOAD_FOLDER = 'uploads'
    MODEL_PATH = 'model'
    CACHE_SIZE = 100
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
app.config.from_object(Config)

# Create upload directory
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)

# Global instances
model_loader = ModelLoader()
image_processor = ImageProcessor()
cache_manager = CacheManager(max_size=Config.CACHE_SIZE)
health_checker = HealthChecker()

# Application startup
@app.before_first_request
def initialize_app():
    """Initialize the application on first request"""
    try:
        logger.info("Initializing ML Backend...")
        
        # Load the model
        model_loader.load_model()
        logger.info("Model loaded successfully")
        
        # Warm up the model with a dummy prediction
        dummy_prediction = model_loader.warm_up()
        logger.info(f"Model warmed up: {dummy_prediction}")
        
        logger.info("Backend initialization complete")
        
    except Exception as e:
        logger.error(f"Failed to initialize backend: {e}")
        logger.error(traceback.format_exc())

# Health check endpoints
@app.route('/', methods=['GET'])
@app.route('/health', methods=['GET'])
def health_check():
    """Comprehensive health check endpoint"""
    try:
        health_status = health_checker.get_health_status(model_loader.model)
        return jsonify(health_status), 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@app.route('/status', methods=['GET'])
def detailed_status():
    """Detailed system status"""
    try:
        return jsonify({
            'status': 'healthy',
            'model_loaded': model_loader.model is not None,
            'model_type': model_loader.model_type,
            'memory_usage': {
                'total': psutil.virtual_memory().total,
                'available': psutil.virtual_memory().available,
                'percent': psutil.virtual_memory().percent,
                'used': psutil.virtual_memory().used
            },
            'cache_stats': cache_manager.get_stats(),
            'uptime': health_checker.get_uptime(),
            'predictions_made': health_checker.prediction_count,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return jsonify({'error': str(e)}), 500

# Main prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint with comprehensive error handling"""
    start_time = time.time()
    
    try:
        # Validate model is loaded
        if model_loader.model is None:
            logger.error("Model not loaded")
            return jsonify({
                'error': 'Model not available',
                'code': 'MODEL_NOT_LOADED'
            }), 503
        
        # Validate request
        validation_error = _validate_request(request)
        if validation_error:
            return validation_error
        
        # Process the uploaded image
        image_file = request.files['image']
        logger.info(f"Processing image: {image_file.filename}")
        
        # Generate cache key
        image_data = image_file.read()
        cache_key = hashlib.md5(image_data).hexdigest()
        
        # Check cache first
        cached_result = cache_manager.get(cache_key)
        if cached_result:
            logger.info(f"Returning cached result for {image_file.filename}")
            return jsonify({
                'success': True,
                'cached': True,
                'filename': image_file.filename,
                'predictions': cached_result['predictions'],
                'processing_time': time.time() - start_time,
                'timestamp': datetime.utcnow().isoformat()
            }), 200
        
        # Reset file pointer
        image_file.seek(0)
        
        # Process image
        processed_image = image_processor.process_image(image_file)
        logger.info("Image processed successfully")
        
        # Make prediction
        predictions = model_loader.predict(processed_image)
        logger.info(f"Prediction completed: {len(predictions)} results")
        
        # Cache the result
        result_data = {'predictions': predictions}
        cache_manager.put(cache_key, result_data)
        
        # Update health checker
        health_checker.record_prediction()
        
        # Clean up memory
        gc.collect()
        
        # Return response
        response = {
            'success': True,
            'cached': False,
            'filename': image_file.filename,
            'predictions': predictions,
            'processing_time': time.time() - start_time,
            'model_type': model_loader.model_type,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        logger.info(f"Prediction successful in {response['processing_time']:.2f}s")
        return jsonify(response), 200
        
    except Exception as e:
        error_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        logger.error(f"Prediction failed [ID: {error_id}]: {e}")
        logger.error(traceback.format_exc())
        
        health_checker.record_error()
        
        return jsonify({
            'success': False,
            'error': str(e),
            'error_id': error_id,
            'processing_time': time.time() - start_time,
            'timestamp': datetime.utcnow().isoformat()
        }), 500

# Batch prediction endpoint
@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Batch prediction endpoint for multiple images"""
    start_time = time.time()
    
    try:
        if model_loader.model is None:
            return jsonify({'error': 'Model not available'}), 503
        
        # Validate batch request
        if 'images' not in request.files:
            return jsonify({'error': 'No images provided'}), 400
        
        images = request.files.getlist('images')
        if not images or len(images) == 0:
            return jsonify({'error': 'No images selected'}), 400
        
        if len(images) > 10:  # Limit batch size
            return jsonify({'error': 'Maximum 10 images per batch'}), 400
        
        results = []
        
        for i, image_file in enumerate(images):
            try:
                # Process each image
                processed_image = image_processor.process_image(image_file)
                predictions = model_loader.predict(processed_image)
                
                results.append({
                    'index': i,
                    'filename': image_file.filename,
                    'success': True,
                    'predictions': predictions
                })
                
            except Exception as e:
                logger.error(f"Failed to process {image_file.filename}: {e}")
                results.append({
                    'index': i,
                    'filename': image_file.filename,
                    'success': False,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'total_images': len(images),
            'processed': len([r for r in results if r['success']]),
            'failed': len([r for r in results if not r['success']]),
            'results': results,
            'processing_time': time.time() - start_time,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        return jsonify({'error': str(e)}), 500

# Model information endpoint
@app.route('/model/info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    try:
        if model_loader.model is None:
            return jsonify({'error': 'Model not loaded'}), 503
        
        info = model_loader.get_model_info()
        return jsonify(info), 200
        
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        return jsonify({'error': str(e)}), 500

# Clear cache endpoint
@app.route('/cache/clear', methods=['POST'])
def clear_cache():
    """Clear the prediction cache"""
    try:
        cleared_count = cache_manager.clear()
        gc.collect()  # Force garbage collection
        
        return jsonify({
            'success': True,
            'cleared_items': cleared_count,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        return jsonify({'error': str(e)}), 500

# Utility functions
def _validate_request(request):
    """Validate the incoming request"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    if not _allowed_file(image_file.filename):
        return jsonify({
            'error': 'Invalid file type',
            'allowed_types': list(Config.ALLOWED_EXTENSIONS)
        }), 400
    
    return None

def _allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

# Error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({
        'error': 'File too large',
        'max_size': '16MB',
        'code': 'FILE_TOO_LARGE'
    }), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        'error': 'Endpoint not found',
        'available_endpoints': [
            'GET /',
            'GET /health',
            'GET /status',
            'POST /predict',
            'POST /predict/batch',
            'GET /model/info',
            'POST /cache/clear'
        ]
    }), 404

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {e}")
    return jsonify({
        'error': 'Internal server error',
        'timestamp': datetime.utcnow().isoformat()
    }), 500

# Cleanup on shutdown
import atexit

def cleanup():
    """Cleanup resources on shutdown"""
    try:
        logger.info("Cleaning up resources...")
        cache_manager.clear()
        if model_loader.model:
            del model_loader.model
        gc.collect()
        logger.info("Cleanup completed")
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")

atexit.register(cleanup)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting ML Backend on port {port}")
    logger.info(f"Debug mode: {debug}")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug,
        threaded=True
    )