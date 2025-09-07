# utils/health_checker.py - System Health Monitor
import psutil
import time
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class HealthChecker:
    """System health monitoring"""
    
    def __init__(self):
        self.start_time = time.time()
        self.prediction_count = 0
        self.error_count = 0
        self.last_prediction_time = None
        self.last_error_time = None
        
    def get_health_status(self, model=None):
        """Get comprehensive health status"""
        try:
            current_time = datetime.utcnow()
            uptime = self.get_uptime()
            
            # Memory usage
            memory = psutil.virtual_memory()
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            
            # Model status
            model_status = {
                'loaded': model is not None,
                'type': str(type(model).__name__) if model else None,
                'memory_footprint': self._estimate_model_size(model) if model else None
            }
            
            # Performance metrics
            performance = {
                'total_predictions': self.prediction_count,
                'total_errors': self.error_count,
                'error_rate': (self.error_count / max(self.prediction_count, 1)) * 100,
                'last_prediction': self.last_prediction_time.isoformat() if self.last_prediction_time else None,
                'last_error': self.last_error_time.isoformat() if self.last_error_time else None
            }
            
            # Determine overall health
            health_score = self._calculate_health_score(memory, cpu_percent, performance)
            
            status = {
                'status': 'healthy' if health_score > 70 else 'degraded' if health_score > 40 else 'unhealthy',
                'health_score': health_score,
                'timestamp': current_time.isoformat(),
                'uptime': uptime,
                'system': {
                    'memory': {
                        'total': memory.total,
                        'available': memory.available,
                        'percent': memory.percent,
                        'used': memory.used
                    },
                    'cpu': {
                        'percent': cpu_percent,
                        'count': psutil.cpu_count()
                    },
                    'disk': {
                        'total': disk.total,
                        'used': disk.used,
                        'free': disk.free,
                        'percent': (disk.used / disk.total) * 100
                    }
                },
                'model': model_status,
                'performance': performance
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def _calculate_health_score(self, memory, cpu_percent, performance):
        """Calculate overall health score (0-100)"""
        try:
            score = 100
            
            # Memory penalty
            if memory.percent > 90:
                score -= 30
            elif memory.percent > 80:
                score -= 20
            elif memory.percent > 70:
                score -= 10
            
            # CPU penalty
            if cpu_percent > 90:
                score -= 25
            elif cpu_percent > 80:
                score -= 15
            elif cpu_percent > 70:
                score -= 10
            
            # Error rate penalty
            error_rate = performance['error_rate']
            if error_rate > 50:
                score -= 40
            elif error_rate > 20:
                score -= 25
            elif error_rate > 10:
                score -= 15
            elif error_rate > 5:
                score -= 10
            
            return max(0, score)
            
        except Exception as e:
            logger.error(f"Health score calculation failed: {e}")
            return 50  # Default neutral score
    
    def _estimate_model_size(self, model):
        """Estimate model memory footprint"""
        try:
            import sys
            
            # Try different methods based on model type
            if hasattr(model, '__sizeof__'):
                return sys.getsizeof(model)
            elif hasattr(model, 'get_weights'):
                # Keras/TensorFlow model
                weights = model.get_weights()
                size = sum(w.nbytes for w in weights)
                return size
            elif hasattr(model, 'state_dict'):
                # PyTorch model
                state_dict = model.state_dict()
                size = sum(param.numel() * param.element_size() for param in state_dict.values())
                return size
            else:
                # Fallback
                return sys.getsizeof(model)
                
        except Exception as e:
            logger.warning(f"Could not estimate model size: {e}")
            return None
    
    def record_prediction(self):
        """Record a successful prediction"""
        self.prediction_count += 1
        self.last_prediction_time = datetime.utcnow()
        logger.debug(f"Recorded prediction #{self.prediction_count}")
    
    def record_error(self):
        """Record an error"""
        self.error_count += 1
        self.last_error_time = datetime.utcnow()
        logger.debug(f"Recorded error #{self.error_count}")
    
    def get_uptime(self):
        """Get system uptime"""
        try:
            uptime_seconds = time.time() - self.start_time
            uptime_str = str(timedelta(seconds=int(uptime_seconds)))
            return {
                'seconds': uptime_seconds,
                'formatted': uptime_str
            }
        except Exception as e:
            logger.error(f"Failed to get uptime: {e}")
            return {'seconds': 0, 'formatted': '0:00:00'}
    
    def get_performance_summary(self):
        """Get performance summary"""
        try:
            total_requests = self.prediction_count + self.error_count
            success_rate = (self.prediction_count / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'total_requests': total_requests,
                'successful_predictions': self.prediction_count,
                'errors': self.error_count,
                'success_rate': round(success_rate, 2),
                'error_rate': round(100 - success_rate, 2),
                'uptime': self.get_uptime(),
                'average_requests_per_hour': self._calculate_requests_per_hour()
            }
            
        except Exception as e:
            logger.error(f"Failed to get performance summary: {e}")
            return {}
    
    def _calculate_requests_per_hour(self):
        """Calculate average requests per hour"""
        try:
            uptime_hours = (time.time() - self.start_time) / 3600
            if uptime_hours == 0:
                return 0
            total_requests = self.prediction_count + self.error_count
            return round(total_requests / uptime_hours, 2)
        except:
            return 0
    
    def is_healthy(self, min_health_score=70):
        """Check if system is healthy"""
        try:
            status = self.get_health_status()
            return status.get('health_score', 0) >= min_health_score
        except:
            return False
    
    def reset_counters(self):
        """Reset performance counters"""
        try:
            old_predictions = self.prediction_count
            old_errors = self.error_count
            
            self.prediction_count = 0
            self.error_count = 0
            self.last_prediction_time = None
            self.last_error_time = None
            
            logger.info(f"Reset counters: {old_predictions} predictions, {old_errors} errors")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset counters: {e}")
            return False