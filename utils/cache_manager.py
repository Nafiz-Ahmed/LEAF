# utils/cache_manager.py - Prediction Cache Manager
import time
from collections import OrderedDict
import logging
import json
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class CacheManager:
    """LRU Cache for prediction results"""
    
    def __init__(self, max_size=100, ttl_seconds=3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds  # Time to live in seconds
        self.cache = OrderedDict()
        self.timestamps = {}
        self.hits = 0
        self.misses = 0
        
    def get(self, key):
        """Get item from cache"""
        try:
            current_time = time.time()
            
            if key in self.cache:
                # Check if item has expired
                if current_time - self.timestamps[key] > self.ttl_seconds:
                    self._remove_item(key)
                    self.misses += 1
                    return None
                
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                self.hits += 1
                logger.debug(f"Cache hit for key: {key[:8]}...")
                return self.cache[key]
            
            self.misses += 1
            logger.debug(f"Cache miss for key: {key[:8]}...")
            return None
            
        except Exception as e:
            logger.error(f"Cache get failed: {e}")
            return None
    
    def put(self, key, value):
        """Put item in cache"""
        try:
            current_time = time.time()
            
            if key in self.cache:
                # Update existing item
                self.cache[key] = value
                self.timestamps[key] = current_time
                self.cache.move_to_end(key)
            else:
                # Add new item
                self.cache[key] = value
                self.timestamps[key] = current_time
                
                # Remove oldest items if cache is full
                while len(self.cache) > self.max_size:
                    oldest_key = next(iter(self.cache))
                    self._remove_item(oldest_key)
            
            logger.debug(f"Cached result for key: {key[:8]}...")
            
        except Exception as e:
            logger.error(f"Cache put failed: {e}")
    
    def _remove_item(self, key):
        """Remove item from cache"""
        try:
            if key in self.cache:
                del self.cache[key]
            if key in self.timestamps:
                del self.timestamps[key]
        except Exception as e:
            logger.error(f"Failed to remove cache item: {e}")
    
    def clear(self):
        """Clear all cache"""
        try:
            count = len(self.cache)
            self.cache.clear()
            self.timestamps.clear()
            logger.info(f"Cache cleared: {count} items removed")
            return count
        except Exception as e:
            logger.error(f"Cache clear failed: {e}")
            return 0
    
    def cleanup_expired(self):
        """Remove expired items"""
        try:
            current_time = time.time()
            expired_keys = []
            
            for key, timestamp in self.timestamps.items():
                if current_time - timestamp > self.ttl_seconds:
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._remove_item(key)
            
            if expired_keys:
                logger.info(f"Cleaned up {len(expired_keys)} expired cache items")
            
            return len(expired_keys)
            
        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")
            return 0
    
    def get_stats(self):
        """Get cache statistics"""
        try:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
            
            current_time = time.time()
            expired_count = sum(1 for ts in self.timestamps.values() 
                              if current_time - ts > self.ttl_seconds)
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': round(hit_rate, 2),
                'expired_items': expired_count,
                'ttl_seconds': self.ttl_seconds
            }
            
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {}
    
    def get_cache_info(self):
        """Get detailed cache information"""
        try:
            current_time = time.time()
            items_info = []
            
            for key, value in list(self.cache.items())[:10]:  # Show only first 10
                age = current_time - self.timestamps.get(key, current_time)
                items_info.append({
                    'key': key[:16] + '...' if len(key) > 16 else key,
                    'age_seconds': round(age, 2),
                    'expired': age > self.ttl_seconds,
                    'size_estimate': len(str(value))
                })
            
            return {
                'stats': self.get_stats(),
                'recent_items': items_info,
                'total_items': len(self.cache)
            }
            
        except Exception as e:
            logger.error(f"Failed to get cache info: {e}")
            return {'error': str(e)}
    
    def resize(self, new_max_size):
        """Resize cache capacity"""
        try:
            old_size = self.max_size
            self.max_size = new_max_size
            
            # Remove items if new size is smaller
            while len(self.cache) > self.max_size:
                oldest_key = next(iter(self.cache))
                self._remove_item(oldest_key)
            
            logger.info(f"Cache resized from {old_size} to {new_max_size}")
            return True
            
        except Exception as e:
            logger.error(f"Cache resize failed: {e}")
            return False
    
    def set_ttl(self, new_ttl_seconds):
        """Update TTL for cache items"""
        try:
            old_ttl = self.ttl_seconds
            self.ttl_seconds = new_ttl_seconds
            
            # Clean up items that are now expired with new TTL
            self.cleanup_expired()
            
            logger.info(f"Cache TTL updated from {old_ttl}s to {new_ttl_seconds}s")
            return True
            
        except Exception as e:
            logger.error(f"TTL update failed: {e}")
            return False