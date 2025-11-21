import time
import threading
from typing import Any, Optional
from Logger import iniciar_logger


class CacheManager:
    """Gerenciador de cache centralizado com TTL para respostas de API."""

    def __init__(self):
        self.cache = {}
        self.lock = threading.Lock()
        self.logger = iniciar_logger("CacheManager")

    def set(self, key: str, value: Any, ttl_seconds: int = 300) -> None:
        """Armazena um valor no cache com TTL."""
        with self.lock:
            expiry = time.time() + ttl_seconds
            self.cache[key] = {
                'value': value,
                'expiry': expiry
            }
            self.logger.debug(f"Cached key '{key}' with TTL {ttl_seconds}s")

    def get(self, key: str) -> Optional[Any]:
        """Recupera um valor do cache se ainda válido."""
        with self.lock:
            if key in self.cache:
                item = self.cache[key]
                if time.time() < item['expiry']:
                    self.logger.debug(f"Cache hit for key '{key}'")
                    return item['value']
                else:
                    # Remove expired item
                    del self.cache[key]
                    self.logger.debug(f"Expired cache key '{key}' removed")
            return None

    def delete(self, key: str) -> None:
        """Remove um item do cache."""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                self.logger.debug(f"Cache key '{key}' deleted")

    def clear(self) -> None:
        """Limpa todo o cache."""
        with self.lock:
            self.cache.clear()
            self.logger.info("Cache cleared")

    def cleanup_expired(self) -> None:
        """Remove itens expirados do cache."""
        with self.lock:
            current_time = time.time()
            expired_keys = [
                k for k, v in self.cache.items() if current_time >= v['expiry']
            ]
            for key in expired_keys:
                del self.cache[key]
            if expired_keys:
                self.logger.debug(
                    f"Cleaned up {len(expired_keys)} expired cache items"
                )

    def get_stats(self) -> dict:
        """Retorna estatísticas do cache."""
        with self.lock:
            total_items = len(self.cache)
            current_time = time.time()
            valid_items = sum(
                1 for v in self.cache.values() if current_time < v['expiry']
            )
            return {
                'total_items': total_items,
                'valid_items': valid_items,
                'expired_items': total_items - valid_items
            }


# Instância global do cache manager
cache_manager = CacheManager()


# Função para limpeza periódica
def start_cache_cleanup():
    """Inicia limpeza periódica do cache em background."""
    def cleanup_worker():
        while True:
            time.sleep(300)  # Limpa a cada 5 minutos
            cache_manager.cleanup_expired()

    thread = threading.Thread(target=cleanup_worker, daemon=True)
    thread.start()
    cache_manager.logger.info("Cache cleanup worker started")
