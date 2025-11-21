from datetime import datetime


class BitcoinPrice:
    """Modelo para dados de preço do Bitcoin"""
    def __init__(self, price, timestamp=None, volume=None):
        self.price = price
        self.timestamp = timestamp or datetime.now()
        self.volume = volume
        self.id = None
