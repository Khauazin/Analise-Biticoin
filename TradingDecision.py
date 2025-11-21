from datetime import datetime
from Logger import iniciar_logger

# Inicializar logger para backend
logger = iniciar_logger("backend")


class TradingDecision:
    """Modelo para decisões de negociação"""
    def __init__(self,
                 decision,
                 price,
                 confidence=None,
                 indicators=None,
                 entry_price=None,
                 take_profit=None,
                 stop_loss=None,
                 best_timeframe=None,
                 optimal_entry_hour=None,
                 position_size=None,
                 trailing_stop=None
                 ):
        self.decision = decision  # "BUY", "SELL", "HOLD"
        self.price = price
        self.confidence = confidence
        self.indicators = indicators or {}
        self.timestamp = datetime.now()
        self.id = None
        self.entry_price = entry_price
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.best_timeframe = best_timeframe
        # Melhor timeframe para trading
        self.optimal_entry_hour = optimal_entry_hour
        # Melhor horário de entrada (0-23)
        self.position_size = position_size
        # Tamanho da posição baseado em risco
        self.trailing_stop = trailing_stop
        # Stop móvel (trailing stop) em %
        logger.info(
            (
               f"Decisão de trading criada: {decision} no preço {price}, "
               f"timeframe: {best_timeframe}, "
               f"horário ótimo: {optimal_entry_hour}, "
               f"position_size: {position_size}, "
               f"trailing_stop: {trailing_stop}"
               )
            )
