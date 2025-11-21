import os
import signal
import time
from Market_Analyzer import MarketAnalyzer
from RiskManager import RiskManager
from Logger import iniciar_logger


def main() -> int:
    logger = iniciar_logger("worker")
    interval = int(os.getenv("BACKGROUND_INTERVAL", "30"))
    logger.info("Worker iniciado (intervalo=%ss)", interval)

    analyzer = MarketAnalyzer()
    risk = RiskManager()

    stop = False

    def _sig_handler(signum, frame):
        nonlocal stop
        logger.info("Sinal recebido (%s), encerrando worker...", signum)
        stop = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _sig_handler)
        except Exception:
            pass

    while not stop:
        try:
            try:
                risk.sync_spot_account(analyzer.binance)
            except Exception:
                pass
            res = analyzer.analyze_market()
            if isinstance(res, dict):
                # Resultados são persistidos
                # via DatabaseManager (decisions/logs)
                pass
        except Exception as e:
            logger.error("Erro no loop do worker: %s", e)
        # aguarda com interrupção possível por sinal
        for _ in range(interval):
            if stop:
                break
            time.sleep(1)

    logger.info("Worker finalizado.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
