from datetime import datetime
import threading
import time

import logging
import os


def iniciar_logger(nome_logger: str = "bitcoin_trader") -> logging.Logger:
    """Configura logger compartilhado (arquivo diário + console), UTF-8.

    - Cria pasta logs/ e arquivo diário.
    - Evita handlers duplicados por logger nomeado.
    - Nível padrão: INFO (inclui avisos e operações principais).
    """
    logs_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(logs_dir, exist_ok=True)

    log_filename = datetime.now().strftime("btc_log_%Y-%m-%d.txt")
    log_path = os.path.join(logs_dir, log_filename)

    logger = logging.getLogger(nome_logger)
    logger.setLevel(logging.INFO)
    fmt = "%(asctime)s - %(levelname)s - %(message)s"

    if not logger.handlers:
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_formatter = logging.Formatter(fmt)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(fmt)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    return logger


def registrar_log(
    logger: logging.Logger,
    mensagem: str,
    nivel: str = "info",
) -> None:
    """Registra mensagens no log compartilhado."""
    lvl = (nivel or "").lower()
    if lvl == "erro":
        logger.error(mensagem)
    elif lvl == "aviso":
        logger.warning(mensagem)
    else:
        logger.info(mensagem)


def salvar_analise_em_arquivo(
    resultados: dict,
    pasta_logs: str | None = None,
) -> None:
    """Salva resultados do MarketAnalyzer em logs/YYYY-MM/
    analise_mercado_YYYYMMDD.txt.
    """
    if pasta_logs is None:
        pasta_logs = os.path.join(
            os.path.dirname(__file__),
            "logs",
            f"{datetime.now():%Y-%m}",
        )
    os.makedirs(pasta_logs, exist_ok=True)

    arquivo = os.path.join(
        pasta_logs,
        f"analise_mercado_{datetime.now():%Y%m%d}.txt",
    )

    def fmt(v, casas=2):
        try:
            if isinstance(v, (int, float)):
                return f"{v:,.{casas}f}"
            return str(v)
        except Exception:
            return str(v)

    resultados = resultados or {}
    ml_predictions = resultados.get('ml_predictions', {})
    best_pred = ml_predictions.get('best_model_prediction', 'N/A')
    neural_pred = ml_predictions.get('neural_network_prediction', 'N/A')

    df_tail = resultados.get('df_tail', []) or []
    linhas_velas: list[str] = []
    if df_tail:
        linhas_velas.append("\n--- ULTIMAS 5 VELAS ---")
        for i, vela in enumerate(df_tail[-5:], 1):
            linhas_velas.append(
                f"Vela {i}: "
                f"O:{fmt(vela.get('open','N/A'))} "
                f"H:{fmt(vela.get('high','N/A'))} "
                f"L:{fmt(vela.get('low','N/A'))} "
                f"C:{fmt(vela.get('close','N/A'))}"
            )

    timestamp = resultados.get(
        'timestamp',
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    )

    linhas = [
        "=" * 80,
        f"ANALISE DE MERCADO BITCOIN - {datetime.now():%Y-%m-%d %H:%M:%S}",
        "=" * 80,
        f"Timestamp: {timestamp}",
        f"Preco Atual BTC: ${fmt(resultados.get('price', 'N/A'))}",
        "",
        "DECISAO DE TRADING:",
        f"  Decisao: {str(resultados.get('decision','HOLD')).upper()}",
        f"  Confianca: {fmt((resultados.get('confidence',0) or 0)*100)}%",
        f"  Preco de Entrada: ${fmt(resultados.get('entry_price','N/A'))}",
        f"  Take Profit: ${fmt(resultados.get('take_profit','N/A'))}",
        f"  Stop Loss: ${fmt(resultados.get('stop_loss','N/A'))}",
        "",
        "INDICADORES TECNICOS:",
        f"  Tendencia: {str(resultados.get('trend','N/A')).upper()}",
        f"  Suporte: ${fmt(resultados.get('support','N/A'))}",
        f"  Resistencia: ${fmt(resultados.get('resistance','N/A'))}",
        f"  Volatilidade: {fmt(resultados.get('volatility','N/A'))}%",
        "",
        "OSCILADORES:",
        f"  RSI: {fmt(resultados.get('rsi','N/A'))}",
        f"  MACD: {fmt(resultados.get('macd','N/A'))}",
        "",
        "MEDIAS MOVEIS:",
        f"  SMA 20: ${fmt(resultados.get('sma_20','N/A'))}",
        f"  SMA 50: ${fmt(resultados.get('sma_50','N/A'))}",
        f"  SMA 200: ${fmt(resultados.get('sma_200','N/A'))}",
        "",
        "BOLLINGER BANDS:",
        f"  Superior: ${fmt(resultados.get('bb_upper','N/A'))}",
        f"  Medio: ${fmt(resultados.get('bb_middle','N/A'))}",
        f"  Inferior: ${fmt(resultados.get('bb_lower','N/A'))}",
        "",
        "PREVISOES ML:",
        f"  Melhor Modelo: ${fmt(best_pred)}",
        f"  Rede Neural: ${fmt(neural_pred)}",
        "",
        f"TradingDecision: {resultados.get('trading_decision','N/A')}",
        *linhas_velas,
        "=" * 80,
        "",
    ]

    try:
        with open(arquivo, "a", encoding="utf-8") as f:
            f.write("\n".join(linhas) + "\n")
        print(f"OK. Log de analise salvo em: {arquivo}")
    except Exception as e:
        print(f"Erro ao salvar log de analise: {e}")


def iniciar_log_automatico(
    analyzer,
    intervalo: int = 60,
    pasta_logs: str | None = None,
) -> None:
    """Roda em background para salvar logs do MarketAnalyzer
    automaticamente.
    """
    def loop():
        while True:
            try:
                resultados = analyzer.analyze_market()
                salvar_analise_em_arquivo(resultados, pasta_logs=pasta_logs)
            except Exception as e:
                print(f"Erro no log automatico: {e}")
            time.sleep(intervalo)

    t = threading.Thread(target=loop, daemon=True)
    t.start()
