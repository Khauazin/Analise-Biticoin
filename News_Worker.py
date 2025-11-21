import time
import requests
import traceback
import logging
import os
try:
    from PyQt5.QtCore import QThread, pyqtSignal
except Exception:
    class QThread:
        def __init__(self, *_, **__):
            pass

        def start(self):
            pass

        def wait(self):
            pass

    def pyqtSignal(*_args, **_kwargs):
        class _S:
            def emit(self, *_a, **_k):
                pass
        return _S()
from textblob import TextBlob
from dateutil import parser
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from cache_manager import cache_manager
from alert_system import alert_system
nltk.download('vader_lexicon', quiet=True)


# Set up logger for NewsWorker
news_logger = logging.getLogger("NewsWorker")
news_logger.setLevel(logging.INFO)
if not news_logger.handlers:
    handler = logging.StreamHandler()
    fmt = '%(asctime)s - %(levelname)s - %(message)s'
    handler.setFormatter(logging.Formatter(fmt))
    news_logger.addHandler(handler)


class NewsWorker(QThread):
    """Thread worker para buscar notícias de criptomoedas"""
    news_ready = pyqtSignal(list)
    error_occurred = pyqtSignal(str)

    # Class-level cache for news
    _news_cache = {}
    _cache_timestamp = 0
    _cache_duration = 600  # 10 minutes cache

    def __init__(self):
        super().__init__()
        self.running = True

    def run(self):
        while self.running:
            try:
                news = self.fetch_crypto_news()
                if news:
                    news_with_sentiment = self.analyze_sentiment(news)
                    self.news_ready.emit(news_with_sentiment)
                else:
                    self.error_occurred.emit("Nenhuma notícia encontrada")
            except Exception as e:
                error_msg = f"Error fetching news: {str(e)}"
                print(f"{error_msg}\n{traceback.format_exc()}")
                self.error_occurred.emit(error_msg)

            time.sleep(900)  # 15 minutos

    def fetch_crypto_news(self):
        """Busca notícias de criptomoedas usando múltiplas
        APIs com fallback e cache aprimorado"""
        # Check centralized cache first
        cache_key = "crypto_news"
        cached_news = cache_manager.get(cache_key)
        if cached_news is not None:
            news_logger.info("Using centralized cached news data")
            return cached_news

        # Try primary API (CryptoPanic)
        news_items = self._fetch_from_cryptopanic()
        if news_items:
            cache_manager.set(cache_key, news_items, 600)  # 10 minutes TTL
            return news_items

        # Fallback to NewsAPI if available
        news_items = self._fetch_from_newsapi()
        if news_items:
            cache_manager.set(cache_key, news_items, 600)
            return news_items

        # All APIs failed
        news_logger.error("All news API attempts failed")
        alert_system.alert_api_failure("News APIs", "All news sources failed")
        self.error_occurred.emit("Falha ao buscar notícias de todas as fontes")
        return []

    def _fetch_from_cryptopanic(self):
        """Busca notícias do CryptoPanic com circuit breaker."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                api_key = os.getenv("CRYPTOPANIC_API_KEY", "")
                if not api_key:
                    news_logger.warning(
                        "API_KEY não configurada; pulando CryptoPanic")
                    return None

                url = "https://cryptopanic.com/api/v1/posts/"
                params = {
                    'auth_token': api_key,
                    'currencies': 'BTC,ETH,LTC',
                    'public': 'true'
                }

                news_logger.info(
                    "Fetching news from CryptoPanic API "
                    f"(attempt {attempt+1}/{max_retries})"
                )
                response = requests.get(url, params=params, timeout=10)

                if response.status_code == 429:
                    wait_time = 60 * (attempt + 1)
                    news_logger.warning(
                        "Rate limit exceeded, retrying in "
                        f"{wait_time} seconds..."
                    )
                    alert_system.alert_rate_limit("CryptoPanic", wait_time)
                    time.sleep(wait_time)
                    continue
                elif response.status_code in [500, 502, 503, 504]:
                    wait_time = 30 * (attempt + 1)
                    news_logger.warning(
                        "Server error "
                        f"({response.status_code}), retrying in "
                        f"{wait_time} seconds..."
                    )
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()
                news_data = response.json()
                news_logger.info(
                    "Successfully fetched news data with "
                    f"{len(news_data.get('results', []))} items"
                )

                if "results" in news_data and news_data["results"]:
                    news_items = []
                    for item in news_data["results"]:
                        published_at = item.get("published_at", "")
                        try:
                            published_timestamp = int(
                                parser.parse(published_at).timestamp())
                        except Exception:
                            published_timestamp = int(time.time())

                        news_items.append({
                            'title': item.get("title", "Sem titulo"),
                            'body': item.get("body", ""),
                            'url': item.get("url", ""),
                            'imageurl': "",
                            'source': item.get(
                                "source", {}
                            ).get("title", "Desconhecido"),
                            'published_on': published_timestamp
                        })

                    return news_items

                return []

            except requests.exceptions.Timeout:
                news_logger.warning(
                    f"Timeout on attempt {attempt+1}, retrying...")
                if attempt < max_retries - 1:
                    time.sleep(5 * (attempt + 1))
                continue
            except requests.exceptions.RequestException as e:
                news_logger.error(f"Network error on attempt {attempt+1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(10 * (attempt + 1))
                continue
            except Exception as e:
                news_logger.error(
                    f"Unexpected error on attempt {attempt+1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5 * (attempt + 1))
                continue

        return None

    def _fetch_from_newsapi(self):
        """Fallback para NewsAPI."""
        try:
            api_key = os.getenv("NEWSAPI_KEY", "")
            if not api_key:
                news_logger.warning(
                    "NewsAPI key not configured, skipping fallback")
                return None

            url = "https://newsapi.org/v2/everything"
            params = {
                'q': 'bitcoin OR cryptocurrency OR crypto',
                'apiKey': api_key,
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 20
            }

            news_logger.info("Fetching news from NewsAPI as fallback")
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            news_data = response.json()
            if "articles" in news_data and news_data["articles"]:
                news_items = []
                for item in news_data["articles"]:
                    published_at = item.get("publishedAt", "")
                    try:
                        published_timestamp = int(
                            parser.parse(published_at).timestamp())
                    except Exception:
                        published_timestamp = int(time.time())

                    news_items.append(
                        {
                           "title": item.get("title", "Sem titulo"),
                           "body": item.get("description", ""),
                           "url": item.get("url", ""),
                           "imageurl": item.get("urlToImage", ""),
                           "source": item.get("source", {}).get(
                               "name",
                               "Desconhecido",
                            ),
                           "published_on": published_timestamp,
                        }
                    )

                news_logger.info(
                    f"Successfully fetched "
                    f"{len(news_items)} news items from NewsAPI"
                )
                return news_items
        except Exception as e:
            news_logger.error(f"Error fetching from NewsAPI: {e}")

        return None

    def analyze_sentiment(self, news_items):
        """Analisa o sentimento das noticias.
        Usa VADER (cripto) e TextBlob como fallback.
        """
        news_logger.info(
            f"Analyzing sentiment for {len(news_items)} news items")
        try:
            sia = SentimentIntensityAnalyzer()
            analyzed_count = 0
            for item in news_items:
                text = (item.get('title', '') + " " +
                        item.get('body', '')).strip()
                if text:
                    # Usar VADER para análise de sentimento
                    sentiment_scores = sia.polarity_scores(text)
                    compound_score = sentiment_scores['compound']
                    # Normalizar para -1 a 1 (VADER compound já está)
                    item['sentiment_score'] = round(compound_score, 3)
                    analyzed_count += 1
                    news_logger.debug(
                        news_logger.debug(
                            f"Sentiment for '{item['title'][:50]}...':"
                            f" {item['sentiment_score']}"
                        )
                    )
                else:
                    item['sentiment_score'] = 0.0
            news_logger.info(
                f"Successfully analyzed sentiment for {analyzed_count}/"
                f"{len(news_items)} items"
            )
            return news_items
        except Exception as e:
            news_logger.error(
                f"Error analyzing sentiment with VADER: {e}, using TextBlob "
                f"as fallback"
            )
            # Fallback para TextBlob
            try:
                analyzed_count = 0
                for item in news_items:
                    text = (item.get('title', '') + " " +
                            item.get('body', '')).strip()
                    if text:
                        blob = TextBlob(text)
                        sentiment = blob.sentiment.polarity
                        item['sentiment_score'] = round(sentiment, 3)
                        analyzed_count += 1
                    else:
                        item['sentiment_score'] = 0.0
                news_logger.info(
                    f"Fallback analysis completed for {analyzed_count} items"
                )
                return news_items
            except Exception as e2:
                news_logger.error(f"Error in TextBlob fallback: {e2}")
                for item in news_items:
                    item['sentiment_score'] = 0.0
                return news_items

    @staticmethod
    def get_average_sentiment_score():
        """Calcula o score de sentimento medio das noticias.
        Ultimas 24h, com logging e fallback.
        """
        news_logger.info(
            "Starting sentiment analysis for average score calculation")
        try:
            # Buscar notícias recentes
            news = NewsWorker().fetch_crypto_news()
            news_logger.info(
                f"Fetched {len(news)} news items for sentiment analysis")

            if not news:
                news_logger.warning(
                    "No news fetched from API, using fallback sentiment data")
                return NewsWorker._get_fallback_sentiment_score()

            # Filtrar notícias das últimas 24 horas
            current_time = time.time()
            recent_news = [
                item for item in news
                if current_time - item['published_on'] <= 86400
            ]
            news_logger.info(
                f"Filtered to {len(recent_news)} recent news items (last 24h)")

            if not recent_news:
                news_logger.warning(
                    "No recent news found, using fallback sentiment data")
                return NewsWorker._get_fallback_sentiment_score()

            # Analisar sentimento
            news_with_sentiment = NewsWorker().analyze_sentiment(recent_news)

            # Calcular media dos scores
            sentiment_scores = [
                item['sentiment_score']
                for item in news_with_sentiment
                if 'sentiment_score' in item
            ]
            news_logger.debug(
                f"Collected sentiment scores: {sentiment_scores}"
            )
            if sentiment_scores:
                average_sentiment = (
                    sum(sentiment_scores) / len(sentiment_scores)
                )
                news_logger.info(
                    f"Calculated average sentiment: {average_sentiment:.3f} "
                    f"from {len(sentiment_scores)} scores"
                )
                return round(average_sentiment, 3)
            else:
                news_logger.warning(
                    "No valid sentiment scores found, using fallback "
                    "sentiment data"
                )
                return NewsWorker._get_fallback_sentiment_score()

        except Exception as e:
            news_logger.error(
                news_logger.error(
                    f"Error calculating average sentiment score: {e}",
                    exc_info=True
                )
            )
            return NewsWorker._get_fallback_sentiment_score()

    @staticmethod
    def _get_fallback_sentiment_score():
        """Retorna um score de sentimento de fallback.
        Baseado em dados simulados ou neutro.
        """
        import random
        # Simular um score de sentimento baseado no mercado atual
        # Durante bear markets, tende a ser mais negativo;
        # durante bull markets, mais positivo
        # Para este caso, usaremos uma distribuicao normal centrada
        # em 0 com baixa variancia
        fallback_score = random.gauss(0, 0.1)  # média 0, desvio padrão 0.1
        fallback_score = max(-0.5, min(0.5, fallback_score)
                             )  # limitar entre -0.5 e 0.5
        news_logger.info(
            f"Using fallback sentiment score: {fallback_score:.3f}")
        return round(fallback_score, 3)

    def stop(self):
        """Parar thread worker"""
        self.running = False
        self.wait()
