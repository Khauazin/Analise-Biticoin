import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging
import os
from Logger import iniciar_logger


class AlertSystem:
    """Sistema de alertas para falhas críticas de API e monitoramento."""

    def __init__(
        self, smtp_server: str = "smtp.gmail.com",
        smtp_port: int = 587,
        sender_email: str = "", sender_password: str = ""
    ):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = sender_email or os.getenv("ALERT_EMAIL", "")
        self.sender_password = sender_password or os.getenv(
            "ALERT_PASSWORD", "")
        self.logger = iniciar_logger("AlertSystem")
        self.alerts_enabled = bool(self.sender_email and self.sender_password)

        if not self.alerts_enabled:
            self.logger.warning(
                "Alert system disabled:"
                "email credentials not configured"
                )

    def send_alert(
        self,
        subject: str,
        message: str,
        recipient: str = None
    ) -> bool:
        """Envia um alerta por email."""
        if not self.alerts_enabled:
            self.logger.warning(f"Alert not sent (disabled): {subject}")
            return False

        if not recipient:
            recipient = self.sender_email  # Fallback para si mesmo

        try:
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = recipient
            msg['Subject'] = f"[CRYPTO TRADER ALERT] {subject}"

            msg.attach(MIMEText(message, 'plain'))

            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.sender_email, self.sender_password)
            text = msg.as_string()
            server.sendmail(self.sender_email, recipient, text)
            server.quit()

            self.logger.info(f"Alert sent successfully: {subject}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to send alert: {e}")
            return False

    def alert_api_failure(self, api_name: str, error_details: str) -> None:
        """Alerta para falha de API."""
        subject = f"API Failure: {api_name}"
        message = f"""
        Critical API Failure Detected!

        API: {api_name}
        Error Details: {error_details}
        Timestamp: {logging.Formatter().formatTime(logging.LogRecord(
            "", 0, "", 0, "", (), None))}

        Please check the system and API status immediately.
        """
        self.send_alert(subject, message)

    def alert_rate_limit(self, api_name: str, retry_after: int = None) -> None:
        """Alerta para rate limiting."""
        subject = f"Rate Limit Hit: {api_name}"
        message = f"""
        Rate Limit Exceeded!

        API: {api_name}
        Retry After: {retry_after} seconds
        Timestamp: {logging.Formatter().formatTime(logging.LogRecord(
            "", 0, "", 0, "", (), None))}

        System will retry automatically with backoff.
        """
        self.send_alert(subject, message)

    def alert_system_health(
        self,
        component: str,
        status: str,
        details: str = ""
    ) -> None:
        """Alerta para status de saúde do sistema."""
        subject = f"System Health: {component} - {status.upper()}"
        message = f"""
        System Health Alert!

        Component: {component}
        Status: {status}
        Details: {details}
        Timestamp: {logging.Formatter().formatTime(logging.LogRecord(
            "", 0, "", 0, "", (), None))}

        Please investigate if status is CRITICAL.
        """
        if status.upper() == "CRITICAL":
            self.send_alert(subject, message)

    def log_alert(self, level: str, message: str) -> None:
        """Log de alerta sem enviar email (para casos não críticos)."""
        if level.upper() == "ERROR":
            self.logger.error(f"ALERT: {message}")
        elif level.upper() == "WARNING":
            self.logger.warning(f"ALERT: {message}")
        else:
            self.logger.info(f"ALERT: {message}")


# Instância global do sistema de alertas
alert_system = AlertSystem()


def initialize_alerts(
    smtp_server: str = None,
    smtp_port: int = None,
    sender_email: str = None,
    sender_password: str = None,
) -> None:
    """Inicializa o sistema de alertas com configurações customizadas."""
    global alert_system
    alert_system = AlertSystem(
        smtp_server,
        smtp_port,
        sender_email,
        sender_password
        )
