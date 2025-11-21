from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QLabel,
    QGridLayout,
    QLineEdit,
    QHBoxLayout,
    QPushButton,
    )
from database import DatabaseManager


class APIKeyDialog(QDialog):
    """Diálogo para inserir chaves API da Binance"""
    def __init__(self, parent=None):
        super().__init__(parent)

        # Obter chaves API atuais
        api_key, api_secret = DatabaseManager.get_api_keys()

        self.setWindowTitle("Configurar API Binance")
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)

        info_label = QLabel(
            "Configure sua chave API da Binance,"
            "para obter dados em tempo real:"
            )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        form_layout = QGridLayout()

        api_key_label = QLabel("API Key:")
        form_layout.addWidget(api_key_label, 0, 0)

        self.api_key_input = QLineEdit()
        self.api_key_input.setText(api_key)
        self.api_key_input.setPlaceholderText("Insira sua API Key da Binance")
        form_layout.addWidget(self.api_key_input, 0, 1)

        api_secret_label = QLabel("API Secret:")
        form_layout.addWidget(api_secret_label, 1, 0)

        self.api_secret_input = QLineEdit()
        self.api_secret_input.setText(api_secret)
        self.api_secret_input.setPlaceholderText(
            "Insira seu API Secret da Binance"
            )
        self.api_secret_input.setEchoMode(QLineEdit.Password)
        form_layout.addWidget(self.api_secret_input, 1, 1)

        layout.addLayout(form_layout)

        note_label = QLabel(
            "Nota: Para apenas visualizar dados públicos de mercado,"
            "você pode deixar estes campos vazios."
            )
        note_label.setWordWrap(True)
        layout.addWidget(note_label)

        buttons_layout = QHBoxLayout()

        cancel_button = QPushButton("Cancelar")
        cancel_button.clicked.connect(self.reject)
        buttons_layout.addWidget(cancel_button)

        save_button = QPushButton("Salvar")
        save_button.clicked.connect(self.accept)
        buttons_layout.addWidget(save_button)

        layout.addLayout(buttons_layout)

    def get_api_keys(self):
        """Retorna as chaves API inseridas"""
        return self.api_key_input.text(), self.api_secret_input.text()
