from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QLabel,
)
from PyQt5.QtCore import Qt


class OrdersWidget(QWidget):
    def __init__(self, binance_connector, symbol: str = 'BTCUSDT'):
        super().__init__()
        self.binance = binance_connector
        self.symbol = symbol

        layout = QVBoxLayout(self)

        header = QHBoxLayout()
        title = QLabel("Ordens Ativas")
        title.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        refresh_btn = QPushButton("Atualizar")
        refresh_btn.clicked.connect(self.refresh_orders)
        header.addWidget(title)
        header.addStretch()
        header.addWidget(refresh_btn)
        layout.addLayout(header)

        self.table = QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels([
            "OrderId",
            "Symbol",
            "Side",
            "Type",
            "Qty",
            "Price",
            "Status",
            "TimeInForce"
        ])
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        layout.addWidget(self.table)

        self.refresh_orders()

    def refresh_orders(self):
        try:
            orders = self.binance.get_open_orders(self.symbol) or []
        except Exception:
            orders = []
        self.table.setRowCount(0)
        for o in orders:
            row = self.table.rowCount()
            self.table.insertRow(row)

            def val(k, default=''):
                return str(o.get(k, default)
                           if o.get(k, default) is not None else '')
            items = [
                val('orderId'), val('symbol'), val('side'), val('type'),
                val('origQty'), val('price'), val('status'), val('timeInForce')
            ]
            for c, text in enumerate(items):
                self.table.setItem(row, c, QTableWidgetItem(text))
