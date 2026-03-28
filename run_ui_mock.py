import sys
from unittest.mock import MagicMock

# Create a mock for tensorflow to bypass installation error on Py 3.14
sys.modules['tensorflow'] = MagicMock()
sys.modules['tensorflow.keras'] = MagicMock()
sys.modules['tensorflow.keras.models'] = MagicMock()
sys.modules['tensorflow.keras.preprocessing'] = MagicMock()
sys.modules['tensorflow.keras.preprocessing.image'] = MagicMock()
sys.modules['tensorflow.keras.layers'] = MagicMock()

import src.gui_app
from PyQt5.QtWidgets import QApplication

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = src.gui_app.SmartParkApp()
    window.show()
    sys.exit(app.exec_())
