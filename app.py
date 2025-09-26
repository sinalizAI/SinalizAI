
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

from main import SinalizAIApp

if __name__ == '__main__':
    try:
        SinalizAIApp().run()
    except Exception as e:
        print(f"Erro ao iniciar a aplicação: {e}")
        sys.exit(1)