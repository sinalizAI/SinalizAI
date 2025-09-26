#!/usr/bin/env python3
"""
Configuração para contornar problemas do MTDev
"""

import os
os.environ['KIVY_WINDOW'] = 'sdl2'
os.environ['KIVY_GL_BACKEND'] = 'gl'

from kivy.config import Config
# Desabilita o MTDev que está causando problema
Config.set('input', 'mtdev', '')
# Configura para usar apenas mouse e teclado
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
# Desabilita multisampling que pode causar problemas
Config.set('graphics', 'multisamples', '0')

# Agora importa o main
if __name__ == "__main__":
    import main