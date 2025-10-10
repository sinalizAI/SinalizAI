
from kivy.config import Config
Config.set('graphics', 'width', '360')
Config.set('graphics', 'height', '640')

from kivy.core.window import Window
from kivymd.app import MDApp
from views.screen_manager import ScreenManagement


class SinalizAIApp(MDApp):
    def build(self):
        Window.size = (360, 640)  # Tamanho médio mobile (padrão smartphone)
        return ScreenManagement()
        
if __name__ == "__main__":
    SinalizAIApp().run()
