
from kivy.config import Config
Config.set('graphics', 'width', '900')
Config.set('graphics', 'height', '700')

from kivy.core.window import Window
from kivymd.app import MDApp
from views.screen_manager import ScreenManagement


class SinalizAIApp(MDApp):
    def build(self):
        Window.size = (900, 700)
        return ScreenManagement()
        
if __name__ == "__main__":
    SinalizAIApp().run()
