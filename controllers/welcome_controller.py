from kivymd.app import MDApp
from kivy.lang import Builder
from kivy.uix.screenmanager import Screen
from kivy.core.window import Window

Builder.load_file("views/welcome_page/welcome.kv")

class WelcomeScreen(Screen):
    pass

class Welcome(MDApp):
    def build(self):
        Window.size = (360, 640)  
        return WelcomeScreen()     

if __name__ == '__main__':
    Welcome().run()
