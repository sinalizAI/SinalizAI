from kivymd.app import MDApp
from kivy.lang import Builder
from kivy.uix.screenmanager import Screen
from kivy.core.window import Window

Builder.load_file("views/home_page/home.kv")

class HomeScreen(Screen):
    pass

class Home(MDApp):
    def build(self):
        Window.size = (360, 640)  
        return HomeScreen()     

if __name__ == '__main__':
    Home().run()
