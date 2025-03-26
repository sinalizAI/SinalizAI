from kivy.core.window import Window
from kivymd.app import MDApp
from kivy.lang import Builder
from kivy.uix.screenmanager import Screen


Builder.load_file("loading.kv")

class LoadingScreen(Screen):
    pass

class Loading(MDApp):
    def build(self):
        Window.size = (360, 640)
        return LoadingScreen()

if __name__ == '__main__':
    Loading().run()