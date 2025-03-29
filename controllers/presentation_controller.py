from kivymd.app import MDApp
from kivy.lang import Builder
from kivy.uix.screenmanager import Screen
from kivy.core.window import Window

Builder.load_file("views/presentation_page/presentation.kv")

class PresentationScreen(Screen):
    pass

class Presentation(MDApp):  
    def build(self):
        Window.size = (360, 640)  
        return PresentationScreen()

if __name__ == '__main__':
    Presentation().run()  