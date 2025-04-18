from kivymd.app import MDApp
from kivy.lang import Builder
from kivy.uix.screenmanager import Screen
from kivy.core.window import Window

Builder.load_file("faq.kv")

class FaqScreen(Screen):
    pass

class   Faq(MDApp):
    def build(self):
        Window.size = (360, 640)  
        return FaqScreen()     

if __name__ == '__main__':
    Faq().run()