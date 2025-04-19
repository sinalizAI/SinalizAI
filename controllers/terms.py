from kivymd.app import MDApp
from kivy.lang import Builder
from kivy.uix.screenmanager import Screen
from kivy.core.window import Window

Builder.load_file("terms.kv")

class TermsScreen(Screen):
    pass

class Terms(MDApp):
    def build(self):
        Window.size = (360, 640)  
        return TermsScreen()     

if __name__ == '__main__':
     Terms().run()
