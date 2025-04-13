from kivymd.app import MDApp
from kivy.lang import Builder
from kivy.uix.screenmanager import Screen
from kivy.core.window import Window

Builder.load_file("About_us.kv")

class AboutUsScreen(Screen):
    pass

class   AboutUs(MDApp):
    def build(self):
        Window.size = (360, 640)  
        return AboutUsScreen()     

if __name__ == '__main__':
    AboutUs().run()