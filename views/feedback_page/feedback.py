from kivymd.app import MDApp
from kivy.lang import Builder
from kivy.uix.screenmanager import Screen
from kivy.core.window import Window

Builder.load_file("feedback.kv")

class FeedbackScreen(Screen):
    pass

class   Feedback(MDApp):
    def build(self):
        Window.size = (360, 640)  
        return FeedbackScreen() 

if __name__ == '__main__':
    Feedback().run()