from kivy.lang import Builder
from kivymd.uix.screenmanager import MDScreenManager
from controllers.welcome_controller import WelcomeScreen
from controllers.login_controller import LoginScreen
from controllers.register_controller import RegisterScreen
from controllers.policy_controller import PolicyScreen
from controllers.terms_controller import TermsScreen

# IMPORTANTE: importar *antes* de carregar o .kv
Builder.load_file("views/welcome_page/welcome.kv")
Builder.load_file("views/login/login.kv")
Builder.load_file("views/register_page/register.kv")
Builder.load_file("views/policy_page/policy.kv")
Builder.load_file("views/terms_page/terms.kv")


class ScreenManagement(MDScreenManager):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_widget(WelcomeScreen(name="welcome"))
        self.add_widget(LoginScreen(name="login"))
        self.add_widget(RegisterScreen(name="register"))
        self.add_widget(RegisterScreen(name="register"))
        self.add_widget(PolicyScreen(name="policy"))
        self.add_widget(TermsScreen(name="terms"))
        
        self.current = "welcome"
        
        
