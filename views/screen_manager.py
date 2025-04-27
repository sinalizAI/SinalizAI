from kivy.lang import Builder
from kivymd.uix.screenmanager import MDScreenManager
from controllers.welcome_controller import WelcomeScreen
from controllers.login_controller import LoginScreen
from controllers.register_controller import RegisterScreen
from controllers.policy_controller import PolicyScreen
from controllers.terms_controller import TermsScreen
from controllers.home_controller import HomeScreen
from controllers.profile_controller import ProfileScreen
from controllers.faq_controller import FaqScreen
from controllers.feedback_controller import FeedbackScreen
from controllers.about_us_controller import AboutUsScreen
from controllers.edit_profile_controller import EditScreen
#from controllers.loading_controller import LoadingScreen

Builder.load_file("views/welcome_page/welcome.kv")
Builder.load_file("views/login/login.kv")
Builder.load_file("views/register_page/register.kv")
Builder.load_file("views/policy_page/policy.kv")
Builder.load_file("views/terms_page/terms.kv")
Builder.load_file("views/home_page/home.kv")
Builder.load_file("views/profile_page/profile.kv")
Builder.load_file("views/faq_page/faq.kv")
Builder.load_file("views/feedback_page/feedback.kv")
Builder.load_file("views/about_us_page/about_us.kv")
Builder.load_file("views/edit_profile_page/edit_profile.kv")
#Builder.load_file("views/loading_page/loading.kv")


class ScreenManagement(MDScreenManager):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_widget(WelcomeScreen(name="welcome"))
        self.add_widget(LoginScreen(name="login"))
        self.add_widget(RegisterScreen(name="register"))
        self.add_widget(PolicyScreen(name="policy"))
        self.add_widget(TermsScreen(name="terms"))
        self.add_widget(HomeScreen(name="home"))
        self.add_widget(ProfileScreen(name="profile"))
        self.add_widget(FaqScreen(name="faq"))
        self.add_widget(FeedbackScreen(name="feedback"))
        self.add_widget(AboutUsScreen(name="aboutus"))
        self.add_widget(EditScreen(name="edit"))
        #self.add_widget(LoadingScreen(name="loading"))
        self.current = "welcome"
