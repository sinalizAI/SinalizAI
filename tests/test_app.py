
def test_translate_alphabet_benchmark(benchmark):
    from controllers.home_controller import HomeScreen
    screen = HomeScreen()
    screen.manager = MagicMock()
    def call_translate_alphabet():
        return screen.translate_alphabet()
    result = benchmark(call_translate_alphabet)
    assert result is None



def test_signs_detection_benchmark(benchmark):
    import numpy as np
    from controllers.signs_detection_controller import SignsDetectionScreen
    screen = SignsDetectionScreen()
    screen.model_loaded = True
    screen.FRAME_COUNT = 16
    screen.HEIGHT = 172
    screen.WIDTH = 172

    screen.model = MagicMock()

    fake_frames = [np.random.randint(0, 255, (172, 172, 3), dtype=np.uint8) for _ in range(16)]
    screen.recorded_frames = [screen.preprocess_frame(f) for f in fake_frames]
    screen.current_state = "PROCESSING"
    def process_signal():
        screen.update_state_machine(fake_frames[0])
        return screen.prediction_result
    result = benchmark(process_signal)
    assert isinstance(result, str)






import pytest
from unittest.mock import MagicMock
from kivymd.app import MDApp
from models import firebase_auth_model
from controllers.login_controller import LoginScreen
from controllers.register_controller import RegisterScreen
from controllers.forgot_password_controller import ForgotScreen
from controllers.reset_confirmation_controller import ConfirmationScreen
from controllers.profile_controller import ProfileScreen
from services.ml.isolated_prediction import load_model_and_predict
from pytest_benchmark.fixture import BenchmarkFixture



class TestApp(MDApp):
    def build(self):
        pass

def setup_module(module):

    if not MDApp.get_running_app():
        app = TestApp()
        app.theme_cls.primary_palette = "Blue"
        app.theme_cls.material_style = "M3"
        app.run = lambda *a, **kw: None
        app._run_prepare()


def test_register_benchmark(benchmark):
    screen = RegisterScreen()
    screen.ids = {"create_button": MagicMock(disabled=False)}
    screen.validate_email = MagicMock(return_value=True)
    screen.validate_password = MagicMock(return_value=True)
    screen.show_error = MagicMock()
    screen.manager = MagicMock()
    def do_register():
        return screen.do_register("teste@exemplo.com", "Senha@123", "Senha@123", "Usuário Teste")
    result = benchmark(do_register)
    assert result is None


def test_login_benchmark(benchmark):
    screen = LoginScreen()
    screen.validate_email = MagicMock(return_value=True)
    screen.manager = MagicMock()
    screen.go_to_home = MagicMock()
    def do_login():
        return screen.do_login("teste@exemplo.com", "Senha@123")
    result = benchmark(do_login)
    assert result is None



def test_reset_password_benchmark(benchmark):
    screen = ForgotScreen()
    screen.ids = {"forgot_input": MagicMock(text="teste@exemplo.com")}
    screen.validate_email = MagicMock(return_value=True)
    screen.manager = MagicMock()
    screen.go_to_reset_confirmation = MagicMock()

    screen.get_friendly_error = MagicMock(return_value="Erro simulado")
    def send_reset_email():
        return screen.send_reset_email()
    result = benchmark(send_reset_email)
    assert result is None


def test_update_display_name_benchmark(benchmark):
    def update_name():
        return firebase_auth_model.update_display_name("token_fake", "Novo Nome")
    result = benchmark(update_name)
    assert isinstance(result, tuple)



def test_register():
    screen = RegisterScreen()
    screen.ids = {"create_button": MagicMock(disabled=False)}
    screen.validate_email = MagicMock(return_value=True)
    screen.validate_password = MagicMock(return_value=True)
    screen.show_error = MagicMock()
    screen.manager = MagicMock()
    result = screen.do_register("teste@exemplo.com", "Senha@123", "Senha@123", "Usuário Teste")
    assert result is None

def test_login():
    screen = LoginScreen()
    screen.validate_email = MagicMock(return_value=True)
    screen.manager = MagicMock()
    screen.go_to_home = MagicMock()
    result = screen.do_login("teste@exemplo.com", "Senha@123")
    assert result is None

def test_reset_password():
    screen = ForgotScreen()
    screen.ids = {"forgot_input": MagicMock(text="teste@exemplo.com")}
    screen.validate_email = MagicMock(return_value=True)
    screen.manager = MagicMock()
    screen.go_to_reset_confirmation = MagicMock()
    result = screen.send_reset_email()
    assert result is None

def test_confirmation_screen():
    screen = ConfirmationScreen()
    screen.user_email = "teste@exemplo.com"
    screen.manager = MagicMock()
    screen.go_to_back = MagicMock()
    result = screen.close_screen()
    assert result is None

def test_delete_account():
    screen = ProfileScreen()
    class Manager:
        user_data = {"idToken": "token_fake"}
    screen.manager = Manager()
    screen.go_to_welcome = MagicMock()
    result = screen.excluir_dados_usuario()
    assert result is None

def test_update_email():
    result = firebase_auth_model.update_email("token_fake", "novoemail@exemplo.com")
    assert isinstance(result, tuple)

def test_update_display_name():
    result = firebase_auth_model.update_display_name("token_fake", "Novo Nome")
    assert isinstance(result, tuple)

def test_update_profile():
    result = firebase_auth_model.update_profile("token_fake", new_email="novoemail@exemplo.com", new_display_name="Novo Nome")
    assert isinstance(result, tuple)

def test_model_prediction_callable():

    assert callable(load_model_and_predict)
