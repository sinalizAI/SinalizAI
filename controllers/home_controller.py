from utils.base_screen import BaseScreen
from utils.message_helper import show_message

class HomeScreen(BaseScreen):

    def on_enter(self):
        import time
        self._start_time = time.time()
        super().on_enter()
        print(" HomeScreen aberta")

    def translate_alphabet(self):
        import time
        self._benchmark_start = time.time()
        try:
            self.manager.current = 'detection'
        except Exception as e:
            print(f"Erro ao abrir câmera: {e}")
            show_message("Não foi possível abrir a câmera para reconhecimento.")
    def log_alphabet_result(self):
        import time
        if hasattr(self, '_benchmark_start'):
            tempo_total = time.time() - self._benchmark_start
            print(f" Tempo total (input->resultado alfabeto): {tempo_total:.3f} segundos")
            from utils.benchmark_logger import log_benchmark
            log_benchmark('modelo_alfabeto_total', tempo_total)
    
    def translate_signs(self):
        import time
        self._benchmark_start_sinais = time.time()
        try:

            self.manager.current = 'signs_detection'
        except Exception as e:
            print(f"Erro ao abrir câmera: {e}")
            show_message("Não foi possível abrir a câmera para reconhecimento de sinais.")

    def log_signs_result(self, resultado=None):
        import time
        if hasattr(self, '_benchmark_start_sinais'):
            tempo_total = time.time() - self._benchmark_start_sinais
            print(f" Tempo total (input->resultado sinais): {tempo_total:.3f} segundos")
            from utils.benchmark_logger import log_benchmark
            log_benchmark('tradutor_sinais_total', tempo_total, {'resultado': resultado})
