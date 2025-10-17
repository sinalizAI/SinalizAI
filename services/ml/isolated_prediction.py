"""
Script isolado para executar predição do modelo MoViNet sem crashes
"""
import os
import sys
import numpy as np
import tensorflow as tf
import pickle
import tempfile

# Try to ensure stdout/stderr use UTF-8 on platforms (helps on Windows)
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

def load_model_and_predict():
    """Carrega modelo e executa predição isoladamente"""
    try:
        # Le os dados de entrada do arquivo temporário
        input_file = sys.argv[1]
        output_file = sys.argv[2]

        with open(input_file, 'rb') as f:
            input_tensor = pickle.load(f)

        print("Processo isolado: carregando modelo...")

        # Força CPU apenas
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        try:
            tf.config.set_visible_devices([], 'GPU')
        except Exception:
            # algumas builds do TF no Windows não suportam set_visible_devices
            pass

        # Muda para diretório do modelo
        os.chdir('services/ml')

        # Carrega modelo
        model = tf.keras.models.load_model('movinet_libras_final_base.keras')

        print("Processo isolado: executando predição...")
        print(f"    Input shape: {input_tensor.shape}")

        # Executa predição
        with tf.device('/CPU:0'):
            predictions = model.predict(input_tensor, verbose=0)

        print("Processo isolado: predição concluída")

        # Salva resultado
        result = {
            'predictions': predictions,
            'success': True,
            'error': None
        }

        with open(output_file, 'wb') as f:
            pickle.dump(result, f)

        print("Processo isolado: resultado salvo")

    except Exception as e:
        # Avoid printing characters that may fail on some consoles
        try:
            print(f"Erro no processo isolado: {e}")
        except Exception:
            # fallback simple write to stderr
            try:
                sys.stderr.write("Erro no processo isolado\n")
                sys.stderr.write(str(e) + "\n")
            except Exception:
                pass
        result = {
            'predictions': None,
            'success': False,
            'error': str(e)
        }

        try:
            with open(output_file, 'wb') as f:
                pickle.dump(result, f)
        except Exception:
            # se não for possível escrever o arquivo, apenas ignore
            pass

if __name__ == '__main__':
    load_model_and_predict()