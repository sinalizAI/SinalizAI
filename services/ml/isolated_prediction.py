


import os
import sys
import numpy as np
import tensorflow as tf
import pickle
import tempfile


try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

def load_model_and_predict():
    
    try:

        input_file = sys.argv[1]
        output_file = sys.argv[2]

        with open(input_file, 'rb') as f:
            input_tensor = pickle.load(f)

        print("Processo isolado: carregando modelo...")


        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        try:
            tf.config.set_visible_devices([], 'GPU')
        except Exception:

            pass


        os.chdir('services/ml')


        model = tf.keras.models.load_model('movinet_libras_final_base.keras')

        print("Processo isolado: executando predição...")
        print(f"    Input shape: {input_tensor.shape}")


        with tf.device('/CPU:0'):
            predictions = model.predict(input_tensor, verbose=0)

        print("Processo isolado: predição concluída")


        result = {
            'predictions': predictions,
            'success': True,
            'error': None
        }

        with open(output_file, 'wb') as f:
            pickle.dump(result, f)

        print("Processo isolado: resultado salvo")

    except Exception as e:

        try:
            print(f"Erro no processo isolado: {e}")
        except Exception:

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

            pass

if __name__ == '__main__':
    load_model_and_predict()