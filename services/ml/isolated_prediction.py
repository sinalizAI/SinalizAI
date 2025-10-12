"""
Script isolado para executar predição do modelo MoViNet sem crashes
"""
import os
import sys
import numpy as np
import tensorflow as tf
import pickle
import tempfile

def load_model_and_predict():
    """Carrega modelo e executa predição isoladamente"""
    try:
        # Le os dados de entrada do arquivo temporário
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        
        with open(input_file, 'rb') as f:
            input_tensor = pickle.load(f)
        
        print(f"🔧 Processo isolado: carregando modelo...")
        
        # Força CPU apenas
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        tf.config.set_visible_devices([], 'GPU')
        
        # Muda para diretório do modelo
        os.chdir('services/ml')
        
        # Carrega modelo
        model = tf.keras.models.load_model('movinet_libras_final_base.keras')
        
        print(f"🔧 Processo isolado: executando predição...")
        print(f"    Input shape: {input_tensor.shape}")
        
        # Executa predição
        with tf.device('/CPU:0'):
            predictions = model.predict(input_tensor, verbose=0)
        
        print(f"🔧 Processo isolado: predição concluída")
        
        # Salva resultado
        result = {
            'predictions': predictions,
            'success': True,
            'error': None
        }
        
        with open(output_file, 'wb') as f:
            pickle.dump(result, f)
        
        print(f"✅ Processo isolado: resultado salvo")
        
    except Exception as e:
        print(f"❌ Erro no processo isolado: {e}")
        result = {
            'predictions': None,
            'success': False,
            'error': str(e)
        }
        
        with open(output_file, 'wb') as f:
            pickle.dump(result, f)

if __name__ == '__main__':
    load_model_and_predict()