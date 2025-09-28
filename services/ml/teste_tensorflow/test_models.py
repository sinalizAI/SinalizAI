#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teste do modelo TensorFlow Lite - SinalizAI
Script para testar rapidamente os modelos .tflite
"""

import os
import sys
from pathlib import Path

# Adicionar o diretório atual ao path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from detect_tensorflow import run_tensorflow

def test_model_float16():
    """Testa o modelo float16"""
    print("=" * 50)
    print("TESTANDO MODELO FLOAT16")
    print("=" * 50)
    
    try:
        run_tensorflow(
            weights=current_dir / "best_float16.tflite",
            source="0",  # Webcam
            view_img=True,
            save_txt=True,
            save_csv=True,
            conf_thres=0.5,
            iou_thres=0.45,
            project=current_dir / "runs/test_float16",
            name="test",
            exist_ok=True
        )
        print("✅ Teste do modelo float16 concluído!")
    except Exception as e:
        print(f"❌ Erro no teste do modelo float16: {e}")

def test_model_float32():
    """Testa o modelo float32"""
    print("=" * 50)
    print("TESTANDO MODELO FLOAT32")
    print("=" * 50)
    
    try:
        run_tensorflow(
            weights=current_dir / "best_float32.tflite",
            source="0",  # Webcam
            view_img=True,
            save_txt=True,
            save_csv=True,
            conf_thres=0.5,
            iou_thres=0.45,
            project=current_dir / "runs/test_float32",
            name="test",
            exist_ok=True
        )
        print("✅ Teste do modelo float32 concluído!")
    except Exception as e:
        print(f"❌ Erro no teste do modelo float32: {e}")

def check_requirements():
    """Verifica se as dependências estão instaladas"""
    print("Verificando dependências...")
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow: {tf.__version__}")
    except ImportError:
        print("❌ TensorFlow não encontrado. Instale com: pip install tensorflow")
        return False
    
    try:
        import cv2
        print(f"✅ OpenCV: {cv2.__version__}")
    except ImportError:
        print("❌ OpenCV não encontrado. Instale com: pip install opencv-python")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
    except ImportError:
        print("❌ NumPy não encontrado. Instale com: pip install numpy")
        return False
    
    return True

if __name__ == "__main__":
    print("SinalizAI - Teste de Modelos TensorFlow Lite")
    print("=" * 60)
    
    if not check_requirements():
        print("\n❌ Instale as dependências necessárias antes de continuar.")
        sys.exit(1)
    
    print("\nEscolha uma opção:")
    print("1. Testar modelo Float16")
    print("2. Testar modelo Float32")
    print("3. Testar ambos os modelos")
    print("0. Sair")
    
    choice = input("\nDigite sua escolha (0-3): ").strip()
    
    if choice == "1":
        test_model_float16()
    elif choice == "2":
        test_model_float32()
    elif choice == "3":
        print("\n🚀 Testando ambos os modelos...")
        test_model_float16()
        print("\n" + "=" * 60)
        test_model_float32()
    elif choice == "0":
        print("Saindo...")
    else:
        print("❌ Opção inválida!")