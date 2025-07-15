#!/usr/bin/env python3
"""
Teste simples de conectividade com Firebase
"""

import requests

def test_connectivity():
    """Testa conectividade básica"""
    
    print("🔗 Testando conectividade com Firebase...")
    
    # URL de teste simples
    test_url = "https://identitytoolkit.googleapis.com/v1/accounts:signUp?key=***REMOVED***"
    
    # Payload de teste simples
    test_payload = {
        "email": "test@example.com",
        "password": "123456",
        "returnSecureToken": True
    }
    
    try:
        print(f"📡 Enviando requisição para Firebase...")
        response = requests.post(test_url, json=test_payload, timeout=10)
        
        print(f"📊 Status Code: {response.status_code}")
        print(f"📄 Response Text: {response.text[:200]}...")
        
        if response.status_code in [200, 400]:  # 400 é esperado para e-mail inválido
            print("✅ Conectividade com Firebase: OK")
            return True
        else:
            print("❌ Problema de conectividade")
            return False
            
    except requests.exceptions.Timeout:
        print("⏰ Timeout - conexão lenta")
        return False
    except requests.exceptions.ConnectionError:
        print("🌐 Erro de conexão - verifique internet")
        return False
    except Exception as e:
        print(f"🚨 Erro: {e}")
        return False

if __name__ == "__main__":
    test_connectivity()
