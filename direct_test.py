#!/usr/bin/env python3
"""
Teste direto de reset de senha
"""

import requests
import json

def test_direct_reset():
    """Teste direto de reset"""
    
    print("🔄 === TESTE DIRETO DE RESET DE SENHA ===\n")
    
    # Use um e-mail que você criou ou que existe
    test_email = "test@example.com"  # Este foi criado no teste anterior
    
    print(f"📧 Testando reset para: {test_email}")
    
    url = "https://identitytoolkit.googleapis.com/v1/accounts:sendOobCode?key=***REMOVED***"
    
    payload = {
        "requestType": "PASSWORD_RESET",
        "email": test_email
    }
    
    print(f"🌐 URL: {url}")
    print(f"📦 Payload: {json.dumps(payload, indent=2)}")
    
    try:
        print(f"\n📡 Enviando requisição...")
        response = requests.post(url, json=payload, timeout=10)
        
        print(f"📊 Status Code: {response.status_code}")
        print(f"📄 Response: {response.text}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n🎉 === SUCESSO! ===")
            print(f"✅ E-mail de reset enviado!")
            print(f"📧 Para: {data.get('email')}")
            print(f"🔗 Tipo: {data.get('kind')}")
            print(f"\n📬 VERIFIQUE:")
            print(f"   • Caixa de entrada de: {test_email}")
            print(f"   • Pasta de SPAM/LIXO ELETRÔNICO")
            print(f"   • Remetente: noreply@sinalizai.firebaseapp.com")
            print(f"   • Assunto: Redefinir sua senha - SinalizAI")
            
            return True
        else:
            error_data = response.json()
            error_message = error_data.get("error", {}).get("message", "")
            
            print(f"\n❌ ERRO: {error_message}")
            
            if "EMAIL_NOT_FOUND" in error_message:
                print(f"\n💡 SOLUÇÃO:")
                print(f"   Este e-mail não existe no Firebase")
                print(f"   Vou tentar criar um usuário primeiro...")
                
                # Tenta criar o usuário
                if create_user_for_test(test_email):
                    print(f"   ✅ Usuário criado! Tentando reset novamente...")
                    return test_direct_reset()  # Recursão para tentar novamente
                else:
                    print(f"   ❌ Falha ao criar usuário")
                    return False
            else:
                print(f"   Erro não identificado: {error_message}")
                return False
        
    except Exception as e:
        print(f"\n🚨 ERRO: {e}")
        return False

def create_user_for_test(email):
    """Cria um usuário para teste"""
    
    print(f"\n👤 Criando usuário: {email}")
    
    url = "https://identitytoolkit.googleapis.com/v1/accounts:signUp?key=***REMOVED***"
    
    payload = {
        "email": email,
        "password": "TesteSenha123!",
        "returnSecureToken": True
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        
        if response.status_code == 200:
            print(f"✅ Usuário criado com sucesso!")
            return True
        else:
            error_data = response.json()
            error_message = error_data.get("error", {}).get("message", "")
            
            if "EMAIL_EXISTS" in error_message:
                print(f"✅ Usuário já existe (perfeito!)")
                return True
            else:
                print(f"❌ Erro ao criar: {error_message}")
                return False
                
    except Exception as e:
        print(f"🚨 Erro ao criar usuário: {e}")
        return False

if __name__ == "__main__":
    test_direct_reset()
