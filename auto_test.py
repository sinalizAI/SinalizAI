#!/usr/bin/env python3
"""
Teste automatizado para criar usuário e testar reset
"""

import requests
import json

# Configuração do seu Firebase
firebase_config = {
    "apiKey": "***REMOVED***"
}

def auto_test():
    """Teste automatizado completo"""
    
    print("🤖 === TESTE AUTOMATIZADO ===\n")
    
    # E-mail de teste (você pode mudar este)
    test_email = "teste.sinalizai@gmail.com"
    test_password = "TesteSenha123!"
    
    print(f"📧 E-mail de teste: {test_email}")
    print(f"🔑 Senha de teste: {test_password}")
    
    # Passo 1: Tentar criar usuário
    print(f"\n🔄 PASSO 1: Criando usuário...")
    create_success = create_user(test_email, test_password)
    
    if create_success or "already exists":
        print(f"\n🔄 PASSO 2: Testando reset de senha...")
        reset_success = test_reset(test_email)
        
        if reset_success:
            print(f"\n🎉 === TESTE COMPLETO REALIZADO COM SUCESSO! ===")
            print(f"✅ Usuário: OK")
            print(f"✅ Reset de senha: OK")
            print(f"\n📬 Verifique o e-mail: {test_email}")
            print(f"📬 Procure também na pasta de SPAM!")
        else:
            print(f"\n❌ Falha no teste de reset de senha")
    else:
        print(f"\n❌ Falha ao criar usuário para teste")

def create_user(email, password):
    """Cria usuário de teste"""
    
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={firebase_config['apiKey']}"
    
    payload = {
        "email": email,
        "password": password,
        "returnSecureToken": True
    }
    
    try:
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Usuário criado com sucesso!")
            print(f"🆔 ID: {data.get('localId', 'N/A')}")
            return True
            
        else:
            error_data = response.json()
            error_message = error_data.get("error", {}).get("message", "")
            
            if "EMAIL_EXISTS" in error_message:
                print(f"✅ Usuário já existe (isso é bom para o teste!)")
                return "already exists"
            else:
                print(f"❌ Erro ao criar: {error_message}")
                return False
                
    except Exception as e:
        print(f"🚨 Erro: {e}")
        return False

def test_reset(email):
    """Testa reset de senha"""
    
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:sendOobCode?key={firebase_config['apiKey']}"
    
    payload = {
        "requestType": "PASSWORD_RESET",
        "email": email
    }
    
    try:
        response = requests.post(url, json=payload)
        
        print(f"📊 Status: {response.status_code}")
        print(f"📄 Response: {response.text}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ E-mail de reset enviado!")
            print(f"📧 Para: {data.get('email', 'N/A')}")
            print(f"🔗 Tipo: {data.get('kind', 'N/A')}")
            return True
            
        else:
            error_data = response.json()
            error_message = error_data.get("error", {}).get("message", "")
            print(f"❌ Erro no reset: {error_message}")
            return False
            
    except Exception as e:
        print(f"🚨 Erro: {e}")
        return False

if __name__ == "__main__":
    auto_test()
