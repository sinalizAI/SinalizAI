#!/usr/bin/env python3
"""
Script para criar usuários de teste no Firebase
"""

import requests
import json

# Configuração do seu Firebase
firebase_config = {
    "apiKey": "***REMOVED***"
}

def create_test_user():
    """Cria um usuário de teste"""
    
    print("👤 === CRIAR USUÁRIO DE TESTE ===\n")
    
    email = input("📧 Digite um e-mail para criar usuário de teste: ").strip()
    if not email:
        print("❌ E-mail não pode estar vazio!")
        return False
    
    # Senha temporária forte
    password = "TesteSenha123!"
    
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={firebase_config['apiKey']}"
    
    payload = {
        "email": email,
        "password": password,
        "returnSecureToken": True
    }
    
    print(f"🔄 Criando usuário: {email}")
    print(f"🔑 Senha temporária: {password}")
    print(f"🌐 URL: {url}")
    
    try:
        response = requests.post(url, json=payload)
        
        print(f"\n📊 Status Code: {response.status_code}")
        print(f"📄 Response: {response.text}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n✅ USUÁRIO CRIADO COM SUCESSO!")
            print(f"📧 E-mail: {data.get('email')}")
            print(f"🆔 ID do usuário: {data.get('localId')}")
            print(f"🔑 Senha: {password}")
            print(f"🎫 Token: {data.get('idToken')[:20]}...")
            
            print(f"\n🧪 Agora você pode testar o reset de senha com:")
            print(f"   python3 test_reset_email.py")
            
            return True
            
        else:
            error_data = response.json()
            error_message = error_data.get("error", {}).get("message", "Erro desconhecido")
            print(f"\n❌ ERRO AO CRIAR USUÁRIO")
            print(f"🚨 Código do erro: {error_message}")
            
            if "EMAIL_EXISTS" in error_message:
                print(f"\n✅ ÓTIMO! Este e-mail já está cadastrado!")
                print(f"   Você pode usar este e-mail para testar o reset de senha")
                print(f"   Execute: python3 test_reset_email.py")
                return True
                
            elif "INVALID_EMAIL" in error_message:
                print(f"\n💡 SOLUÇÃO:")
                print(f"   • Formato de e-mail inválido!")
                print(f"   • Use um formato válido: exemplo@dominio.com")
                
            elif "WEAK_PASSWORD" in error_message:
                print(f"\n💡 SOLUÇÃO:")
                print(f"   • Senha muito fraca (mas isso não deveria acontecer)")
                print(f"   • A senha {password} é forte o suficiente")
                
            else:
                print(f"\n💡 Erro não identificado:")
                print(f"   {json.dumps(error_data, indent=2)}")
                
            return False
            
    except Exception as e:
        print(f"\n🚨 ERRO: {e}")
        return False

def create_multiple_users():
    """Cria múltiplos usuários de teste"""
    
    print("\n👥 === CRIAR MÚLTIPLOS USUÁRIOS ===")
    
    emails = []
    
    while True:
        email = input("\n📧 Digite um e-mail (ou 'sair' para terminar): ").strip()
        if email.lower() in ['sair', 'exit', 'quit', '']:
            break
        emails.append(email)
    
    if not emails:
        print("❌ Nenhum e-mail fornecido!")
        return
    
    print(f"\n🔄 Criando {len(emails)} usuário(s)...")
    
    successful = 0
    existing = 0
    failed = 0
    
    for i, email in enumerate(emails, 1):
        print(f"\n--- Usuário {i}/{len(emails)} ---")
        result = create_user_simple(email)
        
        if result == "success":
            successful += 1
        elif result == "exists":
            existing += 1
        else:
            failed += 1
    
    print(f"\n📊 === RESUMO ===")
    print(f"✅ Criados: {successful}")
    print(f"🔄 Já existiam: {existing}")
    print(f"❌ Falhas: {failed}")
    print(f"📈 Total disponíveis para teste: {successful + existing}")

def create_user_simple(email):
    """Cria um usuário simples, retorna status"""
    
    password = "TesteSenha123!"
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={firebase_config['apiKey']}"
    
    payload = {
        "email": email,
        "password": password,
        "returnSecureToken": True
    }
    
    try:
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            print(f"✅ Criado: {email}")
            return "success"
        else:
            error_data = response.json()
            error_message = error_data.get("error", {}).get("message", "")
            
            if "EMAIL_EXISTS" in error_message:
                print(f"🔄 Já existe: {email}")
                return "exists"
            else:
                print(f"❌ Erro em {email}: {error_message}")
                return "failed"
                
    except Exception as e:
        print(f"🚨 Erro em {email}: {e}")
        return "failed"

if __name__ == "__main__":
    print("🎯 Escolha o tipo de criação:")
    print("1 - Criar um usuário")
    print("2 - Criar múltiplos usuários")
    
    choice = input("\nEscolha (1 ou 2): ").strip()
    
    if choice == "1":
        create_test_user()
    elif choice == "2":
        create_multiple_users()
    else:
        print("❌ Opção inválida!")
        create_test_user()  # Default
