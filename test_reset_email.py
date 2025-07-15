#!/usr/bin/env python3
"""
Script de teste para envio de e-mail de reset de senha Firebase
"""

import requests
import json

# Configuração do seu Firebase
firebase_config = {
    "apiKey": "***REMOVED***"
}

def test_reset_email():
    """Testa o envio de e-mail de reset"""
    
    print("🧪 === TESTE DE ENVIO DE E-MAIL FIREBASE ===\n")
    
    # Use um e-mail que você tem acesso para testar
    test_email = input("📧 Digite um e-mail para teste: ").strip()
    
    if not test_email:
        print("❌ E-mail não pode estar vazio!")
        return False
    
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:sendOobCode?key={firebase_config['apiKey']}"
    
    payload = {
        "requestType": "PASSWORD_RESET",
        "email": test_email
    }
    
    print(f"🔄 Enviando e-mail de reset para: {test_email}")
    print(f"🌐 URL: {url}")
    print(f"📦 Payload: {json.dumps(payload, indent=2)}")
    print("\n" + "="*50)
    
    try:
        response = requests.post(url, json=payload)
        
        print(f"📊 Status Code: {response.status_code}")
        print(f"📄 Response Text: {response.text}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n✅ E-MAIL ENVIADO COM SUCESSO!")
            print(f"📧 E-mail destino: {data.get('email', 'N/A')}")
            print(f"🔗 Tipo: {data.get('kind', 'N/A')}")
            print(f"\n📬 IMPORTANTE:")
            print(f"   • Verifique sua caixa de entrada")
            print(f"   • Verifique também a pasta de SPAM/LIXO ELETRÔNICO")
            print(f"   • O e-mail vem de: noreply@sinalizai.firebaseapp.com")
            print(f"   • Assunto: 'Redefinir sua senha - SinalizAI'")
            return True
            
        else:
            error_data = response.json()
            error_message = error_data.get("error", {}).get("message", "Erro desconhecido")
            print(f"\n❌ ERRO AO ENVIAR E-MAIL")
            print(f"🚨 Código do erro: {error_message}")
            
            # Erros comuns e soluções
            if "EMAIL_NOT_FOUND" in error_message:
                print(f"\n💡 SOLUÇÃO:")
                print(f"   • Este e-mail não está cadastrado no Firebase")
                print(f"   • Primeiro cadastre o usuário ou use um e-mail já cadastrado")
                print(f"   • Execute: python3 create_test_user.py")
                
            elif "INVALID_EMAIL" in error_message:
                print(f"\n💡 SOLUÇÃO:")
                print(f"   • Formato de e-mail inválido!")
                print(f"   • Use um formato válido: exemplo@dominio.com")
                
            elif "TOO_MANY_ATTEMPTS_TRY_LATER" in error_message:
                print(f"\n💡 SOLUÇÃO:")
                print(f"   • Muitas tentativas em pouco tempo")
                print(f"   • Aguarde 15-30 minutos e tente novamente")
                
            elif "INVALID_KEY" in error_message:
                print(f"\n💡 SOLUÇÃO:")
                print(f"   • Chave de API inválida")
                print(f"   • Verifique se a API Key está correta")
                
            else:
                print(f"\n💡 Erro não identificado. Detalhes completos:")
                print(f"   {json.dumps(error_data, indent=2)}")
                
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"\n🚨 ERRO DE CONEXÃO: {e}")
        print(f"💡 Verifique sua conexão com a internet")
        return False
        
    except Exception as e:
        print(f"\n🚨 ERRO INESPERADO: {e}")
        return False

def test_with_multiple_emails():
    """Teste com múltiplos e-mails"""
    
    print("\n🔄 === TESTE COM MÚLTIPLOS E-MAILS ===")
    
    test_emails = []
    
    while True:
        email = input("\n📧 Digite um e-mail (ou 'sair' para terminar): ").strip()
        if email.lower() in ['sair', 'exit', 'quit', '']:
            break
        test_emails.append(email)
    
    if not test_emails:
        print("❌ Nenhum e-mail para testar!")
        return
    
    print(f"\n🧪 Testando {len(test_emails)} e-mail(s)...")
    
    successful = 0
    failed = 0
    
    for i, email in enumerate(test_emails, 1):
        print(f"\n--- Teste {i}/{len(test_emails)} ---")
        print(f"E-mail: {email}")
        
        if test_reset_with_email(email):
            successful += 1
        else:
            failed += 1
    
    print(f"\n📊 === RESUMO DOS TESTES ===")
    print(f"✅ Sucessos: {successful}")
    print(f"❌ Falhas: {failed}")
    print(f"📈 Taxa de sucesso: {(successful/(successful+failed)*100):.1f}%" if (successful+failed) > 0 else "0%")

def test_reset_with_email(email):
    """Teste unitário para um e-mail específico"""
    
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:sendOobCode?key={firebase_config['apiKey']}"
    payload = {
        "requestType": "PASSWORD_RESET", 
        "email": email
    }
    
    try:
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            print(f"✅ Sucesso para {email}")
            return True
        else:
            error_data = response.json()
            error_message = error_data.get("error", {}).get("message", "Erro desconhecido")
            print(f"❌ Falha para {email}: {error_message}")
            return False
            
    except Exception as e:
        print(f"🚨 Erro para {email}: {e}")
        return False

if __name__ == "__main__":
    print("🎯 Escolha o tipo de teste:")
    print("1 - Teste com um e-mail")
    print("2 - Teste com múltiplos e-mails")
    
    choice = input("\nEscolha (1 ou 2): ").strip()
    
    if choice == "1":
        test_reset_email()
    elif choice == "2":
        test_with_multiple_emails()
    else:
        print("❌ Opção inválida!")
        test_reset_email()  # Default para teste simples
