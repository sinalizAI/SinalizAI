#!/usr/bin/env python3
"""
Script específico para remover apenas emojis que restaram no código.
"""
import os
import re
from pathlib import Path

# Extensões a processar (incluindo arquivos de configuração)
EXTS = [".py", ".kv", ".js", ".yml", ".yaml", ".json", ".md", ".txt", ".ipynb", ".toml", ".sh", ".html", ".htm", ".dockerfile"]
# Diretórios a pular
SKIP_DIRS = {".git", "__pycache__", "node_modules", "venv", "env", "build"}

# Regex muito abrangente para capturar TODOS os emojis e símbolos Unicode
EMOJI_PATTERN = re.compile(
    "["
    # Blocos principais de emojis
    "\U0001F300-\U0001F5FF"  # símbolos & pictogramas
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transporte & mapas
    "\U0001F700-\U0001F77F"  # alquímicos
    "\U0001F780-\U0001F7FF"  # formas geométricas estendidas
    "\U0001F800-\U0001F8FF"  # setas suplementares
    "\U0001F900-\U0001F9FF"  # símbolos suplementares
    "\U0001FA00-\U0001FA6F"  # xadrez, etc
    "\U0001FA70-\U0001FAFF"  # símbolos e pictogramas estendidos
    # Símbolos diversos (onde está o )
    "\U00002000-\U0000206F"  # pontuação geral
    "\U00002070-\U0000209F"  # sobrescrito e subscrito
    "\U000020A0-\U000020CF"  # símbolos de moeda
    "\U000020D0-\U000020FF"  # marcas combinantes
    "\U00002100-\U0000214F"  # símbolos letterlike
    "\U00002150-\U0000218F"  # formas numéricas
    "\U00002190-\U000021FF"  # setas
    "\U00002200-\U000022FF"  # operadores matemáticos
    "\U00002300-\U000023FF"  # símbolos técnicos diversos (INCLUI  U+23F1)
    "\U00002400-\U000024FF"  # símbolos de controle
    "\U00002500-\U0000257F"  # elementos de desenho de caixa
    "\U00002580-\U0000259F"  # elementos de bloco
    "\U000025A0-\U000025FF"  # formas geométricas
    "\U00002600-\U000026FF"  # símbolos diversos
    "\U00002700-\U000027BF"  # dingbats
    "\U000027C0-\U000027EF"  # setas diversas suplementares
    "\U000027F0-\U000027FF"  # setas suplementares A
    "\U00002800-\U000028FF"  # padrões braille
    "\U00002900-\U0000297F"  # setas suplementares B
    "\U00002980-\U000029FF"  # operadores matemáticos diversos
    "\U00002A00-\U00002AFF"  # operadores matemáticos suplementares
    "\U00002B00-\U00002BFF"  # símbolos e setas diversos
    "\U0000FE0E-\U0000FE0F"  # seletores de variação
    "]",
    flags=re.UNICODE,
)

def remove_emojis_only(text: str) -> str:
    """Remove apenas emojis do texto, preservando tudo mais"""
    return EMOJI_PATTERN.sub("", text)

def clean_emojis_from_file(file_path):
    """Remove apenas emojis de um arquivo"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        original_content = content
        
        # Remove apenas emojis
        content = remove_emojis_only(content)
        
        # Salva apenas se houve mudança
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Emojis removidos de: {file_path}")
            return True
        return False
    except Exception as e:
        print(f"Erro ao processar {file_path}: {e}")
        return False

def main():
    """Remove emojis de todos os arquivos do projeto"""
    current_dir = Path(".")
    modified_count = 0
    
    print("Iniciando remoção específica de emojis...")
    
    for root, dirs, files in os.walk(current_dir):
        # Pula diretórios desnecessários
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        
        for file in files:
            file_path = Path(root) / file
            if file_path.suffix.lower() in EXTS or file_path.name.startswith("Dockerfile"):
                if clean_emojis_from_file(file_path):
                    modified_count += 1
    
    print(f"\nConcluído! {modified_count} arquivos tiveram emojis removidos.")

if __name__ == '__main__':
    main()