

import os
import re
from pathlib import Path


EXTS = [".py", ".kv", ".js"]

SKIP_DIRS = {".git", "__pycache__", "node_modules", "venv", "env", "build"}


EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F900-\U0001F9FF"
    "\U00002600-\U000026FF"
    "]+",
    flags=re.UNICODE,
)

def remove_emojis(text: str) -> str:
    return EMOJI_PATTERN.sub("", text)

def remove_comments(text: str, ext: str) -> str:
    
    lines = text.splitlines(keepends=True)
    result_lines = []
    
    for line in lines:

        if ext == ".py" or ext == ".kv":

            if "#" in line:

                in_string = False
                quote_char = None
                for i, char in enumerate(line):
                    if char in ['"', "'"] and (i == 0 or line[i-1] != "\\"):
                        if not in_string:
                            in_string = True
                            quote_char = char
                        elif char == quote_char:
                            in_string = False
                            quote_char = None
                    elif char == "#" and not in_string:
                        line = line[:i].rstrip() + ("\n" if line.endswith("\n") else "")
                        break
        elif ext == ".js":

            if "//" in line:

                in_string = False
                quote_char = None
                for i, char in enumerate(line):
                    if char in ['"', "'"] and (i == 0 or line[i-1] != "\\"):
                        if not in_string:
                            in_string = True
                            quote_char = char
                        elif char == quote_char:
                            in_string = False
                            quote_char = None
                    elif i < len(line) - 1 and line[i:i+2] == "//" and not in_string:

                        if i > 0 and line[i-1] == ":":
                            continue
                        line = line[:i].rstrip() + ("\n" if line.endswith("\n") else "")
                        break
        
        result_lines.append(line)
    
    text = ''.join(result_lines)
    

    if ext == ".py":

        text = re.sub(r'', '', text, flags=re.DOTALL)
        text = re.sub(r"", '', text, flags=re.DOTALL)
    elif ext == ".js":

        text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    
    return text

def clean_file(file_path):
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        original_content = content
        ext = Path(file_path).suffix.lower()
        

        content = remove_comments(content, ext)
        

        content = remove_emojis(content)
        

        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Limpo: {file_path}")
            return True
        return False
    except Exception as e:
        print(f"Erro ao processar {file_path}: {e}")
        return False

def main():
    
    current_dir = Path(".")
    modified_count = 0
    
    print("Iniciando limpeza de comentários e emojis...")
    
    for root, dirs, files in os.walk(current_dir):

        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        
        for file in files:
            file_path = Path(root) / file
            if file_path.suffix.lower() in EXTS:
                if clean_file(file_path):
                    modified_count += 1
    
    print(f"\nConcluído! {modified_count} arquivos modificados.")

if __name__ == '__main__':
    main()