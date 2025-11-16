import json
import os
from datetime import datetime

def log_benchmark(acao, tempo_resposta, detalhes=None):
    





    path = os.path.join(os.path.dirname(__file__), '../benchmarks.json')
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        data = []
    entry = {
        'timestamp': datetime.now().isoformat(),
        'acao': acao,
        'tempo_resposta': tempo_resposta,
    }
    if detalhes:
        entry['detalhes'] = detalhes
    data.append(entry)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
