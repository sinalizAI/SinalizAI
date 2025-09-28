# Teste TensorFlow Lite - SinalizAI

Esta pasta contém os arquivos necessários para testar os modelos TensorFlow Lite do SinalizAI.

## Arquivos

- `best_float16.tflite` - Modelo TensorFlow Lite com precisão float16 (menor tamanho)
- `best_float32.tflite` - Modelo TensorFlow Lite com precisão float32 (maior precisão)
- `detect_tensorflow.py` - Script principal adaptado para usar modelos TensorFlow Lite
- `test_models.py` - Script de teste interativo
- `README.md` - Este arquivo

## Dependências

Certifique-se de ter as seguintes dependências instaladas:

```bash
pip install tensorflow opencv-python numpy
```

## Como usar

### Teste Interativo

Execute o script de teste para uma interface amigável:

```bash
python test_models.py
```

### Teste Manual

Para testar o modelo float16 com webcam:

```bash
python detect_tensorflow.py --weights best_float16.tflite --source 0 --view-img --save-csv
```

Para testar o modelo float32 com webcam:

```bash
python detect_tensorflow.py --weights best_float32.tflite --source 0 --view-img --save-csv
```

### Parâmetros Disponíveis

- `--weights`: Caminho para o modelo .tflite
- `--source`: Fonte de entrada (0 para webcam, caminho para arquivo)
- `--imgsz`: Tamanho da imagem (padrão: 640)
- `--conf-thres`: Limiar de confiança (padrão: 0.45)
- `--iou-thres`: Limiar de IoU para NMS (padrão: 0.45)
- `--view-img`: Mostrar resultados em tempo real
- `--save-txt`: Salvar coordenadas em arquivos .txt
- `--save-csv`: Salvar resultados em CSV
- `--project`: Pasta para salvar resultados
- `--name`: Nome do experimento

## Resultados

Os resultados são salvos em:
- `runs/detect_tf/` - Imagens processadas
- `runs/detect_tf/labels/` - Coordenadas das detecções (formato YOLO)
- `runs/detect_tf/predictions.csv` - Resultados em CSV

## Diferenças entre Float16 e Float32

### Float16
- ✅ Menor tamanho do arquivo
- ✅ Mais rápido em dispositivos com suporte
- ✅ Menor uso de memória
- ❌ Potencial perda de precisão

### Float32
- ✅ Maior precisão
- ✅ Melhor estabilidade numérica
- ❌ Maior tamanho do arquivo
- ❌ Mais lento

## Para Build Android

O modelo float16 é recomendado para builds Android devido ao menor tamanho e melhor performance em dispositivos móveis.

## Controles Durante Execução

- Pressione `q` para sair
- A janela de visualização mostra as detecções em tempo real
- Os resultados são automaticamente salvos conforme configurado

## Suporte

Este código foi adaptado especificamente para os modelos de detecção de letras do alfabeto em Libras do projeto SinalizAI.