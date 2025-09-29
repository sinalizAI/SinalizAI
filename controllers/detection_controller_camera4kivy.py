"""
Controller para tela de detecção LIBRAS usando camera4kivy
"""
from camera4kivy import Preview
from utils.base_screen import BaseScreen
from kivy.graphics import Color, Line, Rectangle
from kivy.clock import Clock
import numpy as np
import cv2
import os
import tensorflow as tf
from pathlib import Path
import time

# Classes do alfabeto em LIBRAS
CLASSES = ['A','B','C','D','E','F','G','I','L','M','N','O','P','Q','R','S','T','U','V','W']

# Cores para cada classe
COLORS = [(0, 255, 0)] * len(CLASSES)  # Verde para todas


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    """Redimensiona e faz padding da imagem (igual YOLOv5)"""
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    return im, r, (dw, dh)


def load_tflite_model(model_path):
    """Carrega modelo TensorFlow Lite"""
    try:
        interpreter = tf.lite.Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        print(f"Erro ao carregar modelo: {e}")
        return None


def preprocess_frame(frame, input_size=(640, 640)):
    """Preprocessa frame da câmera (igual YOLOv5)"""
    # Letterbox resize
    img, ratio, pad = letterbox(frame, input_size)
    
    # Converter para RGB e normalizar
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    
    # Adicionar dimensão do batch
    img = np.expand_dims(img, axis=0)
    
    return img, ratio, pad


def non_max_suppression(boxes, scores, classes, conf_threshold=0.65, iou_threshold=0.75, max_det=3):
    """NMS igual ao YOLOv5 original"""
    # Filtrar por confiança
    valid_detections = scores >= conf_threshold
    boxes = boxes[valid_detections]
    scores = scores[valid_detections]
    classes = classes[valid_detections]
    
    if len(boxes) == 0:
        return [], [], []
    
    # Ordenar por score (maior primeiro)
    indices = np.argsort(scores)[::-1]
    
    keep = []
    while len(indices) > 0 and len(keep) < max_det:
        # Pegar o de maior score
        current = indices[0]
        keep.append(current)
        
        if len(indices) == 1:
            break
            
        # Calcular IoU com os demais
        current_box = boxes[current]
        other_boxes = boxes[indices[1:]]
        
        # Calcular interseção
        x1 = np.maximum(current_box[0], other_boxes[:, 0])
        y1 = np.maximum(current_box[1], other_boxes[:, 1])
        x2 = np.minimum(current_box[2], other_boxes[:, 2])
        y2 = np.minimum(current_box[3], other_boxes[:, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        # Calcular áreas
        area_current = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
        area_others = (other_boxes[:, 2] - other_boxes[:, 0]) * (other_boxes[:, 3] - other_boxes[:, 1])
        
        # IoU
        union = area_current + area_others - intersection
        iou = intersection / union
        
        # Manter apenas os com IoU baixo
        indices = indices[1:][iou <= iou_threshold]
    
    return boxes[keep], scores[keep], classes[keep]


def detect_frame(interpreter, frame, conf_threshold=0.65, iou_threshold=0.75, max_det=3):
    """Detecta sinais em um frame (configurações YOLOv5)"""
    # Obter detalhes do modelo
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Preprocessar
    input_size = tuple(input_details[0]['shape'][1:3])
    processed_frame, ratio, pad = preprocess_frame(frame, input_size)
    
    # Inferência
    interpreter.set_tensor(input_details[0]['index'], processed_frame)
    interpreter.invoke()
    
    # Obter resultado
    output = interpreter.get_tensor(output_details[0]['index'])[0]  # Remove batch dimension
    
    # Parse das detecções (formato YOLO: x, y, w, h, conf, class_probs...)
    boxes = []
    scores = []
    classes = []
    
    # Usar o mesmo método do debug que funciona
    if len(output.shape) == 2:
        # Extrair coordenadas e scores
        boxes_raw = output[:4].T  # (num_detections, 4)
        scores_raw = output[4:].T  # (num_detections, num_classes)
        
        # Verificar detecções
        max_scores = np.max(scores_raw, axis=1)
        best_classes = np.argmax(scores_raw, axis=1)
        
        valid_mask = max_scores >= conf_threshold
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) > 0:
            # Ordenar por confiança
            sorted_indices = valid_indices[np.argsort(max_scores[valid_indices])[::-1]]
            
            for idx in sorted_indices[:max_det]:  # Limitar detecções
                x, y, w, h = boxes_raw[idx]
                score = max_scores[idx]
                class_id = best_classes[idx]
                
                # Converter coordenadas para frame original
                # YOLO format: center_x, center_y, width, height (normalized 0-1)
                x_center = x * frame.shape[1]
                y_center = y * frame.shape[0] 
                width = w * frame.shape[1]
                height = h * frame.shape[0]
                
                x1 = int(x_center - width / 2)
                y1 = int(y_center - height / 2)
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)
                
                # Clamp para limites da imagem
                x1 = max(0, min(x1, frame.shape[1] - 1))
                y1 = max(0, min(y1, frame.shape[0] - 1))
                x2 = max(0, min(x2, frame.shape[1] - 1))
                y2 = max(0, min(y2, frame.shape[0] - 1))
                
                # Verificar se a box é válida
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    scores.append(score)
                    classes.append(class_id)
    
    if len(boxes) > 0:
        boxes = np.array(boxes)
        scores = np.array(scores)
        classes = np.array(classes)
        
        # Aplicar NMS
        boxes, scores, classes = non_max_suppression(boxes, scores, classes, conf_threshold, iou_threshold, max_det)
    
    return boxes, scores, classes


def draw_detections(frame, boxes, scores, classes, line_thickness=3, hide_labels=False, hide_conf=False):
    """Desenha as detecções exatamente como o YOLOv5"""
    for box, score, class_id in zip(boxes, scores, classes):
        x1, y1, x2, y2 = box.astype(int)
        
        # Cor da classe
        if class_id < len(COLORS):
            color = COLORS[class_id]
        else:
            color = (255, 255, 255)
        
        # Desenhar retângulo
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, line_thickness)
        
        # Label
        if not hide_labels:
            if class_id < len(CLASSES):
                if hide_conf:
                    label = f"{CLASSES[class_id]}"
                else:
                    label = f"{CLASSES[class_id]} {score:.2f}"
            else:
                label = f"Class {class_id} {score:.2f}"
            
            # Calcular tamanho do texto
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = max(line_thickness - 1, 1)
            
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Fundo do texto
            cv2.rectangle(frame, (x1, y1 - text_height - baseline - 2), 
                         (x1 + text_width, y1), color, -1)
            
            # Texto
            cv2.putText(frame, label, (x1, y1 - baseline - 2), 
                       font, font_scale, (0, 0, 0), thickness)
    
    return frame


def run_clean_detection(model_path, model_name="TensorFlow Lite", duration_seconds=30):
    """Executa detecção limpa igual ao YOLOv5 original"""
    print(f"\n🚀 Iniciando detecção limpa com {model_name}")
    print(f"Modelo: {model_path}")
    print("Configurações YOLOv5: conf=0.65, iou=0.75, max_det=3, line=3")
    print("Pressione 'q' para sair ou ESC")
    
    # Carregar modelo
    interpreter = load_tflite_model(model_path)
    if interpreter is None:
        return False
    
    print("✅ Modelo carregado com sucesso")
    
    # Abrir câmera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Erro: Não foi possível abrir a câmera")
        return False
    
    print("📹 Câmera aberta com sucesso")
    
    # Configurar câmera (mesmas configurações do YOLOv5)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Erro ao capturar frame da câmera")
                break
            
            # Detectar (configurações YOLOv5)
            boxes, scores, classes = detect_frame(
                interpreter, frame, 
                conf_threshold=0.65,    # Mesmo do YOLOv5
                iou_threshold=0.75,     # Mesmo do YOLOv5
                max_det=3               # Mesmo do YOLOv5
            )
            
            # Desenhar detecções (mesmo estilo YOLOv5)
            frame = draw_detections(
                frame, boxes, scores, classes,
                line_thickness=3,       # Mesmo do YOLOv5
                hide_labels=False,      # Mesmo do YOLOv5
                hide_conf=False         # Mesmo do YOLOv5
            )
            
            # Mostrar frame (nome igual ao YOLOv5)
            cv2.imshow('0', frame)  # YOLOv5 usa '0' para webcam
            
            # Verificar teclas
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' ou ESC
                print("👋 Saindo por solicitação do usuário")
                break
            
            # Verificar tempo limite
            if time.time() - start_time > duration_seconds:
                print(f"⏰ Tempo limite de {duration_seconds}s atingido")
                break
                
    except KeyboardInterrupt:
        print("👋 Interrompido pelo usuário")
    
    finally:
        # Limpar recursos
        cap.release()
        cv2.destroyAllWindows()
        print(f"✅ Detecção finalizada!")
    
    return True


class BoxDetectionPreview(Preview):
    """Preview personalizado que implementa detecção com bounding boxes"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.detections = []
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.labels = CLASSES
        self.confidence_threshold = 0.25
        
    def set_model(self, interpreter, input_details, output_details):
        """Define o modelo TensorFlow Lite"""
        self.interpreter = interpreter
        self.input_details = input_details
        self.output_details = output_details
        
    def _run_inference(self, frame):
        """Executa inferência no frame e retorna detecções"""
        try:
            # Usar função existente que faz tudo
            boxes, scores, classes = detect_frame(self.interpreter, frame, 
                                                 conf_threshold=self.confidence_threshold)
            
            return boxes, scores, classes
            
        except Exception as e:
            print(f"❌ Erro na inferência: {e}")
            return [], [], []
        
    def analyze_pixels_callback(self, pixels, image_size, image_pos, image_scale, mirror):
        """Processa frame da câmera e detecta gestos LIBRAS"""
        if not self.interpreter:
            return
            
        try:
            # Converte pixels RGBA para BGR OpenCV
            frame = np.frombuffer(pixels, dtype=np.uint8)
            frame = frame.reshape(image_size[1], image_size[0], 4)
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            
            # Executa detecção YOLO
            boxes, scores, classes = self._run_inference(bgr)
            
            # Converte para coordenadas da tela
            detections_screen = []
            for box, score, class_id in zip(boxes, scores, classes):
                x1, y1, x2, y2 = box.astype(int)
                
                # Converte coordenadas OpenCV -> Kivy
                # Flip Y coordinate (OpenCV origem topo-esquerda, Kivy origem baixo-esquerda)
                y1_kivy = image_size[1] - y2
                y2_kivy = image_size[1] - y1
                
                if mirror:
                    x1_mirror = image_size[0] - x2
                    x2_mirror = image_size[0] - x1
                    x1, x2 = x1_mirror, x2_mirror
                
                # Aplica escala e posição da tela
                screen_x = int(x1 * image_scale + image_pos[0])
                screen_y = int(y1_kivy * image_scale + image_pos[1])
                screen_w = int((x2 - x1) * image_scale)
                screen_h = int((y2_kivy - y1_kivy) * image_scale)
                
                # Obter label da classe
                label = CLASSES[class_id] if class_id < len(CLASSES) else f"Class{class_id}"
                
                detections_screen.append({
                    'x': screen_x,
                    'y': screen_y,
                    'w': screen_w,
                    'h': screen_h,
                    'label': label,
                    'score': score
                })
            
            self.detections = detections_screen
            
            # Atualizar label de resultado se houver detecções (via parent screen)
            if detections_screen and hasattr(self.parent, 'ids') and 'detection_result' in self.parent.ids:
                best_detection = max(detections_screen, key=lambda x: x['score'])
                result_text = f"{best_detection['label']} ({best_detection['score']:.0%})"
                self.parent.ids.detection_result.text = result_text
            elif hasattr(self.parent, 'ids') and 'detection_result' in self.parent.ids:
                self.parent.ids.detection_result.text = ""
            
        except Exception as e:
            print(f"❌ Erro na análise: {e}")
            self.detections = []
    
    def canvas_instructions_callback(self, texture, tex_size, tex_pos):
        """Desenha bounding boxes e labels sobre o preview da câmera"""
        from kivy.graphics import Color, Line, Rectangle
        from kivy.core.text import Label as CoreLabel
        
        # Desenhar bounding boxes e labels
        for det in self.detections:
            # Bounding box verde
            Color(0, 1, 0, 1)  # Verde
            Line(rectangle=(det['x'], det['y'], det['w'], det['h']), width=3)
            
            # Background do texto (semi-transparente)
            Color(0, 1, 0, 0.8)  # Verde semi-transparente
            text_height = 30
            Rectangle(pos=(det['x'], det['y'] + det['h']), size=(max(100, det['w']), text_height))
            
            # Texto com letra e confiança
            text = f"{det['label']} {det['score']:.0%}"
            
            # Criar label de texto
            label = CoreLabel(text=text, font_size=16)
            label.refresh()
            texture = label.texture
            
            if texture:
                Color(1, 1, 1, 1)  # Branco
                Rectangle(
                    texture=texture,
                    pos=(det['x'] + 5, det['y'] + det['h'] + 5),
                    size=texture.size
                )


class DetectionScreen(BaseScreen):
    """Tela de detecção LIBRAS usando camera4kivy"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.preview = None
        self.interpreter = None
        
    def on_enter(self):
        """Chamado quando entra na tela"""
        print("🎬 Entrando na tela de detecção")
        self._load_model()
        self._start_camera()
        
    def _load_model(self):
        """Carrega o modelo TensorFlow Lite"""
        try:
            current_dir = Path(__file__).parent.parent
            model_path = current_dir / "services" / "ml" / "teste_tensorflow" / "best_float16.tflite"
            
            print(f"📁 Carregando modelo: {model_path}")
            self.interpreter = load_tflite_model(model_path) 
            
            if self.interpreter:
                print("✅ Modelo carregado com sucesso")
                # Atualizar status na tela se existir
                if hasattr(self, 'ids') and 'status_label' in self.ids:
                    self.ids.status_label.text = "Modelo carregado - Câmera ativa"
            else:
                print("❌ Erro ao carregar modelo")
                if hasattr(self, 'ids') and 'status_label' in self.ids:
                    self.ids.status_label.text = "Erro ao carregar modelo"
                
        except Exception as e:
            print(f"❌ Erro ao carregar modelo: {e}")
            if hasattr(self, 'ids') and 'status_label' in self.ids:
                self.ids.status_label.text = f"Erro: {e}"
    
    def _start_camera(self):
        """Inicia a câmera com preview personalizado"""
        try:
            print("📷 Iniciando camera4kivy...")
            
            self.preview = BoxDetectionPreview()
            
            if self.interpreter:
                input_details = self.interpreter.get_input_details()
                output_details = self.interpreter.get_output_details()
                self.preview.set_model(self.interpreter, input_details, output_details)
            
            Clock.schedule_once(self._connect_camera, 0.1)
            
        except Exception as e:
            print(f"❌ Erro ao iniciar câmera: {e}")
    
    def _connect_camera(self, dt):
        """Conecta a câmera ao preview"""
        try:
            camera_display = self.ids.get('camera_display')
            if camera_display and self.preview:
                camera_display.add_widget(self.preview)
                # Conecta com análise de pixels habilitada
                self.preview.connect_camera(
                    camera_id="0", 
                    filepath_callback=None,
                    enable_analyze_pixels=True,
                    analyze_pixels_resolution=480,
                    mirror=True
                )
                print("✅ Camera4kivy iniciada")
                if hasattr(self, 'ids') and 'status_label' in self.ids:
                    self.ids.status_label.text = "Detecção ativa - Faça um sinal"
            else:
                print("❌ Erro: camera_display não encontrado")
                if hasattr(self, 'ids') and 'status_label' in self.ids:
                    self.ids.status_label.text = "Erro na câmera"
                
        except Exception as e:
            print(f"❌ Erro ao conectar câmera: {e}")
            if hasattr(self, 'ids') and 'status_label' in self.ids:
                self.ids.status_label.text = f"Erro câmera: {e}"
    
    def go_back(self):
        """Volta para a tela anterior usando BaseScreen"""
        print("📹 Saindo da detecção...")
        
        # Parar câmera e limpar recursos
        try:
            if self.preview:
                self.preview.disconnect_camera()
                print("📹 Câmera desconectada")
        except Exception as e:
            print(f"⚠️ Erro ao desconectar câmera: {e}")
        
        # Usa o método go_to_back do BaseScreen
        self.go_to_back()