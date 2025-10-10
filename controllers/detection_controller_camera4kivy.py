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

# Cores vibrantes para cada classe (formato BGR para OpenCV)
COLORS = [
    (255, 87, 51),    # A - Laranja vibrante
    (0, 123, 255),    # B - Azul
    (255, 193, 7),    # C - Amarelo dourado
    (220, 53, 69),    # D - Vermelho
    (25, 135, 84),    # E - Verde escuro
    (111, 66, 193),   # F - Roxo
    (255, 105, 180),  # G - Rosa choque
    (32, 201, 151),   # I - Verde água
    (255, 69, 0),     # L - Laranja vermelho
    (138, 43, 226),   # M - Azul violeta
    (255, 20, 147),   # N - Rosa profundo
    (0, 191, 255),    # O - Azul céu
    (50, 205, 50),    # P - Verde lima
    (255, 140, 0),    # Q - Laranja escuro
    (199, 21, 133),   # R - Magenta escuro
    (0, 206, 209),    # S - Turquesa
    (148, 0, 211),    # T - Violeta escuro
    (255, 215, 0),    # U - Dourado
    (70, 130, 180),   # V - Azul aço
    (34, 139, 34)     # W - Verde floresta
]

# Cores normalizadas para Kivy (0-1)
COLORS_KIVY = [(b/255.0, g/255.0, r/255.0, 1.0) for r, g, b in COLORS]

def get_contrast_color(bg_color):
    """Retorna cor de texto contrastante (branco ou preto) baseada na cor de fundo"""
    if len(bg_color) >= 3:
        # Calcula luminância usando fórmula padrão
        r, g, b = bg_color[2], bg_color[1], bg_color[0]  # BGR para RGB
        luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
        return (255, 255, 255) if luminance < 0.5 else (0, 0, 0)  # Branco ou preto
    return (255, 255, 255)  # Branco como padrão

def get_contrast_color_kivy(bg_color):
    """Retorna cor de texto contrastante para Kivy (0-1)"""
    if len(bg_color) >= 3:
        # Converte de Kivy (0-1) para RGB (0-255) temporariamente
        r, g, b = bg_color[0] * 255, bg_color[1] * 255, bg_color[2] * 255
        luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
        return (1, 1, 1, 1) if luminance < 0.5 else (0, 0, 0, 1)  # Branco ou preto
    return (1, 1, 1, 1)  # Branco como padrão


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
    """Desenha as detecções com cores diferenciadas e texto contrastante"""
    for box, score, class_id in zip(boxes, scores, classes):
        x1, y1, x2, y2 = box.astype(int)
        
        # Cor da classe
        if class_id < len(COLORS):
            color = COLORS[class_id]
        else:
            color = (128, 128, 128)  # Cinza para classes desconhecidas
        
        # Desenhar retângulo com linha mais espessa para destaque
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, line_thickness)
        
        # Label
        if not hide_labels:
            if class_id < len(CLASSES):
                if hide_conf:
                    label = f"{CLASSES[class_id]}"
                else:
                    label = f"{CLASSES[class_id]} {score:.0%}"  # Percentual em vez de decimal
            else:
                label = f"Class {class_id} {score:.0%}"
            
            # Calcular tamanho do texto
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7  # Texto um pouco maior
            thickness = max(line_thickness - 1, 1)
            
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Adicionar padding ao fundo do texto
            padding = 4
            bg_x1 = x1
            bg_y1 = y1 - text_height - baseline - padding
            bg_x2 = x1 + text_width + padding * 2
            bg_y2 = y1
            
            # Fundo do texto com a mesma cor da caixa
            cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
            
            # Cor do texto contrastante
            text_color = get_contrast_color(color)
            
            # Texto com cor contrastante
            cv2.putText(frame, label, (x1 + padding, y1 - baseline - padding//2), 
                       font, font_scale, text_color, thickness)
    
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
        """Desenha bounding boxes e labels com cores diferenciadas sobre o preview da câmera"""
        from kivy.graphics import Color, Line, Rectangle
        from kivy.core.text import Label as CoreLabel
        
        # Desenhar bounding boxes e labels
        for det in self.detections:
            # Obter índice da classe para a cor
            class_index = -1
            if det['label'] in CLASSES:
                class_index = CLASSES.index(det['label'])
            
            # Cor da bounding box
            if class_index >= 0 and class_index < len(COLORS_KIVY):
                box_color = COLORS_KIVY[class_index]
            else:
                box_color = (0.5, 0.5, 0.5, 1.0)  # Cinza para classes desconhecidas
            
            # Desenhar bounding box
            Color(*box_color)
            Line(rectangle=(det['x'], det['y'], det['w'], det['h']), width=4)
            
            # Texto com letra e confiança
            text = f"{det['label']} {det['score']:.0%}"
            
            # Criar label de texto
            label = CoreLabel(text=text, font_size=18, bold=True)
            label.refresh()
            text_texture = label.texture
            
            if text_texture:
                # Calcular tamanho do fundo do texto
                padding = 8
                bg_width = max(120, text_texture.width + padding * 2)
                bg_height = text_texture.height + padding
                
                # Background do texto com a cor da classe
                Color(*box_color)
                Rectangle(
                    pos=(det['x'], det['y'] + det['h']), 
                    size=(bg_width, bg_height)
                )
                
                # Cor do texto contrastante
                text_color = get_contrast_color_kivy(box_color)
                Color(*text_color)
                
                # Texto com cor contrastante
                Rectangle(
                    texture=text_texture,
                    pos=(det['x'] + padding, det['y'] + det['h'] + padding//2),
                    size=text_texture.size
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
        
        # Garante que o botão X seja visível após carregar tudo
        Clock.schedule_once(lambda dt: self._ensure_button_visibility(), 2.0)
        
    def _load_model(self):
        """Carrega o modelo TensorFlow Lite"""
        try:
            current_dir = Path(__file__).parent.parent
            model_path = current_dir / "services" / "ml" / "teste_tensorflow" / "best_float16.tflite"
            
            print(f"📁 Carregando modelo: {model_path}")
            self.interpreter = load_tflite_model(model_path) 
            
            if self.interpreter:
                print("✅ Modelo carregado com sucesso")
            else:
                print("❌ Erro ao carregar modelo")
                
        except Exception as e:
            print(f"❌ Erro ao carregar modelo: {e}")
    
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
                
                # FORÇA o botão X para ficar sempre visível após conectar a câmera
                self._ensure_button_visibility()
            else:
                print("❌ Erro: camera_display não encontrado")
                
        except Exception as e:
            print(f"❌ Erro ao conectar câmera: {e}")
            
    def _ensure_button_visibility(self):
        """Garante que o botão X sempre fique visível por cima da câmera"""
        try:
            close_button = self.ids.get('close_button')
            if close_button:
                # Remove e re-adiciona o botão para colocá-lo no topo da pilha de widgets
                parent = close_button.parent
                if parent:
                    parent.remove_widget(close_button)
                    parent.add_widget(close_button)
                    print("🔴 Botão X reposicionado para frente")
        except Exception as e:
            print(f"⚠️ Erro ao reposicionar botão: {e}")
    
    def go_back(self):
        """Volta para a tela de home após limpar recursos da câmera"""
        print("📹 Saindo da detecção...")
        
        # Parar câmera e limpar recursos
        try:
            if self.preview:
                self.preview.disconnect_camera()
                print("📹 Câmera desconectada")
        except Exception as e:
            print(f"⚠️ Erro ao desconectar câmera: {e}")
        
        # Volta para home diretamente (não usa BaseScreen fallback)
        print("🏠 Voltando para home...")
        self.manager.transition.direction = 'right'
        self.manager.current = 'home'