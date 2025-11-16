
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


CLASSES = ['A','B','C','D','E','F','G','I','L','M','N','O','P','Q','R','S','T','U','V','W']


COLORS = [
    (255, 87, 51),
    (0, 123, 255),
    (255, 193, 7),
    (220, 53, 69),
    (25, 135, 84),
    (111, 66, 193),
    (255, 105, 180),
    (32, 201, 151),
    (255, 69, 0),
    (138, 43, 226),
    (255, 20, 147),
    (0, 191, 255),
    (50, 205, 50),
    (255, 140, 0),
    (199, 21, 133),
    (0, 206, 209),
    (148, 0, 211),
    (255, 215, 0),
    (70, 130, 180),
    (34, 139, 34)
]


COLORS_KIVY = [(b/255.0, g/255.0, r/255.0, 1.0) for r, g, b in COLORS]

def get_contrast_color(bg_color):
    
    if len(bg_color) >= 3:

        r, g, b = bg_color[2], bg_color[1], bg_color[0]
        luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
        return (255, 255, 255) if luminance < 0.5 else (0, 0, 0)
    return (255, 255, 255)

def get_contrast_color_kivy(bg_color):
    
    if len(bg_color) >= 3:

        r, g, b = bg_color[0] * 255, bg_color[1] * 255, bg_color[2] * 255
        luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
        return (1, 1, 1, 1) if luminance < 0.5 else (0, 0, 0, 1)
    return (1, 1, 1, 1)


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)


    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])


    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    return im, r, (dw, dh)


def load_tflite_model(model_path):
    
    try:
        interpreter = tf.lite.Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        print(f"Erro ao carregar modelo: {e}")
        return None


def preprocess_frame(frame, input_size=(640, 640)):
    

    img, ratio, pad = letterbox(frame, input_size)
    

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    

    img = np.expand_dims(img, axis=0)
    
    return img, ratio, pad


def non_max_suppression(boxes, scores, classes, conf_threshold=0.65, iou_threshold=0.75, max_det=3):
    

    valid_detections = scores >= conf_threshold
    boxes = boxes[valid_detections]
    scores = scores[valid_detections]
    classes = classes[valid_detections]
    
    if len(boxes) == 0:
        return [], [], []
    

    indices = np.argsort(scores)[::-1]
    
    keep = []
    while len(indices) > 0 and len(keep) < max_det:

        current = indices[0]
        keep.append(current)
        
        if len(indices) == 1:
            break
            

        current_box = boxes[current]
        other_boxes = boxes[indices[1:]]
        

        x1 = np.maximum(current_box[0], other_boxes[:, 0])
        y1 = np.maximum(current_box[1], other_boxes[:, 1])
        x2 = np.minimum(current_box[2], other_boxes[:, 2])
        y2 = np.minimum(current_box[3], other_boxes[:, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        

        area_current = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
        area_others = (other_boxes[:, 2] - other_boxes[:, 0]) * (other_boxes[:, 3] - other_boxes[:, 1])
        

        union = area_current + area_others - intersection
        iou = intersection / union
        

        indices = indices[1:][iou <= iou_threshold]
    
    return boxes[keep], scores[keep], classes[keep]


def detect_frame(interpreter, frame, conf_threshold=0.65, iou_threshold=0.75, max_det=3):
    

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    

    input_size = tuple(input_details[0]['shape'][1:3])
    processed_frame, ratio, pad = preprocess_frame(frame, input_size)
    

    interpreter.set_tensor(input_details[0]['index'], processed_frame)
    interpreter.invoke()
    

    output = interpreter.get_tensor(output_details[0]['index'])[0]
    

    boxes = []
    scores = []
    classes = []
    

    if len(output.shape) == 2:

        boxes_raw = output[:4].T
        scores_raw = output[4:].T
        

        max_scores = np.max(scores_raw, axis=1)
        best_classes = np.argmax(scores_raw, axis=1)
        
        valid_mask = max_scores >= conf_threshold
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) > 0:

            sorted_indices = valid_indices[np.argsort(max_scores[valid_indices])[::-1]]
            
            for idx in sorted_indices[:max_det]:
                x, y, w, h = boxes_raw[idx]
                score = max_scores[idx]
                class_id = best_classes[idx]
                


                x_center = x * frame.shape[1]
                y_center = y * frame.shape[0] 
                width = w * frame.shape[1]
                height = h * frame.shape[0]
                
                x1 = int(x_center - width / 2)
                y1 = int(y_center - height / 2)
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)
                

                x1 = max(0, min(x1, frame.shape[1] - 1))
                y1 = max(0, min(y1, frame.shape[0] - 1))
                x2 = max(0, min(x2, frame.shape[1] - 1))
                y2 = max(0, min(y2, frame.shape[0] - 1))
                

                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    scores.append(score)
                    classes.append(class_id)
    
    if len(boxes) > 0:
        boxes = np.array(boxes)
        scores = np.array(scores)
        classes = np.array(classes)
        

        boxes, scores, classes = non_max_suppression(boxes, scores, classes, conf_threshold, iou_threshold, max_det)
    
    return boxes, scores, classes


def draw_detections(frame, boxes, scores, classes, line_thickness=3, hide_labels=False, hide_conf=False):
    
    for box, score, class_id in zip(boxes, scores, classes):
        x1, y1, x2, y2 = box.astype(int)
        

        if class_id < len(COLORS):
            color = COLORS[class_id]
        else:
            color = (128, 128, 128)
        

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, line_thickness)
        

        if not hide_labels:
            if class_id < len(CLASSES):
                if hide_conf:
                    label = f"{CLASSES[class_id]}"
                else:
                    label = f"{CLASSES[class_id]} {score:.0%}"
            else:
                label = f"Class {class_id} {score:.0%}"
            

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = max(line_thickness - 1, 1)
            
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            

            padding = 4
            bg_x1 = x1
            bg_y1 = y1 - text_height - baseline - padding
            bg_x2 = x1 + text_width + padding * 2
            bg_y2 = y1
            

            cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
            

            text_color = get_contrast_color(color)
            

            cv2.putText(frame, label, (x1 + padding, y1 - baseline - padding//2), 
                       font, font_scale, text_color, thickness)
    
    return frame


def run_clean_detection(model_path, model_name="TensorFlow Lite", duration_seconds=30):
    
    print(f"\n Iniciando detecção limpa com {model_name}")
    print(f"Modelo: {model_path}")
    print("Configurações YOLOv5: conf=0.65, iou=0.75, max_det=3, line=3")
    print("Pressione 'q' para sair ou ESC")
    

    interpreter = load_tflite_model(model_path)
    if interpreter is None:
        return False
    
    print(" Modelo carregado com sucesso")
    

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(" Erro: Não foi possível abrir a câmera")
        return False
    
    print(" Câmera aberta com sucesso")
    

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print(" Erro ao capturar frame da câmera")
                break
            

            boxes, scores, classes = detect_frame(
                interpreter, frame, 
                conf_threshold=0.65,
                iou_threshold=0.75,
                max_det=3
            )
            

            frame = draw_detections(
                frame, boxes, scores, classes,
                line_thickness=3,
                hide_labels=False,
                hide_conf=False
            )
            

            cv2.imshow('0', frame)
            

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print(" Saindo por solicitação do usuário")
                break
            

            if time.time() - start_time > duration_seconds:
                print(f" Tempo limite de {duration_seconds}s atingido")
                break
                
    except KeyboardInterrupt:
        print(" Interrompido pelo usuário")
    
    finally:

        cap.release()
        cv2.destroyAllWindows()
        print(f" Detecção finalizada!")
    
    return True


class BoxDetectionPreview(Preview):
    
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.detections = []
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.labels = CLASSES
        self.confidence_threshold = 0.25
        
    def set_model(self, interpreter, input_details, output_details):
        
        self.interpreter = interpreter
        self.input_details = input_details
        self.output_details = output_details
        
    def _run_inference(self, frame):
        
        try:

            boxes, scores, classes = detect_frame(self.interpreter, frame, 
                                                 conf_threshold=self.confidence_threshold)
            
            return boxes, scores, classes
            
        except Exception as e:
            print(f" Erro na inferência: {e}")
            return [], [], []
        
    def analyze_pixels_callback(self, pixels, image_size, image_pos, image_scale, mirror):
        import time
        if not hasattr(self, '_benchmark_start'):
            self._benchmark_start = time.time()
        print(f"[DEBUG] analyze_pixels_callback chamado. Tempo desde input: {time.time() - self._benchmark_start:.3f}s")
        
        if not self.interpreter:
            return
            
        try:

            frame = np.frombuffer(pixels, dtype=np.uint8)
            frame = frame.reshape(image_size[1], image_size[0], 4)
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            

            boxes, scores, classes = self._run_inference(bgr)
            

            detections_screen = []
            for box, score, class_id in zip(boxes, scores, classes):
                x1, y1, x2, y2 = box.astype(int)
                


                y1_kivy = image_size[1] - y2
                y2_kivy = image_size[1] - y1
                
                if mirror:
                    x1_mirror = image_size[0] - x2
                    x2_mirror = image_size[0] - x1
                    x1, x2 = x1_mirror, x2_mirror
                

                screen_x = int(x1 * image_scale + image_pos[0])
                screen_y = int(y1_kivy * image_scale + image_pos[1])
                screen_w = int((x2 - x1) * image_scale)
                screen_h = int((y2_kivy - y1_kivy) * image_scale)
                

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
            

            if detections_screen and hasattr(self.parent, 'ids') and 'detection_result' in self.parent.ids:
                best_detection = max(detections_screen, key=lambda x: x['score'])
                result_text = f"{best_detection['label']} ({best_detection['score']:.0%})"
                tempo_total = time.time() - self._benchmark_start
                print(f"[BENCHMARK] Alfabeto: input até desenhar na tela = {tempo_total:.3f} segundos | Resultado: {result_text}")
                self.parent.ids.detection_result.text = result_text
                from utils.benchmark_logger import log_benchmark
                log_benchmark('modelo_alfabeto_total', tempo_total, {'resultado': result_text})
                self._benchmark_start = time.time()
            elif hasattr(self.parent, 'ids') and 'detection_result' in self.parent.ids:
                print(f"[DEBUG] Nenhuma detecção. Tempo desde input: {time.time() - self._benchmark_start:.3f}s")
            elif hasattr(self.parent, 'ids') and 'detection_result' in self.parent.ids:
                self.parent.ids.detection_result.text = ""
            
        except Exception as e:
            print(f" Erro na análise: {e}")
            self.detections = []
    
    def canvas_instructions_callback(self, texture, tex_size, tex_pos):
        
        from kivy.graphics import Color, Line, Rectangle
        from kivy.core.text import Label as CoreLabel
        

        for det in self.detections:

            class_index = -1
            if det['label'] in CLASSES:
                class_index = CLASSES.index(det['label'])
            

            if class_index >= 0 and class_index < len(COLORS_KIVY):
                box_color = COLORS_KIVY[class_index]
            else:
                box_color = (0.5, 0.5, 0.5, 1.0)
            

            Color(*box_color)
            Line(rectangle=(det['x'], det['y'], det['w'], det['h']), width=4)
            

            text = f"{det['label']} {det['score']:.0%}"
            

            label = CoreLabel(text=text, font_size=18, bold=True)
            label.refresh()
            text_texture = label.texture
            
            if text_texture:

                padding = 8
                bg_width = max(120, text_texture.width + padding * 2)
                bg_height = text_texture.height + padding
                

                Color(*box_color)
                Rectangle(
                    pos=(det['x'], det['y'] + det['h']), 
                    size=(bg_width, bg_height)
                )
                

                text_color = get_contrast_color_kivy(box_color)
                Color(*text_color)
                

                Rectangle(
                    texture=text_texture,
                    pos=(det['x'] + padding, det['y'] + det['h'] + padding//2),
                    size=text_texture.size
                )


class DetectionScreen(BaseScreen):
    
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.preview = None
        self.interpreter = None
        
    def on_enter(self):
        
        print(" Entrando na tela de detecção")
        self._load_model()
        self._start_camera()
        

        Clock.schedule_once(lambda dt: self._ensure_button_visibility(), 2.0)
        
    def _load_model(self):
        
        try:
            current_dir = Path(__file__).parent.parent
            model_path = current_dir / "services" / "ml" / "teste_tensorflow" / "best_float16.tflite"
            
            print(f" Carregando modelo: {model_path}")
            self.interpreter = load_tflite_model(model_path) 
            
            if self.interpreter:
                print(" Modelo carregado com sucesso")
            else:
                print(" Erro ao carregar modelo")
                
        except Exception as e:
            print(f" Erro ao carregar modelo: {e}")
    
    def _start_camera(self):
        
        try:
            print(" Iniciando camera4kivy...")
            
            self.preview = BoxDetectionPreview()
            
            if self.interpreter:
                input_details = self.interpreter.get_input_details()
                output_details = self.interpreter.get_output_details()
                self.preview.set_model(self.interpreter, input_details, output_details)
            
            Clock.schedule_once(self._connect_camera, 0.1)
            
        except Exception as e:
            print(f" Erro ao iniciar câmera: {e}")
    
    def _connect_camera(self, dt):
        
        try:
            camera_display = self.ids.get('camera_display')
            if camera_display and self.preview:
                camera_display.add_widget(self.preview)

                self.preview.connect_camera(
                    camera_id="0", 
                    filepath_callback=None,
                    enable_analyze_pixels=True,
                    analyze_pixels_resolution=480,
                    mirror=True
                )
                print(" Camera4kivy iniciada")
                

                self._ensure_button_visibility()
            else:
                print(" Erro: camera_display não encontrado")
                
        except Exception as e:
            print(f" Erro ao conectar câmera: {e}")
            
    def _ensure_button_visibility(self):
        
        try:
            close_button = self.ids.get('close_button')
            if close_button:

                parent = close_button.parent
                if parent:
                    parent.remove_widget(close_button)
                    parent.add_widget(close_button)
                    print(" Botão X reposicionado para frente")
        except Exception as e:
            print(f" Erro ao reposicionar botão: {e}")
    
    def go_back(self):
        
        print(" Saindo da detecção...")
        

        try:
            if self.preview:
                self.preview.disconnect_camera()
                print(" Câmera desconectada")
        except Exception as e:
            print(f" Erro ao desconectar câmera: {e}")
        

        print(" Voltando para home...")
        self.manager.transition.direction = 'right'
        self.manager.current = 'home'