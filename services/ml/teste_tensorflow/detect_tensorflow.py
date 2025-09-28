# Detecção com TensorFlow Lite - Adaptado para SinalizAI# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Baseado no código original detect.py do YOLOv5

# python detect.py --weights alfabeto-final.pt --data ./data.yaml --source 0

import argparse 

import csvimport argparse

import osimport csv

import platformimport os

import sysimport platform

from pathlib import Pathimport sys

import numpy as npfrom pathlib import Path

import cv2

import tensorflow as tfimport torch



FILE = Path(__file__).resolve()FILE = Path(__file__).resolve()

ROOT = FILE.parents[0] ROOT = FILE.parents[0] 

if str(ROOT) not in sys.path:if str(ROOT) not in sys.path:

    sys.path.append(str(ROOT))    sys.path.append(str(ROOT))

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))ROOT = Path(os.path.relpath(ROOT, Path.cwd()))



# Classes do alfabeto (adaptado para o seu modelo)from ultralytics.utils.plotting import Annotator, colors, save_one_box

CLASSES = [

    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',from models.common import DetectMultiBackend

    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams

]from utils.general import (

    LOGGER,

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):    Profile,

    """Redimensiona e faz padding da imagem para o formato YOLO"""    check_file,

    shape = im.shape[:2]  # current shape [height, width]    check_img_size,

    if isinstance(new_shape, int):    check_imshow,

        new_shape = (new_shape, new_shape)    check_requirements,

    colorstr,

    # Scale ratio (new / old)    cv2,

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])    increment_path,

    if not scaleup:  # only scale down, do not scale up (for better val mAP)    non_max_suppression,

        r = min(r, 1.0)    print_args,

    scale_boxes,

    # Compute padding    strip_optimizer,

    ratio = r, r  # width, height ratios    xyxy2xywh,

    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r)))

    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh paddingfrom utils.torch_utils import select_device, smart_inference_mode



    if auto:  # minimum rectangle

        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding@smart_inference_mode()

    elif scaleFill:  # stretchdef run(

        dw, dh = 0.0, 0.0    weights=ROOT / "../alfabeto.pt",

        new_unpad = (new_shape[1], new_shape[0])    source=ROOT / "0",

        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios    data=ROOT / "data/data.yaml",

    imgsz=(640, 640),

    dw /= 2  # divide padding into 2 sides    conf_thres=0.45,

    dh /= 2    iou_thres=0.45,

    max_det=1000,

    if shape[::-1] != new_unpad:  # resize    device="0",

        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)    view_img=True,

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))    save_txt=True,

    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))    save_format=0,

    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border    save_csv=False,

    return im, ratio, (dw, dh)    save_conf=True,

    save_crop=False,

def non_max_suppression_tf(boxes, scores, classes, max_detections=100, score_threshold=0.45, iou_threshold=0.45):    nosave=False, 

    """Aplica Non-Maximum Suppression usando TensorFlow"""    classes=None, 

    # Converter para tensores do TensorFlow    agnostic_nms=False,

    boxes_tf = tf.constant(boxes, dtype=tf.float32)    augment=False, 

    scores_tf = tf.constant(scores, dtype=tf.float32)    visualize=False,

        update=False, 

    # Aplicar NMS    project=ROOT / "runs/detect", 

    selected_indices = tf.image.non_max_suppression(    name="alfa_", 

        boxes_tf, scores_tf, max_detections, iou_threshold, score_threshold    exist_ok=False, 

    )    line_thickness=3, 

        hide_labels=False, 

    # Obter resultados filtrados    hide_conf=False,

    selected_indices = selected_indices.numpy()    half=False,  

    filtered_boxes = boxes[selected_indices]    dnn=False,  

    filtered_scores = scores[selected_indices]    vid_stride=1,

    filtered_classes = classes[selected_indices]):

    

    return filtered_boxes, filtered_scores, filtered_classes, selected_indices    source = str(source)

    save_img = not nosave and not source.endswith(".txt") 

def xywh2xyxy(x):    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)

    """Converte nx4 boxes de [x, y, w, h] para [x1, y1, x2, y2]"""    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))

    y = np.copy(x)    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)

    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x    screenshot = source.lower().startswith("screen")

    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y    if is_url and is_file:

    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x        source = check_file(source) 

    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y

    return y    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  

    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  

def xyxy2xywh(x):

    """Converte nx4 boxes de [x1, y1, x2, y2] para [x, y, w, h]"""    device = select_device(device)

    y = np.copy(x)    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)

    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center    stride, names, pt = model.stride, model.names, model.pt

    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center    imgsz = check_img_size(imgsz, s=stride) 

    y[:, 2] = x[:, 2] - x[:, 0]  # width

    y[:, 3] = x[:, 3] - x[:, 1]  # height    bs = 1 

    return y    if webcam:

        view_img = check_imshow(warn=True)

def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)

    """Reescala boxes de img1_shape para img0_shape"""        bs = len(dataset)

    if ratio_pad is None:  # calculate from img0_shape    elif screenshot:

        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)

        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding    else:

    else:        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)

        gain = ratio_pad[0][0]    vid_path, vid_writer = [None] * bs, [None] * bs

        pad = ratio_pad[1]

    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))

    boxes[:, [0, 2]] -= pad[0]  # x padding    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))

    boxes[:, [1, 3]] -= pad[1]  # y padding    for path, im, im0s, vid_cap, s in dataset:

    boxes[:, :4] /= gain        with dt[0]:

    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, img0_shape[1])  # x1, x2            im = torch.from_numpy(im).to(model.device)

    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, img0_shape[0])  # y1, y2            im = im.half() if model.fp16 else im.float() 

    return boxes            im /= 255

            if len(im.shape) == 3:

def run_tensorflow(                im = im[None]

    weights=ROOT / "best_float16.tflite",            if model.xml and im.shape[0] > 1:

    source="0",                ims = torch.chunk(im, im.shape[0], 0)

    imgsz=640,

    conf_thres=0.45,        with dt[1]:

    iou_thres=0.45,            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False

    max_det=1000,            if model.xml and im.shape[0] > 1:

    view_img=True,                pred = None

    save_txt=True,                for image in ims:

    save_csv=False,                    if pred is None:

    save_conf=True,                        pred = model(image, augment=augment, visualize=visualize).unsqueeze(0)

    save_crop=False,                    else:

    nosave=False,                        pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)

    classes=None,                pred = [pred, None]

    project=ROOT / "runs/detect_tf",            else:

    name="exp",                pred = model(im, augment=augment, visualize=visualize)

    exist_ok=False,        with dt[2]:

    line_thickness=3,            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

    hide_labels=False,        

    hide_conf=False,        csv_path = save_dir / "predictions.csv"

):

    """        def write_to_csv(image_name, prediction, confidence):

    Executa detecção usando modelo TensorFlow Lite            data = {"Image Name": image_name, "Prediction": prediction, "Confidence": confidence}

    """            file_exists = os.path.isfile(csv_path)

                with open(csv_path, mode="a", newline="") as f:

    # Configurar caminhos                writer = csv.DictWriter(f, fieldnames=data.keys())

    source = str(source)                if not file_exists:

    save_img = not nosave and not source.endswith(".txt")                    writer.writeheader()

                    writer.writerow(data)

    # Criar diretório de salvamento

    save_dir = Path(project) / name        for i, det in enumerate(pred): 

    counter = 1            seen += 1

    while save_dir.exists() and not exist_ok:            if webcam:  

        save_dir = Path(project) / f"{name}_{counter}"                p, im0, frame = path[i], im0s[i].copy(), dataset.count

        counter += 1                s += f"{i}: "

                else:

    save_dir.mkdir(parents=True, exist_ok=True)                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

    (save_dir / "labels").mkdir(exist_ok=True)

                p = Path(p) 

    # Carregar modelo TensorFlow Lite            save_path = str(save_dir / p.name)  

    print(f"Carregando modelo TensorFlow Lite: {weights}")            txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")

    interpreter = tf.lite.Interpreter(model_path=str(weights))            s += "{:g}x{:g} ".format(*im.shape[2:])  

    interpreter.allocate_tensors()            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  

                imc = im0.copy() if save_crop else im0 

    # Obter detalhes do modelo            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

    input_details = interpreter.get_input_details()            if len(det):

    output_details = interpreter.get_output_details()                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

    

    print(f"Input shape: {input_details[0]['shape']}")                for c in det[:, 5].unique():

    print(f"Output shape: {output_details[0]['shape']}")                    n = (det[:, 5] == c).sum() 

                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, " 

    # Configurar fonte de vídeo

    webcam = source.isnumeric() or source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))                for *xyxy, conf, cls in reversed(det):

                        c = int(cls) 

    if webcam:                    label = names[c] if hide_conf else f"{names[c]}"

        cap = cv2.VideoCapture(int(source) if source.isnumeric() else source)                    confidence = float(conf)

        if not cap.isOpened():                    confidence_str = f"{confidence:.2f}"

            print(f"Erro: Não foi possível abrir a câmera {source}")

            return                    if save_csv:

    else:                        write_to_csv(p.name, label, confidence_str)

        # Para arquivos de imagem/vídeo

        if os.path.isfile(source):                    if save_txt:  

            if source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):                        if save_format == 0:

                cap = cv2.VideoCapture(source)                            coords = (

            else:                                (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()

                # Imagem única                            )  

                img0 = cv2.imread(source)                        else:

                cap = None                            coords = (torch.tensor(xyxy).view(1, 4) / gn).view(-1).tolist()

        else:                        line = (cls, *coords, conf) if save_conf else (cls, *coords)  

            print(f"Erro: Fonte não encontrada: {source}")                        with open(f"{txt_path}.txt", "a") as f:

            return                            f.write(("%g " * len(line)).rstrip() % line + "\n")

    

    # CSV para salvar resultados                    if save_img or save_crop or view_img:

    csv_path = save_dir / "predictions.csv"                        c = int(cls)  

                            label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")

    def write_to_csv(image_name, prediction, confidence):                        annotator.box_label(xyxy, label, color=colors(c, True))

        data = {"Image Name": image_name, "Prediction": prediction, "Confidence": confidence}                    if save_crop:

        file_exists = os.path.isfile(csv_path)                        save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True)

        with open(csv_path, mode="a", newline="", encoding='utf-8') as f:

            writer = csv.DictWriter(f, fieldnames=data.keys())            

            if not file_exists:            im0 = annotator.result()

                writer.writeheader()            if view_img:

            writer.writerow(data)                if platform.system() == "Linux" and p not in windows:

                        windows.append(p)

    frame_count = 0                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  

                        cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])

    try:                cv2.imshow(str(p), im0)

        while True:                cv2.waitKey(1)  

            if cap is not None:

                ret, img0 = cap.read()            if save_img:

                if not ret:                if dataset.mode == "image":

                    break                    cv2.imwrite(save_path, im0)

                            else:

            frame_count += 1                    if vid_path[i] != save_path:

                                    vid_path[i] = save_path

            # Pré-processamento da imagem                        if isinstance(vid_writer[i], cv2.VideoWriter):

            img = letterbox(img0, imgsz, stride=32)[0]                            vid_writer[i].release()

            img = img.transpose((2, 0, 1))[::-1]  # HWC para CHW, BGR para RGB                        if vid_cap:

            img = np.ascontiguousarray(img)                            fps = vid_cap.get(cv2.CAP_PROP_FPS)

            img = img.astype(np.float32) / 255.0  # Normalizar                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))

                                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if len(img.shape) == 3:                        else:

                img = img[None]  # Adicionar dimensão do batch                            fps, w, h = 30, im0.shape[1], im0.shape[0]

                                    save_path = str(Path(save_path).with_suffix(".mp4"))

            # Inferência                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

            interpreter.set_tensor(input_details[0]['index'], img)                    vid_writer[i].write(im0)

            interpreter.invoke()

            pred = interpreter.get_tensor(output_details[0]['index'])        # Print time (inference-only)

                    LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1e3:.1f}ms")

            # Processar detecções

            pred = pred[0]  # Remover dimensão do batch    # Print results

                t = tuple(x.t / seen * 1e3 for x in dt)

            # Filtrar por confiança    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)

            conf_mask = pred[:, 4] > conf_thres    if save_txt or save_img:

            pred = pred[conf_mask]        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""

                    LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")

            if len(pred) == 0:    if update:

                print(f"Frame {frame_count}: Nenhuma detecção")        strip_optimizer(weights[0])

                if view_img:

                    cv2.imshow('SinalizAI - TensorFlow Lite', img0)

                    if cv2.waitKey(1) & 0xFF == ord('q'):def parse_opt():

                        break    

                continue    parser = argparse.ArgumentParser()

                parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "../alfabeto.pt")

            # Extrair boxes, scores e classes    parser.add_argument("--source", type=str, default= "0")

            boxes = pred[:, :4]  # x, y, w, h    parser.add_argument("--data", type=str, default=ROOT / "data/data.yaml")

            scores = pred[:, 4]  # objectness    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640])

            class_scores = pred[:, 5:]  # class scores    parser.add_argument("--conf-thres", type=float, default=0.65)

                parser.add_argument("--iou-thres", type=float, default=0.75)

            # Obter classe com maior score    parser.add_argument("--max-det", type=int, default=3)

            class_ids = np.argmax(class_scores, axis=1)    parser.add_argument("--device", default="")

            class_confidences = np.max(class_scores, axis=1)    parser.add_argument("--view-img", action="store_true")

                parser.add_argument("--save-txt", action="store_true")

            # Score final = objectness * class_confidence    parser.add_argument("--save-format", type=int, default=0)

            final_scores = scores * class_confidences    parser.add_argument("--save-csv", action="store_true")

                parser.add_argument("--save-conf", action="store_true")

            # Converter boxes para formato xyxy    parser.add_argument("--save-crop", action="store_true")

            boxes_xyxy = xywh2xyxy(boxes)    parser.add_argument("--nosave", action="store_false")

                parser.add_argument("--classes", nargs="+", type=int)

            # Reescalar boxes para o tamanho original da imagem    parser.add_argument("--agnostic-nms", action="store_true")

            boxes_xyxy = scale_boxes(img.shape[2:], boxes_xyxy, img0.shape)    parser.add_argument("--augment", action="store_true")

                parser.add_argument("--visualize", action="store_true")

            # Aplicar NMS    parser.add_argument("--update", action="store_true")

            try:    parser.add_argument("--project", default=ROOT / "runs/detect")

                # Converter para formato [y1, x1, y2, x2] para tf.image.non_max_suppression    parser.add_argument("--name", default="exp")

                boxes_tf_format = boxes_xyxy[:, [1, 0, 3, 2]]    parser.add_argument("--exist-ok", action="store_true")

                    parser.add_argument("--line-thickness", default=3, type=int)

                selected_indices = tf.image.non_max_suppression(    parser.add_argument("--hide-labels", default=False, action="store_true")

                    boxes_tf_format,    parser.add_argument("--hide-conf", default=False, action="store_true")

                    final_scores,    parser.add_argument("--half", action="store_true")

                    max_det,    parser.add_argument("--dnn", action="store_true")

                    iou_threshold=iou_thres,    parser.add_argument("--vid-stride", type=int, default=1)

                    score_threshold=conf_thres    opt = parser.parse_args()

                ).numpy()    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1

                    print_args(vars(opt))

                # Filtrar detecções    return opt

                final_boxes = boxes_xyxy[selected_indices]

                final_scores_filtered = final_scores[selected_indices]def main(opt):

                final_classes = class_ids[selected_indices]    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))

                    run(**vars(opt))

            except Exception as e:

                print(f"Erro no NMS: {e}")if __name__ == "__main__":

                continue    opt = parse_opt()

                main(opt)

            # Desenhar detecções
            for i, (box, score, cls) in enumerate(zip(final_boxes, final_scores_filtered, final_classes)):
                x1, y1, x2, y2 = box.astype(int)
                
                # Validar classe
                if cls < len(CLASSES):
                    label = CLASSES[cls]
                else:
                    label = f"Class_{cls}"
                
                confidence = score
                
                # Salvar em CSV se solicitado
                if save_csv:
                    write_to_csv(f"frame_{frame_count}", label, f"{confidence:.2f}")
                
                # Salvar coordenadas em arquivo texto se solicitado
                if save_txt:
                    txt_path = save_dir / "labels" / f"frame_{frame_count}.txt"
                    # Converter coordenadas para formato YOLO (normalizado)
                    img_h, img_w = img0.shape[:2]
                    x_center = ((x1 + x2) / 2) / img_w
                    y_center = ((y1 + y2) / 2) / img_h
                    width = (x2 - x1) / img_w
                    height = (y2 - y1) / img_h
                    
                    with open(txt_path, "a") as f:
                        if save_conf:
                            f.write(f"{cls} {x_center} {y_center} {width} {height} {confidence:.6f}\n")
                        else:
                            f.write(f"{cls} {x_center} {y_center} {width} {height}\n")
                
                # Desenhar box e label
                if not hide_labels:
                    if hide_conf:
                        display_label = label
                    else:
                        display_label = f"{label} {confidence:.2f}"
                    
                    # Cor baseada na classe
                    color = (
                        int(np.random.randint(0, 255)),
                        int(np.random.randint(0, 255)),
                        int(np.random.randint(0, 255))
                    )
                    
                    # Desenhar retângulo
                    cv2.rectangle(img0, (x1, y1), (x2, y2), color, line_thickness)
                    
                    # Desenhar label
                    label_size = cv2.getTextSize(display_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(img0, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
                    cv2.putText(img0, display_label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            print(f"Frame {frame_count}: {len(final_boxes)} detecções")
            
            # Mostrar imagem
            if view_img:
                cv2.imshow('SinalizAI - TensorFlow Lite', img0)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Salvar imagem
            if save_img:
                cv2.imwrite(str(save_dir / f"frame_{frame_count}.jpg"), img0)
            
            # Para imagem única, parar após processar
            if cap is None:
                break
                
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário")
    
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        
        print(f"\nResultados salvos em: {save_dir}")
        if save_csv:
            print(f"CSV salvo em: {csv_path}")

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=ROOT / "best_float16.tflite", help="Caminho para o modelo .tflite")
    parser.add_argument("--source", type=str, default="0", help="Fonte (0 para webcam, caminho para arquivo)")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="Tamanho da imagem")
    parser.add_argument("--conf-thres", type=float, default=0.45, help="Limiar de confiança")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="Limiar de IoU para NMS")
    parser.add_argument("--max-det", type=int, default=1000, help="Máximo de detecções")
    parser.add_argument("--view-img", action="store_true", help="Mostrar resultados")
    parser.add_argument("--save-txt", action="store_true", help="Salvar resultados em *.txt")
    parser.add_argument("--save-csv", action="store_true", help="Salvar resultados em CSV")
    parser.add_argument("--save-conf", action="store_true", help="Salvar confidências em --save-txt")
    parser.add_argument("--save-crop", action="store_true", help="Salvar crops das detecções")
    parser.add_argument("--nosave", action="store_true", help="Não salvar imagens/vídeos")
    parser.add_argument("--classes", nargs="+", type=int, help="Filtrar por classes")
    parser.add_argument("--project", default=ROOT / "runs/detect_tf", help="Pasta do projeto")
    parser.add_argument("--name", default="exp", help="Nome do experimento")
    parser.add_argument("--exist-ok", action="store_true", help="OK se projeto/nome existir")
    parser.add_argument("--line-thickness", default=3, type=int, help="Espessura da linha das bounding boxes")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="Ocultar labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="Ocultar confidências")
    
    opt = parser.parse_args()
    return opt

def main(opt):
    run_tensorflow(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)