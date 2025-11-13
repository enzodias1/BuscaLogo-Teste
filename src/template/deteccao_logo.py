import cv2
import numpy as np
import openvino as ov  
from PIL import Image
from utils import get_cv2_image  

class logo_detection_pipeline:
    def __init__(self, detection_model_path):
        core = ov.Core()
        self.detecion_model = core.compile_model(
            core.read_model(detection_model_path), 'CPU')

    def letterbox(self, im, new_shape=(1280, 1280), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
        shape = im.shape[:2]  
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:
            r = min(r, 1.0)

        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - \
            new_unpad[1]  

        if auto:  
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  

        dw /= 2  
        dh /= 2

        if shape[::-1] != new_unpad: 
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(
            im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  
        return im, r, (dw, dh)

    def detect(self, image_path, min_score, min_area_size, max_height, header):
        img0 = get_cv2_image(image_path)
        if img0 is None:
            print(
                f"Falha ao descarregar a imagem em logo_detection: {image_path}")
            return []

        img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        height, width, ch = img0.shape
        if header:
            if height > max_height:
                height = max_height
            if width > 3000:
                width = 3000
            img = img0[0:height, 0:width]
        else:
            if width > 3000:
                width = 3000
            img = img0[max(0, height-448):height, 0:width]

        image = img.copy()
        image, ratio, dwdh = self.letterbox(image, auto=False)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)

        im = image.astype(np.float32)
        im /= 255

        result = self.detecion_model(im)
        pred = result[list(result.keys())[0]]

        regions = []
        for i, (batch_id, x1, y1, x2, y2, cls_id, score) in enumerate(pred):
            box = np.array([x1, y1, x2, y2])
            box -= np.array(dwdh * 2)
            box /= ratio
            box = box.round().astype(np.int32).tolist()
            regions.append(
                {"score": score, "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]})

        regions = [region for region in regions if region['score'] > min_score]
        regions = [{**region, 'size': (region['x2'] - region['x1'])
                    * (region['y2'] - region['y1'])} for region in regions]
        regions = [region for region in regions if region['size']
                   >= min_area_size]

        cropped_images = []
        for region in regions:
            x1, y1, x2, y2 = region['x1'], region['y1'], region['x2'], region['y2']
            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0
            if x2 > img.shape[1]:
                x2 = img.shape[1]
            if y2 > img.shape[0]:
                y2 = img.shape[0]

            if x1 < x2 and y1 < y2: 
                cropped_images.append(
                    Image.fromarray(img.copy()[y1:y2, x1:x2]))

        return cropped_images
