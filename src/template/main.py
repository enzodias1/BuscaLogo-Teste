import ast
import os
from deteccao_logo import logo_detection_pipeline
from aprimoramento_regiao import region_enhancement
from reconhecimento import region_recognition_pipeline
from serperapi import get_images_from_google
from utils import get_images_from_html
from io import BytesIO
import numpy as np
import base64
import time
from sklearn.cluster import KMeans
from PIL import Image

def convert_pil_to_base64(pil_image):
	buffered = BytesIO()
	pil_image = pil_image.convert("RGB")  
	pil_image.save(buffered, format="PNG")
	return base64.b64encode(buffered.getvalue()).decode('utf-8')

def extract_colors(cluster, centroids, exact=False):
	labels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
	(hist, _) = np.histogram(cluster.labels_, bins=labels)
	hist = hist.astype("float")
	hist /= hist.sum()

	if not exact:
		centroids = centroids.astype("int")

	colors = sorted([(percent, color) for (percent, color) in zip(hist, centroids)], key=lambda x: x[0])
	
	return colors

def fill_image_background(raw_image):
	TRANSP_THR = 128
	fill_color = []
	if raw_image.shape[-1] < 4:
		return raw_image, fill_color

	wdt = raw_image.shape[0]
	hgt = raw_image.shape[1]

	border_image = np.ndarray((wdt + 2, hgt + 2, 4))
	border_image[1:1 + wdt, 1:1 + hgt] = raw_image

	clts_data = []
	for x in range(1, wdt + 1):
		for y in range(1, hgt + 1):
			if border_image[x][y][-1] >= TRANSP_THR and (
					border_image[x - 1][y][-1] < TRANSP_THR or border_image[x][y - 1][-1] < TRANSP_THR or
					border_image[x + 1][y][-1] < TRANSP_THR or border_image[x][y + 1][-1] < TRANSP_THR):
				clts_data.append(border_image[x][y][:3].tolist())

	cluster = KMeans(n_clusters=5).fit(np.array(clts_data))
	colors = extract_colors(cluster, cluster.cluster_centers_)

	dominant_color = colors[-1][1].tolist()
	dominant_color_average = int(sum(dominant_color) / 3)

	if dominant_color_average > 170:
		fill_color = [0, 0, 0]
	else:
		fill_color = [255, 255, 255]

	new_image = np.ndarray((wdt, hgt, 3))

	for x in range(wdt):
		for y in range(hgt):
			if raw_image[x][y][-1] < TRANSP_THR:
				new_image[x][y] = fill_color
			else:
				new_image[x][y] = raw_image[x][y][:3]

	return new_image, fill_color

def convert_image(img_str):
	if not img_str.mode.upper().startswith('RGB'):
		img_str = img_str.convert('RGBA')
	nparr = np.array(img_str)
	return nparr
    
def fill_base64_image(decoded_img):
	if not isinstance(decoded_img, bytes):
		decoded_img = base64.b64decode(decoded_img)
		
	raw_img = Image.open(BytesIO(decoded_img))
	new_image, _ = fill_image_background(convert_image(raw_img))
	new_image = Image.fromarray(np.array(new_image, dtype=np.uint8))
	img_byte_arr = BytesIO()
	new_image.save(img_byte_arr, 'PNG')
	img_byte_arr = img_byte_arr.getvalue()
	return base64.b64encode(img_byte_arr).decode()


def convert_numpy_floats(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_floats(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_floats(item) for item in obj]
    return obj

class BuscaLogo:
	def __init__(self):
		self.logo_detection_pipeline = logo_detection_pipeline(detection_model_path=os.path.join(os.getcwd(), "models/yolov7_1280x1280_best_recall.xml"))
		self.region_recognition_pipeline = region_recognition_pipeline()
		self.region_enhancement = region_enhancement()
		self.get_images_from_google = get_images_from_google()

	def invoke_pipeline(self, input):
		t0_total_pipeline = time.time()
		print("Initializing the pipeline...")
		min_score, min_area_size, max_height = 0.001, 800, 448
		company_name, screenshot_url, html_url = input['company_name'], input['screenshot_url'], input['html_url']

		print("Initializing logo_detection_pipeline...")
		t0_logo_detection_pipeline = time.time()
		cropped_images = self.logo_detection_pipeline.detect(screenshot_url, min_score, min_area_size, max_height, header=True)
		cropped_images.extend(self.logo_detection_pipeline.detect(screenshot_url, min_score, min_area_size, max_height, header=False))
		print("total time logo_detection_pipeline:", time.time() - t0_logo_detection_pipeline)

		print("Initializing serperAPI_pipeline...")
		t0_serperAPI_pipeline = time.time()
		cropped_images.extend(self.get_images_from_google.get_images(query='{} logo'.format(company_name)))
		print("total time serperAPI_pipeline:", time.time() - t0_serperAPI_pipeline)

		print("Initializing down_image_from_html_pipeline...")
		down_image_from_html_pipeline_t0 = time.time()
		html_images = get_images_from_html(html_url)
		cropped_images.extend(html_images)
		print("total time down_image_from_html_pipeline:", time.time() - down_image_from_html_pipeline_t0)

		print("Initializing region_enhancement_pipeline...")
		region_enhancement_pipeline_t0 = time.time()
		print("total cropped_images: ", len(cropped_images))
		improved_images = self.region_enhancement.run_region_enhancement_pipeline(cropped_images[0:50])[0:40]  
		print("total time region_enhancement_pipeline:", time.time() - region_enhancement_pipeline_t0)
		print("total improved_images:", len(improved_images))

		print("Initializing region_recognition_pipeline...")
		region_recognition_pipeline_t0 = time.time()
		brand_logos_indexes = self.region_recognition_pipeline.run_region_recognition_pipeline(html_url, screenshot_url,[response['image'] for response in improved_images])
		brand_logos_indexes = ast.literal_eval(brand_logos_indexes)
		print("total time region_recognition_pipeline:", time.time() - region_recognition_pipeline_t0)

		if len(improved_images) == 0:
			return []
		brand_logos = [improved_images[int(region['region'].split("_")[1])] for region in brand_logos_indexes['brand_logos']]
		for logo in brand_logos:
			print(type(logo['image']))
			logo['max_dim'] = max(logo['image'].size)
		brand_logos_gold = [logo for logo in brand_logos if logo['max_dim'] > 50]
		brand_logos_gold = [{**logo, 'image': fill_base64_image(convert_pil_to_base64(logo['image']))} for logo in brand_logos_gold]
		print("TOTAL PIPELINE TIME: ", time.time() - t0_total_pipeline)
  
		return convert_numpy_floats(brand_logos_gold)