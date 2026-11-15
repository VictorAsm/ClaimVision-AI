import io
import base64
import os
import re
from datetime import datetime
from PIL import Image
import numpy as np
from flask import request
from flask_restx import Namespace, Resource

from backend import config
from ml.models.object_detection_model import ObjectDetector
from ml.models.classifier_model import ClassifierModel
from ml.models.nlp_transformer import NLPTransformer
from ml.preprocessing.processor import Preprocesor

# Inicialización de modelos y componentes de ML
modelClassifier = ClassifierModel(
    model_path=config.CLASSIFICATION_MODEL_PATH,
    labels_path=config.CLASSIFICATION_LABEL_PATH,
    model_checkpoints_path=config.CLASSIFICATION_CHECKPOINT_PATH,
)

modelDetector = ObjectDetector(
    model_path=config.OBJECT_DETECTION_MODEL_PATH,
    labels_path=config.OBJECT_DETECTION_LABEL_PATH,
    save_path=config.NO_CLASSIFY_PATH,
    model_checkpoints_path=config.OBJECT_DETECTION_CHECKPOINT_PATH,
    batch_size=4,
    threshold=0.5
)

modelNLP = NLPTransformer(
    api_key=config.GEMINI_API_KEY
)

modelPreprocessor = Preprocesor(
    no_classify_path=config.NO_CLASSIFY_PATH,
    retrain_raw_path=config.RAW_PATH_RETRAIN,
    original_raw_path=config.RAW_PATH_ORIGINAL,
    retrain_processed_path=config.PROCESSED_PATH_RETRAIN,
    original_processed_path=config.PROCESSED_PATH_ORIGINAL,
)

api = Namespace('api', description='Endpoints para inferencia, entrenamiento y procesamiento de siniestros')


def sanitize_numpy(data):
    """Convierte tipos de NumPy (ej. float32, int64) a tipos estándar de Python para serialización JSON."""
    if isinstance(data, dict):
        return {k: sanitize_numpy(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_numpy(v) for v in data]
    elif isinstance(data, (np.generic, np.ndarray)):
        return data.tolist() if isinstance(data, np.ndarray) else data.item()
    return data


@api.route('/ping')
class PingResource(Resource):
    def get(self):
        return {"status": "ok", "message": "pong"}, 200

    def post(self):
        return {"status": "ok", "message": "pong"}, 200


@api.route('/process_data')
class ProcessDataResource(Resource):
    def post(self):
        try:
            data = request.json or {}
            process_method = data.get('method')
            modelPreprocessor.process_data(process_method, "car-images")
            return {"status": "ok", "message": "Datos procesados exitosamente"}, 200
        except Exception as e:
            return {"error": str(e)}, 500


@api.route('/train')
class TrainResource(Resource):
    def post(self):
        try:
            modelClassifier.train(f"{config.PROCESSED_PATH_ORIGINAL}/csv", 1)
            modelDetector.train(config.PROCESSED_PATH_ORIGINAL, 1)
            return {"status": "ok", "message": "Modelos entrenados exitosamente"}, 200
        except Exception as e:
            return {"error": str(e)}, 500


@api.route('/predict')
class PredictResource(Resource):
    def post(self):
        try:
            data = request.json or {}
            base64_image = data.get('image')
            siniestro = data.get('siniestro')

            if not base64_image:
                return {"error": "No image provided"}, 400

            # Limpiar prefijo data URI si viene de un frontend web (data:image/jpeg;base64,...)
            pattern = r'^data:image\/(jpeg|jpg|png);base64,'
            base64_image = re.sub(pattern, '', base64_image)
            image_bytes = base64.b64decode(base64_image)

            # Cargar imagen en memoria sin necesidad de I/O en disco
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            # Opcional: Persistir copia de auditoría si está configurado
            image_name = datetime.now().strftime("%Y%m%d_%H%M%S")
            audit_dir = os.path.join(config.NO_CLASSIFY_PATH, "images")
            if os.path.exists(audit_dir):
                audit_path = os.path.join(audit_dir, f"{image_name}.jpg")
                with open(audit_path, 'wb') as f:
                    f.write(image_bytes)

            # Ejecutar inferencias
            classifier_outputs = modelClassifier.predict(image)
            detector_outputs = modelDetector.predict(image, image_name)

            # Verificar si los modelos están disponibles
            models_unavailable = (
                isinstance(classifier_outputs, str) and "No hay un modelo cargado" in classifier_outputs
            ) or (
                isinstance(detector_outputs, str) and "No hay un modelo cargado" in detector_outputs
            )

            if models_unavailable:
                return {
                    'status': 'WARNING',
                    'msg': "Modelos no cargados o requieren entrenamiento previo",
                    'data': {
                        'damaged_scores': classifier_outputs,
                        'detected_objects': detector_outputs
                    }
                }, 200

            # Generación de informe pericial con Gemini LLM (o fallback determinístico si no hay API key)
            nlp_outputs = modelNLP.predict(siniestro, classifier_outputs, detector_outputs)

            response_data = {
                'status': 'OK',
                'msg': nlp_outputs,
                'data': {
                    'damaged_scores': sanitize_numpy(classifier_outputs),
                    'detected_objects': sanitize_numpy(detector_outputs)
                }
            }

            return response_data, 200

        except Exception as e:
            return {"error": str(e)}, 500
