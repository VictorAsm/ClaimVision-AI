import sys
from pathlib import Path

# Asegurar que la raíz del proyecto esté en sys.path para importar ml y config
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from flask import Flask
from flask_restx import Api
from flask_cors import CORS
from backend.routes.routes import api as api_namespace

app = Flask(__name__)
CORS(app)

api = Api(
    app,
    title='ClaimVision-AI API',
    version='1.0',
    description='API para clasificación de daños, detección de piezas dañadas y síntesis multimodal de siniestros'
)

api.add_namespace(api_namespace)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
