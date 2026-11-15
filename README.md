# ClaimVision-AI: Automated Vehicle Damage Assessment & Multimodal Claim Reporting

[![Python 3.11](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Faster--R--CNN-EE4C2C.svg)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow%20%2F%20Keras-CNN%20Classification-FF6F00.svg)](https://www.tensorflow.org/)
[![Flask RESTX](https://img.shields.io/badge/API-Flask--RESTX%20(Swagger%20UI)-000000.svg)](https://flask-restx.readthedocs.io/)
[![LLM](https://img.shields.io/badge/Google%20Gemini-2.5--Flash-4285F4.svg)](https://ai.google.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Sistema integral end-to-end de **Visión Computacional y Procesamiento de Lenguaje Natural (Multimodal)** diseñado para el triaje, peritaje técnico y catalogación automatizada de daños vehiculares en siniestros de seguros automotores (**InsurTech**).

---

## 📌 Contexto y Problemática

En la industria de seguros automotrices, la auditoría y liquidación de siniestros enfrenta cuellos de botella críticos debido a la evaluación manual de fotografías de siniestros:
- Tiempos de respuesta lentos para los asegurados.
- Subjetividad e inconsistencia en la estimación de severidad.
- Sobrecarga operativa de peritos e inspectores en siniestros leves o repetitivos.

**ClaimVision-AI** resuelve este problema mediante un pipeline automatizado de 3 capas que recibe la fotografía del siniestro junto con su metadata y genera un informe técnico pericial estructurado en segundos.

---

## 🏗️ Arquitectura del Sistema

El pipeline procesa la evidencia a través de tres componentes especializados e independientes:

```
                  ┌───────────────────────────────────────────────┐
                  │          Fotografía del Siniestro             │
                  └───────────────────────┬───────────────────────┘
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    ▼                                           ▼
      ┌───────────────────────────┐               ┌───────────────────────────┐
      │  Clasificación de Daño    │               │   Detección de Daños      │
      │  (CNN Deep Learning)      │               │   (Faster R-CNN PyTorch)  │
      │                           │               │                           │
      │  • damaged                │               │  • Bounding boxes         │
      │  • good                   │               │  • Componentes afectados  │
      │  • unknown                │               │  • Confidence scores      │
      └─────────────┬─────────────┘               └─────────────┬─────────────┘
                    │                                           │
                    └─────────────────────┬─────────────────────┘
                                          │
                                          ▼
                         ┌─────────────────────────────────┐
                         │   Metadata del Siniestro        │
                         │   (Fecha, Ubicación, Relato)   │
                         └────────────────┬────────────────┘
                                          │
                                          ▼
                         ┌─────────────────────────────────┐
                         │   Síntesis Pericial (LLM)       │
                         │   (Google Gemini 2.5 Flash)     │
                         └────────────────┬────────────────┘
                                          │
                                          ▼
                         ┌─────────────────────────────────┐
                         │   Respuesta JSON Estructurada   │
                         │   (API REST / Frontend UI)      │
                         └─────────────────────────────────┘
```

1. **Clasificador de Severidad (TensorFlow / Keras):** Red Neuronal Convolucional profunda que clasifica la imagen en tres estados: vehículo dañado (`damaged`), sin daños aparentes (`good`) o imagen no clasificable (`unknown`).
2. **Localizador de Autopartes Dañadas (PyTorch / Torchvision):** Modelo de detección de objetos basado en Faster R-CNN con backbone ResNet, entrenado sobre anotaciones en formato COCO para identificar y delimitar con bounding boxes las áreas afectadas (paragolpes, capó, puertas, ópticas, etc.).
3. **Módulo de Síntesis Pericial (NLP / LLM):** Integra los scores de clasificación, la detección espacial de autopartes y los datos del acta del siniestro para redactar una minuta técnica objetiva y contextualizada en lenguaje natural.

---

## 📁 Estructura del Proyecto

El repositorio está organizado como un monorepo modular desacoplado:

```text
ClaimVision-AI/
├── ml/                         # Núcleo de Machine Learning
│   ├── models/                 # Arquitecturas: Classifier, ObjectDetector, NLPSynthesizer
│   ├── preprocessing/          # Pipelines de datos, Dataloaders y transformaciones COCO
│   ├── weights/                # Pesos de los modelos (.pth y .keras - locales)
│   ├── data/                   # Datasets crudos y procesados (locales)
│   └── tests/                  # Casos de prueba e imágenes de siniestros
├── backend/                    # Capa de Servicio y API REST
│   ├── app.py                  # Entrypoint de Flask y documentación Swagger UI
│   ├── routes/                 # Controladores y endpoints de la API (/predict, /train, etc.)
│   └── config.py               # Configuración de runtime y rutas
├── frontend/                   # Interfaz de usuario en React para demostración interactiva
├── notebooks/                  # Ciclo completo de investigación, EDA y entrenamiento
│   ├── 01_dataset_preprocessing.ipynb
│   ├── 02_data_augmentation.ipynb
│   ├── 03_damage_classification_cnn.ipynb
│   ├── 03b_multilabel_experiment.ipynb
│   ├── 04_llm_claim_synthesis.ipynb
│   ├── 05_object_detection_faster_rcnn.ipynb
│   ├── 06_end_to_end_inference.ipynb
│   └── 07_evaluation_metrics.ipynb
├── docs/                       # Documentación técnica y colección Postman
├── .env.example                # Template de variables de entorno
├── requirements.txt            # Dependencias del proyecto
└── README.md
```

---

## 🔬 Flujo de Investigación y Experimentación (Notebooks)

La carpeta `notebooks/` documenta de forma reproducible el ciclo de vida del desarrollo de los modelos:

- `01_dataset_preprocessing.ipynb`: Ingesta, estructuración de imágenes y normalización de etiquetas COCO JSON a DataFrames.
- `02_data_augmentation.ipynb`: Técnicas de aumento de datos sintéticos y partición balanceada de datasets (Train/Val/Test).
- `03_damage_classification_cnn.ipynb`: Diseño, entrenamiento, regularización (Dropout, BatchNorm) y validación de la CNN de severidad.
- `03b_multilabel_experiment.ipynb`: Experimentación de clasificación multietiqueta para predicción simultánea de atributos.
- `04_llm_claim_synthesis.ipynb`: Prototipado del prompt engineering y conexión con la API de Google Gemini (`google-genai`).
- `05_object_detection_faster_rcnn.ipynb`: Fine-tuning de Faster R-CNN para la detección y delimitación de daños estructurales.
- `06_end_to_end_inference.ipynb`: Pipeline integrado de inferencia combinando visión y lenguaje.
- `07_evaluation_metrics.ipynb`: Evaluación exhaustiva: curvas de pérdida/accuracy, mAP (Mean Average Precision), matrices de confusión y métricas por clase.

---

## 🚀 Puesta en Marcha

### 1. Requisitos Previos
- Python 3.11 o superior.
- Git.

### 2. Configuración del Entorno
Clonar el repositorio y crear el entorno virtual:
```bash
python -m venv venv

# En Linux / macOS:
source venv/bin/activate

# En Windows:
.\venv\Scripts\activate
```

Instalar dependencias:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Variables de Entorno
Copiar el archivo de ejemplo y configurar las credenciales:
```bash
cp .env.example .env
```
Editar `.env` y definir la clave de API de Google Gemini (si se omite, el sistema opera automáticamente con un motor de dictamen determinístico basado en reglas):
```env
GEMINI_API_KEY=tu_api_key_de_gemini_aqui
```

### 4. Ejecución de la API
Iniciar el servidor de backend:
```bash
python backend/app.py
```
- La API quedará disponible en `http://localhost:5000/`.
- Acceder a `http://localhost:5000/` en el navegador para explorar la documentación interactiva generada con **Swagger UI**.
- Colección de pruebas disponible en: `docs/ClaimVision-AI.postman_collection.json`.

---

## 📄 Licencia

Distribuido bajo la Licencia MIT. Consulta el archivo `LICENSE` para más información.
