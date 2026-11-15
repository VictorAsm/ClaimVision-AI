import json

try:
    from google import genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False


class NLPTransformer:
    def __init__(self, api_key: str = ""):
        self.api_key = (api_key or "").strip()
        self.client = None
        self.model_name = "gemini-3.8-flash"

        if self.api_key and HAS_GENAI:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                print(f"[NLPTransformer] Advertencia al inicializar cliente Gemini: {e}")

    def _build_payload(self, siniestro_data: dict, damaged_scores: dict, detected_objects: list) -> dict:
        siniestro_info = (siniestro_data or {}).get("siniestro", siniestro_data or {})
        ubicacion = siniestro_info.get("ubicacion", {})

        return {
            "siniestro": {
                "numero": siniestro_info.get("numero", "S/N"),
                "fecha": siniestro_info.get("fecha", "No especificada"),
                "ubicacion": {
                    "direccion": ubicacion.get("direccion", "No especificada"),
                    "pais": ubicacion.get("pais", "No especificado"),
                },
                "descripcion": siniestro_info.get("descripcion", "Sin descripción adicional"),
            },
            "damaged_scores": damaged_scores,
            "detected_objects": detected_objects,
        }

    def _deterministic_report(self, payload: dict) -> str:
        """Genera un dictamen pericial determinístico cuando no hay API Key de Gemini configurada."""
        scores = payload.get("damaged_scores") or {}
        detections = payload.get("detected_objects") or []
        siniestro = payload.get("siniestro") or {}

        if not isinstance(scores, dict) or not scores:
            return "No se pudieron obtener puntajes de clasificación para emitir el dictamen."

        top_class = max(scores, key=scores.get)
        top_score = float(scores[top_class]) * 100

        if top_class.lower() == "unknown":
            return (
                f"Dictamen Técnico (Siniestro #{siniestro.get('numero')}): "
                f"La imagen suministrada no corresponde a un vehículo reconocible o carece de encuadre válido "
                f"(confianza: {top_score:.1f}%). Se solicita recargar evidencia fotográfica."
            )

        if top_class.lower() == "good":
            return (
                f"Dictamen Técnico (Siniestro #{siniestro.get('numero')}): "
                f"El análisis de visión computacional no detecta daños estructurales aparentes en el vehículo "
                f"(estado '{top_class}' con {top_score:.1f}% de confianza)."
            )

        # Caso: damaged
        if detections and isinstance(detections, list):
            parts = []
            for d in detections:
                if isinstance(d, dict) and "label" in d:
                    conf = float(d.get("score", 0)) * 100
                    parts.append(f"{d['label']} ({conf:.1f}%)")
            parts_str = ", ".join(parts) if parts else "zonas pendientes de peritaje manual"
            return (
                f"Dictamen Técnico (Siniestro #{siniestro.get('numero')}): "
                f"Se confirma presencia de daño vehicular (confianza: {top_score:.1f}%). "
                f"Secciones comprometidas detectadas: {parts_str}. "
                f"Consistente con el reporte en {siniestro.get('ubicacion', {}).get('direccion', 'la ubicación declarada')}."
            )

        return (
            f"Dictamen Técnico (Siniestro #{siniestro.get('numero')}): "
            f"El clasificador global detecta daño vehicular ({top_score:.1f}%), aunque el detector de partes "
            f"no aisló piezas específicas sobre el umbral de confianza. Se sugiere revisión pericial complementaria."
        )

    def predict(self, siniestro_data: dict, damaged_scores: dict, detected_objects: list) -> str:
        payload = self._build_payload(siniestro_data, damaged_scores, detected_objects)

        # Si no hay cliente Gemini activo, usar fallback determinístico
        if not self.client:
            return self._deterministic_report(payload)

        prompt = (
            "Eres un perito liquidador de siniestros automotrices experto. "
            "Analiza el siguiente JSON que combina los datos declarados del siniestro con las predicciones "
            "de dos modelos de visión artificial (`damaged_scores` del clasificador de estado y `detected_objects` del detector de piezas dañadas).\n\n"
            "Reglas para tu respuesta:\n"
            "1. Si la clase con mayor puntaje en `damaged_scores` es `damaged`, verifica los elementos en `detected_objects` y redacta un informe pericial natural, claro y profesional detallando los daños encontrados y su coherencia con la descripción del hecho.\n"
            "2. Si la clase con mayor puntaje es `good`, indica formalmente que la inspección visual no arroja evidencia de siniestro ni daños materiales.\n"
            "3. Si la clase con mayor puntaje es `unknown`, indica que la fotografía cargada no es válida o no corresponde a un vehículo.\n"
            "4. Redacta en español de forma directa y concisa (máximo 1 párrafo), sin mencionar nombres de variables técnicas ni formato JSON.\n\n"
            f"Datos del caso:\n{json.dumps(payload, ensure_ascii=False, default=str)}"
        )

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            if response and response.text:
                return response.text.strip()
        except Exception as e:
            print(f"[NLPTransformer] Error en llamada a Gemini API ({e}). Usando reporte determinístico.")

        return self._deterministic_report(payload)
