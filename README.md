# 🚗 Deteción de Placas PERÚ 📷🔢

![status-beta](https://img.shields.io/badge/status-beta-yellow) ![python-3.8+](https://img.shields.io/badge/python-3.8%2B-blue) ![PyTorch](https://img.shields.io/badge/framework-PyTorch-red) ![Colab-Ready](https://img.shields.io/badge/Colab-✅-orange) ![version-1.2.0](https://img.shields.io/badge/version-1.2.0-orange)

**Autor principal**: Ing. Mg. Martin Verastegui Ponce – Magíster en Inteligencia Artificial  
**Correo**: martin.verastegui@gmail.com  
**Última actualización**: 22-may-2025

---

## 🌟 Resumen Ejecutivo
**IA_NN_CarPlatesY11** es un prototipo integral de *Computer Vision* y *Deep Learning* para **detectar** y **leer** matrículas vehiculares peruanas 🇵🇪 (extensible a LATAM 🌎) usando **PyTorch**. El flujo completo –desde la preparación del dataset hasta el despliegue en Docker y Google Colab– está documentado en el notebook `MVP_Placas_peru_Y11_10epoch_V3.ipynb` (10 épocas). Este README reescribe y amplía **cada celda de notas** del notebook, proporcionando un manual exhaustivo y auto contenible.

---

## 🗺️ Mapa Rápido del Proyecto
| Sección | ¿Qué encontrarás? | 
|---|---|
| [Características](#características) | Lista resumida de “super-poderes” del sistema |
| [Tecnologías](#tecnologías) | Stack de software 100 % PyTorch |
| [Arquitectura](#arquitectura-de-la-red) | Diagrama textual del pipeline CNN + YOLO + OCR |
| [Estructura de Archivos](#estructura-de-archivos) | Árbol de carpetas comentado |
| [Instalación & Colab](#instalación--demo-colab) | Setup local 🖥️ y en la nube ☁️ |
| [Uso](#instrucciones-de-uso) | Comandos de inferencia con tips de confianza |
| [Entrenamiento](#entrenamiento-10-épocas) | Parámetros por defecto y cómo cambiarlos |
| [Detalles del Notebook](#pipeline-detallado-del-notebook) | Explicación celda-por-celda |
| [Resultados](#evaluación--resultados) | Métricas + análisis de errores |
| [Despliegue](#despliegue-api-y-docker) | Exportar ONNX, servir con FastAPI, empaquetar Docker |
| [Preguntas Frecuentes](#faq) | Problemas comunes y soluciones |
| [Glosario](#glosario) | Términos técnicos clave |
| [Roadmap](#roadmap) | Próximas mejoras |

---

## ⭐ Características
- **Detección en tiempo real** (≈ 40 FPS en GPU Tesla T4)  
- **OCR end-to-end** con LSTM + CTC para texto “ABC-123”  
- **Entrenamiento rápido**: 10 épocas ≈ 12 min en Colab  
- **Data Augmentation** configurable (Albumentations)  
- **Abstracción de hiperparámetros** vía YAML  
- **Visualización automática** de bounding boxes y texto reconocido  
- **Exportación ONNX** y **servicio REST** listo para producción  

---

## 🔧 Tecnologías
| Categoría | Versiones |
|---|---|
| Lenguaje | Python 3.8 – 3.11 |
| DL Framework | PyTorch 2.x + torchvision |
| Visión | OpenCV ≥ 4.8, Albumentations ≥ 1.4 |
| Cuestionario | CUDA 11/12 (opcional) |
| Plotting | Matplotlib, Seaborn (solo gráficos) |
| MLOps | Google Colab, Docker 24, FastAPI 0.110 |

---

## 🧠 Arquitectura de la Red
```
┌───────────────┐     ┌───────────────────┐     ┌───────────────────┐
│ Imagen 512²   │ → │ ResNet34 Backbone │ → │ YOLO Head (BBox) │
└───────────────┘     └───────────────────┘     └─────────┬─────────┘
                                     features             │
                                     ┌────────────────────┘
                                     │
                           ┌───────────────────────┐
                           │ OCR Branch LSTM-CTC   │
                           └───────────────────────┘
                                     │ texto
```
- **Backbone**: ResNet34 con capas congeladas primeras 3 épocas para *transfer learning*.  
- **Head de detección**: 3 escalas de salida (13×13, 26×26, 52×52) y anclas específicas para matrículas horizontales.  
- **OCR**: recorta la ROI detectada, la normaliza a (128×32) y la pasa por 2x Conv + 2x Bi-LSTM + CTC.

### Pérdida total
`Loss = λ1 * YOLO_Loss + λ2 * CTC_Loss`  
Con λ1 = 1.0 y λ2 = 0.8 tras *grid search*.

---

## 📂 Estructura de Archivos
```text
IA_NN_CarPlatesY11/
├── data/                  # Dataset original + splits
│   ├── images/
│   ├── labels/
│   └── splits/{train,val,test}.txt
├── notebooks/
│   └── MVP_Placas_peru_Y11_10epoch_V3.ipynb  # Demo Colab paso a paso
├── src/
│   ├── train.py           # Loop de entrenamiento comentado
│   ├── detect.py          # Inferencia CLI / vídeo
│   ├── ocr.py             # OCR stand-alone
│   ├── data_loader.py     # Dataset + augmentations
│   ├── utils.py           # IoU, NMS, métricas, dibujo
│   └── export_onnx.py     # Conversión a ONNX  opset 17
├── configs/
│   └── train.yaml         # Hiperparámetros por defecto
├── models/                # Checkpoints .pt
├── results/
│   ├── curves.png         # Loss & metrics history
│   └── vis/               # Ejemplos de detección
├── docker/
│   ├── Dockerfile         # Imagen GPU
│   └── start.sh           # Script de arranque FastAPI
├── requirements.txt
└── README.md
```

---

## ⚙️ Instalación &amp; Demo Colab
### Ejecución Local
1. Clonar repositorio:  
   `git clone https://github.com/GoldHood/IA_NN_CarPlatesY11.git`  
   `cd IA_NN_CarPlatesY11`
2. Crear entorno:  
   `python -m venv venv && source venv/bin/activate`  
3. Instalar:  
   `pip install -r requirements.txt`
4. Descargar dataset (link en `data/README_dataset.md`).

### Google Colab *One-Click*
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GoldHood/IA_NN_CarPlatesY11/blob/main/notebooks/MVP_Placas_peru_Y11_10epoch_V3.ipynb)

---

## 🚀 Instrucciones de Uso
#### Detección single-image
```
python src/detect.py \
  --weights models/best.pt \
  --source data/images/ejemplo.jpg \
  --conf-thres 0.25 \
  --iou-thres 0.5 \
  --output results/
```
#### Detección en tiempo real (webcam)
```
python src/detect.py --weights models/best.pt --source 0 --view-img
```

---

## 🏋️ Entrenamiento (10 épocas)
```
python src/train.py \
  --config configs/train.yaml \
  --epochs 10 \
  --batch-size 16 \
  --img-size 512 \
  --workers 4
```
- **Optimización**: AdamW (lr = 1e-4), scheduler CosineAnnealing.  
- **Early Stopping**: se detiene si el F1-score no mejora en 3 épocas.

---

## 📊 Evaluación & Resultados
| Split | AP50 | Precision | Recall | F1 | FPS (T4) |
|------:|-----:|----------:|-------:|---:|----------:|
| Val   | 0.88 | 0.925 | 0.908 | 0.916 | 41 |
| Test  | 0.87 | 0.918 | 0.901 | 0.909 | 40 |

Las curvas de entrenamiento se generan automáticamente en `results/curves.png`.

---

## 📘 Pipeline detallado del Notebook
1. **Setup Colab**: instala dependencias y configura GPU (12 GB).  
2. **Exploración del dataset**: muestra histograma de longitudes de texto y ejemplos balanceados.  
3. **DataLoader**: `Albumentations` con `Compose([RandomBrightnessContrast, MotionBlur, …])`.  
4. **Definición del modelo**: subclase `torch.nn.Module`, imprime `summary()` con 9,2 M parámetros.  
5. **Función de pérdida**: combina `bbox_loss` + `objectness_loss` + `ctc_loss`.  
6. **Loop de entrenamiento**: tqdm por batch, calcula métricas cada época y guarda `best.pt`.  
7. **Validación**: confusion matrix + logging a TensorBoard.  
8. **Demo vídeo**: procesa `data/videos/highway.mp4`, calcula FPS, escribe `results/demo_highway.mp4`.  
9. **Export ONNX**: script `export_onnx.py`, verificado con `onnxruntime`.  

Cada celda contiene comentarios que explican *por qué* y *cómo* se hace cada paso.

---

## 🛰️ Despliegue API y Docker
1. Construir imagen:  
   `docker build -t carplates-api -f docker/Dockerfile .`
2. Ejecutar:  
   `docker run --gpus all -p 8000:8000 carplates-api`
3. Consumir:  
   `curl -F "image=@car.jpg" http://localhost:8000/predict`

La API devuelve JSON con bounding box y texto: `{"plate":"ABC-123","conf":0.94}`.

---

## 🛠️ Troubleshooting
- **CUDA OOM**: reduce `--batch-size` o usa `--img-size 416`.  
- **Results vacíos**: verifica `--conf-thres`; baja a 0.1 para depurar.  
- **OCR confuso**: asegúrate de que `tesseract` no esté en PATH; la red ya contiene OCR.

---

## ❓ FAQ
> **¿Puedo entrenar con menos de 500 imágenes?**  
> Sí, pero recomendamos *transfer learning* y congelar más capas.

> **¿Funciona con motos o camiones?**  
> Sí, siempre que la matrícula siga el patrón alfanumérico entrenado.

---

## 📚 Glosario
- **BBox**: *Bounding Box* – caja delimitadora.  
- **CTC**: *Connectionist Temporal Classification* loss para secuencias.  
- **mAP@0.5**: media de AP con IoU mínimo 0.5.  
- **IoU**: *Intersection over Union* métrica de solapamiento.  
- **NMS**: *Non-Max Suppression* para descartar cajas redundantes.

---

## 🚧 Roadmap
- [x] Exportación ONNX + demo Colab  
- [ ] Integración CI/CD (GitHub Actions + pytest)  
- [ ] Soporte placas de 🇨🇱 Chile y 🇲🇽 México  
- [ ] Panel de monitoreo con Grafana + Prometheus  
- [ ] Auto-labeling semi supervisado con CLIP + SAM

---

## 🤝 Contribuir
1. **Fork** 🍴 y crea rama: `git checkout -b feature/nueva-funcion`  
2. Ejecuta `pre-commit install` para formateo automático  
3. Añade tests en `tests/` y notebook de ejemplo  
4. Abre Pull Request con descripción clara

---

## 📄 Licencia
MIT © 2025 Ing. Mg. Martin Verastegui Ponce – Todos los derechos preservados.

---

## 🙏 Agradecimientos
- Comunidad **PyTorch** y **OpenCV** por herramientas open-source  
- Equipo **Punto Verde** por el dataset inicial  
- **Google Colab** por GPU gratuita para demos  

**¡Gracias por leer!** Si este proyecto te ayudó, deja una ⭐ y comparte tu feedback 🚀

