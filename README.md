# 🚗 Deteción de Placas PERÚ 📷🔢

![status-beta](https://img.shields.io/badge/status-beta-yellow) ![python-3.8+](https://img.shields.io/badge/python-3.8%2B-blue) ![license-MIT](https://img.shields.io/badge/license-MIT-green) ![version-1.0.0](https://img.shields.io/badge/version-1.0.0-orange)

**Autor**: Ing. Mg. Martin Verastegui Ponce   |  Magíster en IA  
**Correo**: martin.verastegui@gmail.com  
**Fecha**: 2025-05-22

---

## 📖 Descripción

**IA_NN_CarPlatesY11** es un proyecto de **Inteligencia Artificial** que implementa una **Red Neuronal Profunda** para la **detección** y **lectura** de **matrículas vehiculares** en imágenes fijas y secuencias de vídeo. Está diseñado como un MVP (Producto Mínimo Viable) entrenado con nuestras propias imágenes de placas peruanas 🇵🇪, aunque su arquitectura permite adaptarse a contextos de Latinoamérica y el mundo 🌎.

Este repositorio contiene:  
- 📸 **Preprocesado** y aumentación de datos (Data Augmentation)  
- 🏋️ **Entrenamiento** de modelo con PyTorch y TensorFlow  
- 🔎 **Detección** en tiempo real usando OpenCV  
- 📈 **Evaluación** de desempeño: precisión, recall, F1-score  
- 🛠️ **Guías** para personalizar hiperparámetros, dataset y arquitectura  
- 🤝 **Guía** completa para contribuir y desplegar  

---

## 📑 Tabla de Contenidos

1. [Características ⭐](#características-️)  
2. [Tecnologías 🔧](#tecnologías-️)  
3. [Arquitectura de la Red 🧠](#arquitectura-de-la-red-️)  
4. [Estructura del Proyecto 📂](#estructura-del-proyecto-️)  
5. [Requisitos Previos & Instalación ⚙️](#requisitos-previos--instalación-️)  
6. [Uso Rápido 🚀](#uso-rápido-️)  
7. [Experimentos & Resultados 📊](#experimentos--resultados-️)  
8. [Métricas de Evaluación 📏](#métricas-de-evaluación-️)  
9. [Dataset 📚](#dataset-️)  
10. [Personalización 🔄](#personalización-️)  
11. [Contribuye 🤝](#contribuye-️)  
12. [Mantenimiento 🧹](#mantenimiento-️)  
13. [Licencia 📄](#licencia-️)  
14. [Contacto & Agradecimientos 🙏](#contacto--agradecimientos-️)  
15. [Referencias 📚](#referencias-️)  

---

## ⭐ Características

- 📸 **Detección en tiempo real** de matrículas con alta precisión  
- 🎨 **Preprocesado**: filtrado, corrección de brillo/contraste, normalización  
- 🔄 **Aumentación de datos**: rotación, zoom, ruido, cambio de perspectiva  
- 🏋️ **Entrenamiento** flexible: soporta CPU/GPU, multi-epoch, early stopping  
- 🔍 **Inferencia**: script `detect.py` para imágenes y vídeo  
- 📈 **Visualización** de curvas de entrenamiento (pérdida, accuracy)  
- ⚙️ **Configuración** por YAML/JSON para hiperparámetros  
- 🌐 **Internacional**: adaptaciones fáciles a otros países y formatos de placa  

---

## 🔧 Tecnologías

- Python 3.8+  
- PyTorch 2.x / TensorFlow 2.x  
- OpenCV para procesamiento de imágenes  
- Albumentations para aumentación de datos  
- Matplotlib / TensorBoard para gráficas  
- NumPy, Pandas  
- Docker (opcional) para contenedorización  
- GitHub Actions (planteado) para CI/CD  

---

## 🧠 Arquitectura de la Red

1. **Backbone**: CNN (p.ej. ResNet-34 adaptada)  
2. **Cabeza de detección**: anclas y bounding boxes  
3. **OCR**: capa LSTM + CTC para reconocimiento de caracteres  
4. **Post-procesado**: filtrado por confianza, NMS (Non-Max Suppression)  

Flujo de datos:  
- Entrada (RGB 512×512)  
    └─► CNN Backbone  
           └─► Feature Maps  
                  ├─► Detector de cajas  
                  └─► Reconocimiento OCR  
                         └─► Texto plano de matrícula  

---

## 📂 Estructura del Proyecto

IA_NN_CarPlatesY11/  
├── data/  
│   ├── images/           📁 Imágenes originales  
│   ├── labels/           📁 Anotaciones (YOLO txt / COCO json)  
│   └── splits/           📁 train/val/test  
├── models/               📁 Pesos entrenados (.pt, .h5)  
├── notebooks/            📁 Experimentación en Jupyter  
│   └── MVP_Placas_peru_Y11_10epoch_V3.ipynb  
├── src/  
│   ├── train.py          🏋️ Script de entrenamiento  
│   ├── detect.py         🔎 Script de detección  
│   ├── utils.py          🧩 Funciones auxiliares  
│   └── data_loader.py    📦 Carga y aumentación de datos  
├── results/              📊 Curvas, logs y reportes  
├── requirements.txt      📦 Dependencias  
├── Dockerfile            🐳 Contenedor (opcional)  
├── CONTRIBUTING.md       🤝 Guía de contribución  
├── LICENSE               📄 Licencia MIT  
└── README.md             📝 Este documento  

---

## ⚙️ Requisitos Previos & Instalación

1. Clona el repositorio      
        git clone https://github.com/GoldHood/IA_NN_CarPlatesY11.git      
        cd IA_NN_CarPlatesY11  

2. Crea un entorno virtual      
        python -m venv venv      
        source venv/bin/activate  # Linux/macOS      
        venv\Scripts\activate     # Windows  

3. Instala dependencias      
        pip install --upgrade pip      
        pip install -r requirements.txt  

4. (Opcional) Docker      
        docker build -t carplates-detector .      
        docker run --gpus all -v $(pwd)/data:/app/data carplates-detector  

---

## 🚀 Uso Rápido

### 👟 Entrenamiento

    python src/train.py \
      --config configs/train_config.yml \
      --epochs 50 \
      --batch-size 16 \
      --lr 1e-4  

- `--config`: ruta al archivo YAML con rutas y parámetros  
- `--epochs`: número de épocas  
- `--batch-size`: tamaño de lote  
- `--lr`: tasa de aprendizaje  

### 🔍 Detección en Imagen

    python src/detect.py \
      --weights models/best_model.pt \
      --source data/images/test/car1.jpg \
      --output results/  

- `--weights`: pesos del modelo  
- `--source`: ruta a imagen o carpeta de imágenes  
- `--output`: carpeta donde se guardan los resultados  

### 🎥 Detección en Vídeo

    python src/detect.py \
      --weights models/best_model.pt \
      --source data/videos/highway.mp4 \
      --view-img \
      --output results/videos/  

---

## 📊 Experimentos & Resultados

| Experimento       | Épocas | Batch | LR     | Precisión | Recall | F1-score |
|-------------------|-------:|------:|-------:|----------:|-------:|---------:|
| Baseline ResNet34 |     10 |    32 | 1e-3   |     92.3% |  89.5% |    90.9% |
| Augmentation V2   |     20 |    16 | 5e-4   |     94.1% |  91.2% |    92.6% |
| OCR+CTC head      |     30 |    16 | 1e-4   |     95.4% |  93.0% |    94.2% |  

Gráficas de pérdida y accuracy están en `results/curves/`.

---

## 📏 Métricas de Evaluación

- **Precisión (Precision)**: TP / (TP + FP)  
- **Recall**: TP / (TP + FN)  
- **F1-score**: 2 · (Precision·Recall) / (Precision+Recall)  
- **mAP@0.5**: Mean Average Precision con IoU ≥ 0.5  

---

## 📚 Dataset

- **Origen**: Capturas propias de matrículas peruanas  
- **Formato**:  
  - Imágenes `.jpg` o `.png`  
  - Anotaciones YOLO `.txt` (x_center, y_center, w, h)  
- **Split**: 70% train / 15% val / 15% test  
- **Aumentación**: rotación ±15°, flip horizontal, ajuste de brillo, ruido gaussiano  

---

## 🔄 Personalización

- Cambia hiperparámetros en `configs/train_config.yml`  
- Sustituye el backbone en `src/train.py`  
- Ajusta técnicas de aumentación en `src/data_loader.py`  
- Modifica umbrales de confianza y NMS en `src/detect.py`  

---

## 🤝 Contribuye

¡Todas las ⭐️ son bienvenidas!  
1. **Fork** 🍴 este repositorio  
2. Crea una **branch**: `git checkout -b feature/nueva-funcion`  
3. Realiza tus **commits**: `git commit -am 'Agrega X'`  
4. **Push** a tu branch: `git push origin feature/nueva-funcion`  
5. Abre un **Pull Request** 🚀  

Consulta `CONTRIBUTING.md` para más detalles.

---

## 🧹 Mantenimiento

- Actualiza dependencias con `pip install --upgrade -r requirements.txt`  
- Corre tests (por implementar) con `pytest tests/`  
- Integra GitHub Actions para CI/CD y docker lint  

---
## 🤝 Contribuir
1. Fork 🍴 → Branch `feature/X` → Commit → Push → PR 🚀
2. Añade tests y ejemplos en `notebooks/`
3. Sigue pautas de estilo en `CONTRIBUTING.md`

---
## 📄 Licencia

Este proyecto está licenciado bajo **MIT License**. Consulta el archivo `LICENSE` para más información.





---

🎉 ¡Gracias por visitar **IA_NN_CarPlatesY11**!  
Si te gustó el proyecto, dale una ⭐️ y comparte tus mejoras 🚀
