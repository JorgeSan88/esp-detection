# Detección de Sillas de Ruedas en ESP32-S3

Guía completa para entrenar, exportar y desplegar un modelo de detección de sillas de ruedas en un chip ESP32-S3 usando el framework **ESP-Detection** (basado en YOLOv11 de Ultralytics).

---

## Tabla de Contenidos

1. [Requisitos del sistema](#1-requisitos-del-sistema)
2. [Instalación del entorno](#2-instalación-del-entorno)
3. [Estructura del proyecto](#3-estructura-del-proyecto)
4. [Dataset](#4-dataset)
5. [Entrenamiento](#5-entrenamiento)
6. [Dónde se guarda el modelo entrenado](#6-dónde-se-guarda-el-modelo-entrenado)
7. [Exportar el modelo a ESP32](#7-exportar-el-modelo-a-esp32)
8. [Descripción del ejemplo generado para ESP32](#8-descripción-del-ejemplo-generado-para-esp32)
9. [Flashear el ejemplo en la ESP32-S3](#9-flashear-el-ejemplo-en-la-esp32-s3)
10. [Qué hace el ejemplo al ejecutarse](#10-qué-hace-el-ejemplo-al-ejecutarse)
11. [Cómo agregar una cámara para detección en tiempo real](#11-cómo-agregar-una-cámara-para-detección-en-tiempo-real)
12. [Scripts creados en este proyecto](#12-scripts-creados-en-este-proyecto)
13. [Solución de problemas comunes](#13-solución-de-problemas-comunes)

---

## 1. Requisitos del sistema

### PC para entrenamiento
| Componente | Requerimiento |
|---|---|
| Sistema operativo | Windows 10/11, Ubuntu 20.04+, macOS |
| Python | 3.10 o 3.11 (recomendado 3.11) |
| RAM | Mínimo 8 GB (recomendado 16 GB) |
| GPU | NVIDIA con soporte CUDA (recomendado, el entrenamiento en CPU es muy lento) |
| Almacenamiento | ~5 GB libres para dataset, modelos y dependencias |

### GPU recomendada
El proyecto fue probado con **NVIDIA GeForce con CUDA 12.8 y PyTorch 2.10.0**.  
Sin GPU, el entrenamiento de 300 épocas puede tardar muchas horas o días.

### Para flashear en ESP32
| Componente | Requerimiento |
|---|---|
| Placa | ESP32-S3 (cualquier variante) |
| ESP-IDF | versión `release/v5.3` o superior |
| Cable | USB para flashear y monitorear |

---

## 2. Instalación del entorno

### Paso 1: Clonar el repositorio

```bash
git clone <url-del-repositorio>
cd esp-detection
```

### Paso 2: Crear entorno virtual

```bash
python -m venv .venv
```

Activar en Windows:
```bash
.venv\Scripts\activate
```

Activar en Linux/macOS:
```bash
source .venv/bin/activate
```

### Paso 3: Instalar PyTorch con CUDA (si tienes GPU NVIDIA)

> ⚠️ **Importante:** Instala PyTorch primero, antes del resto de dependencias.

```bash
# Para CUDA 12.8 (verifica tu versión en https://pytorch.org/get-started/locally/)
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128

# Sin GPU (CPU solamente, entrenamiento muy lento)
pip install torch torchvision
```

### Paso 4: Instalar el resto de dependencias

```bash
pip install -r requirements.txt
```

### Paso 5: Verificar que la GPU es detectada

```bash
python check_gpu.py
```

Deberías ver algo como:
```
✅ CUDA disponible: SÍ
✅ GPU actual: 0 (NVIDIA GeForce RTX XXXX)
✅ PyTorch version: 2.10.0+cu128
🚀 El entrenamiento se ejecutará en GPU
```

---

## 3. Estructura del proyecto

```
esp-detection/
│
├── cfg/
│   ├── datasets/
│   │   └── wheelchair_data.yaml     ← Configuración del dataset
│   └── models/
│       └── espdet_pico.yaml         ← Arquitectura del modelo
│
├── datasets/
│   └── wheelchair_data/
│       ├── images/
│       │   ├── train/               ← Imágenes de entrenamiento
│       │   └── val/                 ← Imágenes de validación
│       └── labels/
│           ├── train/               ← Etiquetas (formato YOLO)
│           └── val/
│
├── deploy/
│   ├── export.py                    ← Exporta .pt → .onnx
│   ├── quantize.py                  ← Cuantiza .onnx → .espdl
│   ├── espdet_example_template/     ← Template del proyecto C++ para ESP32
│   └── espdet_model_template/       ← Template del modelo C++ para ESP32
│
├── runs/
│   └── detect/
│       ├── train/                   ← Primer entrenamiento
│       ├── train2/                  ← Segundo entrenamiento
│       ├── train3/                  ← ...
│       └── train4/                  ← Último entrenamiento
│           ├── weights/
│           │   ├── best.pt          ← ⭐ MODELO FINAL (mejor época)
│           │   └── last.pt          ← Modelo de la última época
│           ├── args.yaml            ← Parámetros usados en el entrenamiento
│           └── results.csv          ← Métricas por época
│
├── train.py                         ← Script base de entrenamiento
├── espdet_run_wheelchair.py         ← Script completo: entrena + exporta
├── export_trained_model.py          ← Solo exporta (sin re-entrenar)
├── check_gpu.py                     ← Verifica disponibilidad de GPU
├── install_dependencies.py          ← Instala dependencias faltantes
├── wheelchair_detector.espdl        ← ⭐ Modelo cuantizado para ESP32
└── requirements.txt                 ← Dependencias del proyecto
```

---

## 4. Dataset

El dataset de sillas de ruedas está en `datasets/wheelchair_data/` y sigue el **formato YOLO**:

- Cada imagen tiene un archivo `.txt` correspondiente con las etiquetas
- Cada línea en el `.txt` es: `clase x_center y_center width height` (valores normalizados de 0 a 1)
- Solo existe la clase `0: wheelchair`

La configuración del dataset está en [`cfg/datasets/wheelchair_data.yaml`](cfg/datasets/wheelchair_data.yaml):

```yaml
path: datasets/wheelchair_data
train: images/train
val: images/val

names:
  0: wheelchair
```

---

## 5. Entrenamiento

### Opción A: Pipeline completo (recomendado para nuevos entrenamientos)

Este script entrena el modelo y luego exporta todo automáticamente:

```bash
python espdet_run_wheelchair.py
```

**Qué hace internamente:**
1. Verifica si hay GPU disponible
2. Entrena el modelo durante 300 épocas con el dataset `wheelchair_data`
3. Exporta a ONNX
4. Cuantiza para ESP32-S3
5. Genera el proyecto C++

**Parámetros configurables** (editar directamente en el script):

```python
CLASS_NAME   = "wheelchair"      # Nombre de la clase
IMG_SIZE     = [224, 224]        # Tamaño de imagen de entrada
EPOCHS       = 300               # Número de épocas
BATCH_SIZE   = 128               # Batch size (reducir si se queda sin VRAM)
TARGET       = "esp32s3"         # Chip objetivo
```

> 💡 **Tip de GPU:** Si durante el entrenamiento aparece `CUDA out of memory`, reduce `BATCH_SIZE` a 64 o 32.

### Opción B: Solo entrenar (sin exportar)

```bash
python train.py
```

### Monitorear el progreso del entrenamiento

Durante el entrenamiento verás en consola líneas como:
```
Epoch  10/300  GPU_mem: 2.5G  box_loss: 1.234  cls_loss: 0.567  dfl_loss: 0.891
               Class     Images  Instances      P      R  mAP50  mAP50-95
               all          142       312  0.654  0.589  0.621    0.412
```

- **box_loss**: Error en la localización del bounding box (debe bajar)
- **cls_loss**: Error en la clasificación (debe bajar)
- **mAP50**: Precisión del modelo (debe subir, máximo 1.0)
- **GPU_mem**: Memoria de GPU usada

---

## 6. Dónde se guarda el modelo entrenado

Cada vez que ejecutas el entrenamiento se crea una nueva carpeta en `runs/detect/`:

```
runs/detect/
├── train/          ← 1er entrenamiento
├── train2/         ← 2do entrenamiento
├── train3/         ← 3er entrenamiento
└── train4/         ← 4to entrenamiento (el más reciente)
    ├── weights/
    │   ├── best.pt    ← ⭐ El que se usa para exportar
    │   └── last.pt    ← Última época (no necesariamente el mejor)
    ├── args.yaml      ← Todos los parámetros usados
    └── results.csv    ← Métricas de cada época
```

### ¿Cuál modelo usar?

**Siempre usar `best.pt`** — es el modelo guardado en la época donde se obtuvo el mejor `mAP50-95` en el set de validación.

`last.pt` solo sirve para continuar un entrenamiento interrumpido.

### Ver las métricas del entrenamiento

Abre `runs/detect/train4/results.csv` en Excel o ejecuta:

```bash
python -c "
import pandas as pd
df = pd.read_csv('runs/detect/train4/results.csv')
print(df[['epoch','metrics/mAP50(B)','metrics/mAP50-95(B)']].tail(10))
"
```

---

## 7. Exportar el modelo a ESP32

Si ya tienes un modelo entrenado y NO quieres volver a entrenar:

```bash
python export_trained_model.py
```

Este script realiza tres pasos:

### Paso 1/3 — Convertir `.pt` → `.onnx`

- **Entrada:** `runs/detect/train4/weights/best.pt`
- **Salida:** `runs/detect/train4/weights/best.onnx`
- El formato ONNX es un estándar de intercambio de modelos de IA, compatible con muchas plataformas

### Paso 2/3 — Cuantizar `.onnx` → `.espdl`

- **Entrada:** `runs/detect/train4/weights/best.onnx`
- **Salida:** `wheelchair_detector.espdl`
- La cuantización convierte los pesos de float32 a int8 (8 bits)
- Reduce el tamaño del modelo ~4x y lo hace compatible con el hardware de la ESP32
- Usa imágenes de validación como **datos de calibración** para mantener la precisión

### Paso 3/3 — Generar proyecto C++

- **Salida:** `esp-dl/examples/wheelchair_detect/` — proyecto completo listo para compilar
- El modelo `.espdl` se copia a `esp-dl/models/wheelchair_detect/models/s3/`

### Flujo visual

```
best.pt  →  [export.py]  →  best.onnx  →  [quantize.py]  →  wheelchair_detector.espdl
                                                                        ↓
                                             esp-dl/models/wheelchair_detect/models/s3/
                                             esp-dl/examples/wheelchair_detect/
```

---

## 8. Descripción del ejemplo generado para ESP32

El proyecto C++ generado en `esp-dl/examples/wheelchair_detect/` tiene esta estructura:

```
wheelchair_detect/
├── CMakeLists.txt                  ← Configuración de build del ejemplo
├── partitions.csv                  ← Tabla de particiones de la flash
├── sdkconfig.defaults              ← Configuración por defecto del SDK
├── sdkconfig.defaults.esp32s3     ← Configuración específica para ESP32-S3
└── main/
    ├── app_main.cpp                ← ⭐ Código principal de la aplicación
    ├── CMakeLists.txt              ← Build del main
    ├── idf_component.yml           ← Dependencias del componente
    └── <imagen_de_prueba>.jpg      ← Imagen embebida en el firmware
```

### ¿Qué hace `app_main.cpp`?

```cpp
void app_main(void) {
    // 1. Carga la imagen .jpg embebida en el firmware
    auto img = dl::image::sw_decode_jpeg(jpeg_img, DL_IMAGE_PIX_TYPE_RGB888);

    // 2. Crea el detector con el modelo cuantizado
    ESPDetDetect *detect = new ESPDetDetect();

    // 3. Ejecuta la detección UNA VEZ sobre la imagen
    auto &detect_results = detect->run(img);

    // 4. Imprime los resultados en el terminal serial
    for (const auto &res : detect_results) {
        ESP_LOGI(TAG, "[category: %d, score: %f, x1: %d, y1: %d, x2: %d, y2: %d]",
                 res.category, res.score,
                 res.box[0], res.box[1], res.box[2], res.box[3]);
    }
}
```

**Lo que verás en el terminal serial:**
```
I (1234) wheelchair_detect: [category: 0, score: 0.893, x1: 45, y1: 30, x2: 180, y2: 200]
I (1235) wheelchair_detect: [category: 0, score: 0.761, x1: 220, y1: 55, x2: 310, y2: 195]
```

- `category: 0` → clase `wheelchair`
- `score: 0.893` → confianza del 89.3%
- `x1, y1, x2, y2` → coordenadas del bounding box en píxeles

> ⚠️ **Importante:** El ejemplo actual analiza **una sola imagen estática** que está guardada dentro del firmware. **NO usa cámara** y la detección ocurre una sola vez al arrancar.

---

## 9. Flashear el ejemplo en la ESP32-S3

### Requisitos previos

1. **Instalar ESP-IDF** (versión `release/v5.3` o superior):
   - Guía oficial: https://idf.espressif.com/
   - En Windows se recomienda usar [ESP-IDF Tools Installer](https://dl.espressif.com/dl/esp-idf/)

2. **Conectar la ESP32-S3** por USB a tu PC

### Comandos para compilar y flashear

```bash
# 1. Entrar a la carpeta del ejemplo
cd esp-dl/examples/wheelchair_detect

# 2. Configurar el target
idf.py set-target esp32s3

# 3. (Opcional) Abrir menú de configuración
idf.py menuconfig

# 4. Compilar
idf.py build

# 5. Flashear y abrir monitor serial
idf.py flash monitor
```

### Salida esperada en el monitor serial

```
I (500) main_task: Started on CPU0
I (510) main_task: Calling app_main()
I (1200) wheelchair_detect: [category: 0, score: 0.87, x1: 23, y1: 10, x2: 156, y2: 180]
I (1201) main_task: Returned from app_main()
```

Para salir del monitor serial: `Ctrl + ]`

---

## 10. Qué hace el ejemplo al ejecutarse

```
ESP32-S3 arranca
      │
      ▼
Carga la imagen de prueba (.jpg embebida en flash)
      │
      ▼
Carga el modelo wheelchair_detector.espdl desde la flash
      │
      ▼
Ejecuta la detección sobre la imagen (una sola vez, ~126ms en ESP32-S3)
      │
      ▼
Imprime los resultados por el puerto serial (UART)
      │
      ▼
Termina (la ESP32-S3 queda en espera)
```

**Limitaciones del ejemplo actual:**
- Solo procesa una imagen (la que está embebida en el firmware)
- Para cambiar la imagen hay que recompilar y re-flashear
- No tiene visualización (no dibuja bounding boxes)
- No es continuo (no hace detección en loop)

---

## 11. Cómo agregar una cámara para detección en tiempo real

Para hacer detección continua con cámara necesitas:

### Hardware recomendado

| Placa | Cámara incluida | Notas |
|---|---|---|
| **ESP32-S3-EYE** | ✅ OV2640 integrada | Opción más fácil |
| **ESP32-S3-DevKitC** | ❌ | Conectar módulo OV2640 o OV5640 |
| **Seeed XIAO ESP32S3 Sense** | ✅ OV2640 integrada | Compacta |

### Modificar `app_main.cpp`

Reemplaza el contenido del `app_main.cpp` generado con algo como esto:

```cpp
#include "espdet_detect.hpp"
#include "esp_camera.h"          // Biblioteca de cámara de ESP-IDF
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

const char *TAG = "wheelchair_detect";

// Configuración de pines para ESP32-S3-EYE (ajusta según tu hardware)
#define CAM_PIN_PWDN    -1
#define CAM_PIN_RESET   -1
#define CAM_PIN_XCLK    15
#define CAM_PIN_SIOD    4
#define CAM_PIN_SIOC    5
#define CAM_PIN_D7      16
#define CAM_PIN_D6      17
#define CAM_PIN_D5      18
#define CAM_PIN_D4      12
#define CAM_PIN_D3      10
#define CAM_PIN_D2      8
#define CAM_PIN_D1      9
#define CAM_PIN_D0      11
#define CAM_PIN_VSYNC   6
#define CAM_PIN_HREF    7
#define CAM_PIN_PCLK    13

extern "C" void app_main(void)
{
    // Inicializar cámara
    camera_config_t config = {
        .pin_pwdn     = CAM_PIN_PWDN,
        .pin_reset    = CAM_PIN_RESET,
        .pin_xclk     = CAM_PIN_XCLK,
        .pin_sscb_sda = CAM_PIN_SIOD,
        .pin_sscb_scl = CAM_PIN_SIOC,
        .pin_d7 = CAM_PIN_D7, .pin_d6 = CAM_PIN_D6,
        .pin_d5 = CAM_PIN_D5, .pin_d4 = CAM_PIN_D4,
        .pin_d3 = CAM_PIN_D3, .pin_d2 = CAM_PIN_D2,
        .pin_d1 = CAM_PIN_D1, .pin_d0 = CAM_PIN_D0,
        .pin_vsync = CAM_PIN_VSYNC,
        .pin_href  = CAM_PIN_HREF,
        .pin_pclk  = CAM_PIN_PCLK,
        .xclk_freq_hz = 20000000,
        .ledc_timer   = LEDC_TIMER_0,
        .ledc_channel = LEDC_CHANNEL_0,
        .pixel_format = PIXFORMAT_RGB888,   // El modelo espera RGB
        .frame_size   = FRAMESIZE_224X224,  // Tamaño del entrenamiento: 224x224
        .fb_count     = 2,
    };
    esp_camera_init(&config);

    // Crear detector (carga el modelo una vez)
    ESPDetDetect *detect = new ESPDetDetect();
    ESP_LOGI(TAG, "Modelo cargado. Iniciando detección en tiempo real...");

    // Bucle de detección continua
    while (true) {
        // Capturar frame
        camera_fb_t *fb = esp_camera_fb_get();
        if (!fb) {
            ESP_LOGE(TAG, "Error capturando imagen");
            continue;
        }

        // Construir estructura de imagen para el modelo
        dl::image::img_t img = {
            .data   = fb->buf,
            .width  = (int)fb->width,
            .height = (int)fb->height,
            .pix_type = dl::image::DL_IMAGE_PIX_TYPE_RGB888
        };

        // Ejecutar detección
        auto &results = detect->run(img);

        // Imprimir resultados
        if (results.empty()) {
            ESP_LOGI(TAG, "No se detectaron sillas de ruedas");
        } else {
            for (const auto &res : results) {
                ESP_LOGI(TAG,
                    "Silla detectada! score=%.2f | bbox=[%d,%d,%d,%d]",
                    res.score,
                    res.box[0], res.box[1], res.box[2], res.box[3]);
            }
        }

        // Liberar el frame
        esp_camera_fb_return(fb);

        // Pequeña pausa para no saturar
        vTaskDelay(pdMS_TO_TICKS(50));  // ~20 FPS máximo
    }
}
```

### Dependencias adicionales en `idf_component.yml`

Agregar al `idf_component.yml` del ejemplo:
```yaml
dependencies:
  esp32-camera:
    git: https://github.com/espressif/esp32-camera.git
```

> 💡 **Nota de rendimiento:** En ESP32-S3 se espera aproximadamente **7 FPS** con imagen de 224x224 según los benchmarks del proyecto.

---

## 12. Scripts creados en este proyecto

| Script | Descripción | Cuándo usarlo |
|---|---|---|
| `espdet_run_wheelchair.py` | Pipeline completo: entrena + exporta | Para un entrenamiento nuevo de principio a fin |
| `export_trained_model.py` | Solo exporta (sin entrenar) | Cuando ya tienes `best.pt` y quieres generar el `.espdl` |
| `train.py` | Solo entrena | Si quieres ajustar el entrenamiento manualmente |
| `check_gpu.py` | Verifica GPU | Antes de entrenar por primera vez |
| `install_dependencies.py` | Instala dependencias faltantes | Si fallan imports al ejecutar |
| `deploy/export.py` | Convierte `.pt` → `.onnx` | Usado internamente por los scripts de exportación |
| `deploy/quantize.py` | Convierte `.onnx` → `.espdl` | Usado internamente por los scripts de exportación |

---

## 13. Solución de problemas comunes

### ❌ `ModuleNotFoundError: No module named 'onnxscript'`

```bash
.venv\Scripts\python.exe -m pip install onnxscript
```

### ❌ `CUDA out of memory` durante el entrenamiento

Reducir el batch size en `espdet_run_wheelchair.py`:
```python
BATCH_SIZE = 64   # O incluso 32
```

### ❌ Error al instalar `onnxruntime-gpu` (acceso denegado)

Instalar la versión CPU (suficiente para cuantización):
```bash
.venv\Scripts\python.exe -m pip install onnxruntime
```

### ❌ `onnxsim` no se puede instalar

No es crítico. El pipeline funciona sin él. Solo instálalo si tienes cmake:
```bash
# Primero instalar cmake desde https://cmake.org/download/
pip install onnxsim
```

### ❌ `FileNotFoundError: No se encontró el modelo`

Verifica que el entrenamiento se completó:
```bash
# Verificar que existe el archivo
ls runs/detect/train4/weights/best.pt
```

### ❌ Error al compilar con ESP-IDF

Asegúrate de usar ESP-IDF `release/v5.3` o superior:
```bash
idf.py --version
```

---

## Resumen del flujo completo

```
DATASET                    ENTRENAMIENTO              EXPORTACIÓN           ESP32-S3
wheelchair_data/    →    espdet_run_wheelchair.py  →  best.onnx        →   esp-dl/examples/
  images/train/            (300 épocas, GPU)           ↓                   wheelchair_detect/
  images/val/                    ↓               wheelchair_detector.espdl      ↓
  labels/                    best.pt                    (cuantizado 8-bit)   idf.py flash
                          runs/detect/train4/                                    ↓
                                                                           Serial output:
                                                                           category:0 score:0.89
                                                                           x1:45 y1:30 ...
```
