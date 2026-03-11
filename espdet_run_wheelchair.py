"""
Script configurado para entrenar y desplegar modelo de detección de sillas de ruedas en ESP32-S3

Este script ejecuta todo el flujo completo:
1. Entrena el modelo con el dataset de wheelchair_data
2. Exporta a formato ONNX
3. Cuantiza a formato ESP-DL (.espdl)
4. Genera el proyecto C++ listo para ESP32-S3

Uso:
    python espdet_run_wheelchair.py

Para personalizar parámetros, edita las variables en la sección CONFIGURACIÓN
"""

import os
import argparse
from pathlib import Path
import shutil
import subprocess
import re
import torch

from train import Train
from deploy.export import Export
from deploy.quantize import quant_espdet


def check_gpu_availability():
    """Verifica y muestra información sobre la disponibilidad de GPU"""
    print("\n" + "🖥️  " + "="*55)
    print("VERIFICACIÓN DE DISPOSITIVO")
    print("="*60)
    
    cuda_available = torch.cuda.is_available()
    
    if cuda_available:
        gpu_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        
        print(f"✅ CUDA disponible: SÍ")
        print(f"✅ GPUs detectadas: {gpu_count}")
        print(f"✅ GPU actual: {current_device} ({gpu_name})")
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA version: {torch.version.cuda}")
        print("\n🚀 El entrenamiento se ejecutará en GPU")
        return "0"  # Usar GPU 0
    else:
        print(f"⚠️  CUDA disponible: NO")
        print(f"⚠️  PyTorch version: {torch.__version__}")
        print(f"\n⚠️  El entrenamiento se ejecutará en CPU (será más lento)")
        print(f"   Para usar GPU, asegúrate de tener:")
        print(f"   1. Una GPU NVIDIA compatible")
        print(f"   2. PyTorch instalado con soporte CUDA")
        print(f"      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        return "cpu"
    

def rename_project(root_dir: Path, replacements: dict):
    """Renombra referencias en archivos del template"""
    target_extensions = {".cpp", ".hpp", ".txt", ".yml"}
    for file in root_dir.rglob("*"):
        if file.suffix in target_extensions or file.name == "Kconfig":
            try:
                content = file.read_text(encoding="utf-8")
                original_content = content
                # find add_custom_command lines
                skipped_lines = re.findall(r".*add_custom_command.*", content)
                placeholders = {}
                for i, line in enumerate(skipped_lines):
                    placeholder = f"__PLACEHOLDER_{i}__"
                    content = content.replace(line, placeholder)
                    placeholders[placeholder] = line
                for old, new in replacements.items():
                    content = content.replace(old, new)
                for placeholder, line in placeholders.items():
                    content = content.replace(placeholder, line)
                if content != original_content:
                    file.write_text(content, encoding="utf-8")
                    print(f"Replaced in: {file}")

            except Exception as e:
                print(f"Failed to process {file}: {e}")


def run_wheelchair_detection():
    """
    Proceso completo para entrenar y desplegar modelo de detección de sillas de ruedas
    """
    
    # Verificar GPU primero
    device = check_gpu_availability()
    
    # ========== CONFIGURACIÓN ==========
    # Puedes modificar estos valores según tus necesidades
    
    CLASS_NAME = "wheelchair"              # Nombre de la clase a detectar
    DATASET = "cfg/datasets/wheelchair_data.yaml"  # Path al dataset YAML
    IMG_SIZE = [224, 224]                  # Tamaño de entrada [altura, ancho]
    TARGET = "esp32s3"                     # Chip ESP32 objetivo: "esp32s3" o "esp32p4"
    EPOCHS = 300                           # Número de épocas de entrenamiento
    BATCH_SIZE = 128                       # Tamaño de batch
    
    # Paths de salida
    ESPDL_MODEL_NAME = "wheelchair_detector.espdl"  # Nombre del modelo cuantizado
    CALIB_DATA = "datasets/wheelchair_data/images/val"  # Imágenes para calibración
    TEST_IMAGE = "datasets/wheelchair_data/images/val/1013_jpg.rf.6c51efff88fafdf7ff3ab95165537299.jpg"  # Imagen de prueba
    
    # Usar pretrained (opcional)
    PRETRAINED_PATH = None  # Cambiar a path de .pt si quieres usar pesos preentrenados
    
    # ===================================
    
    print("\n" + "="*60)
    print("🔧 CONFIGURACIÓN DEL ENTRENAMIENTO")
    print("="*60)
    print(f"Clase a detectar:    {CLASS_NAME}")
    print(f"Dataset:             {DATASET}")
    print(f"Tamaño de imagen:    {IMG_SIZE[0]}x{IMG_SIZE[1]}")
    print(f"Target ESP32:        {TARGET.upper()}")
    print(f"Épocas:              {EPOCHS}")
    print(f"Batch size:          {BATCH_SIZE}")
    print(f"Device:              {device.upper() if device == 'cpu' else f'GPU {device}'}")
    print(f"Modelo de salida:    {ESPDL_MODEL_NAME}")
    print("="*60 + "\n")
    
    # Verificar que existe el dataset
    if not os.path.exists(DATASET):
        raise FileNotFoundError(f"No se encontró el dataset en: {DATASET}")
    
    # Verificar que existe la carpeta de calibración
    if not os.path.exists(CALIB_DATA):
        raise FileNotFoundError(f"No se encontró la carpeta de calibración en: {CALIB_DATA}")
    
    # Verificar que existe la imagen de prueba
    if not os.path.exists(TEST_IMAGE):
        print(f"⚠️  ADVERTENCIA: No se encontró la imagen de prueba: {TEST_IMAGE}")
        print("   Se usará una imagen de la carpeta de validación")
        # Buscar cualquier imagen .jpg en val
        val_images = list(Path(CALIB_DATA).glob("*.jpg"))
        if val_images:
            TEST_IMAGE = str(val_images[0])
            print(f"   Usando: {TEST_IMAGE}")
        else:
            raise FileNotFoundError("No se encontraron imágenes .jpg en la carpeta de validación")
    
    # PASO 1: ENTRENAMIENTO
    print("\n" + "🚀 " + "="*55)
    print("PASO 1: ENTRENAMIENTO DEL MODELO")
    print("="*60)
    
    h, w = IMG_SIZE
    if h != w:
        print("📐 Usando estrategia rect=True (imagen no cuadrada)")
        results = Train(
            pretrained_path=PRETRAINED_PATH,
            dataset=DATASET,
            imgsz=IMG_SIZE,
            epochs=EPOCHS,
            batch=BATCH_SIZE,
            device=device,  # Usar device detectado
            rect=True
        )
    else:
        print("📐 Usando estrategia estándar (imagen cuadrada)")
        results = Train(
            pretrained_path=PRETRAINED_PATH,
            dataset=DATASET,
            imgsz=IMG_SIZE,
            epochs=EPOCHS,
            batch=BATCH_SIZE,
            device=device  # Usar device detectado
        )
    
    # Obtener path del mejor modelo
    model_path = os.path.join(str(results.save_dir), "weights/best.pt")
    print(f"\n✅ Modelo entrenado guardado en: {model_path}")
    
    # PASO 2: EXPORTAR A ONNX
    print("\n" + "📦 " + "="*55)
    print("PASO 2: EXPORTAR MODELO A FORMATO ONNX")
    print("="*60)
    
    Export(model_path, IMG_SIZE)
    onnx_path = model_path.replace(".pt", ".onnx")
    print(f"✅ Modelo ONNX guardado en: {onnx_path}")
    
    # PASO 3: CUANTIZACIÓN
    print("\n" + "⚙️  " + "="*55)
    print("PASO 3: CUANTIZAR MODELO A FORMATO ESP-DL")
    print("="*60)
    print(f"Usando imágenes de calibración de: {CALIB_DATA}")
    
    quant_espdet(
        onnx_path=onnx_path,
        target=TARGET,
        num_of_bits=8,
        device='cpu',
        batchsz=32,
        imgsz=IMG_SIZE,
        calib_dir=CALIB_DATA,
        espdl_model_path=ESPDL_MODEL_NAME,
    )
    print(f"✅ Modelo cuantizado guardado en: {ESPDL_MODEL_NAME}")
    
    # PASO 4: GENERAR PROYECTO C++ PARA ESP32
    print("\n" + "🔨 " + "="*55)
    print("PASO 4: GENERAR PROYECTO C++ PARA ESP32")
    print("="*60)
    
    # Descargar esp-dl si no existe
    esp_dl_url = "https://github.com/espressif/esp-dl.git"
    esp_dl_path = "esp-dl"
    
    if not os.path.exists(esp_dl_path):
        print(f"Clonando repositorio esp-dl desde {esp_dl_url}...")
        subprocess.run(["git", "clone", esp_dl_url, esp_dl_path])
    else:
        print(f"Repositorio esp-dl ya existe en {esp_dl_path}")
    
    # Crear carpetas del proyecto
    examples_path = os.path.join(esp_dl_path, "examples")
    models_path = os.path.join(esp_dl_path, "models")
    custom_example_path = os.path.join(examples_path, CLASS_NAME + "_detect")
    custom_model_path = os.path.join(models_path, CLASS_NAME + "_detect")
    
    os.makedirs(custom_example_path, exist_ok=True)
    os.makedirs(custom_model_path, exist_ok=True)
    
    # Copiar templates
    shutil.copytree("deploy/espdet_model_template", custom_model_path, dirs_exist_ok=True)
    shutil.copytree("deploy/espdet_example_template", custom_example_path, dirs_exist_ok=True)
    
    # Preparar reemplazos
    test_img_name = os.path.basename(TEST_IMAGE)
    replacements = {
        "custom": CLASS_NAME,
        "CUSTOM": CLASS_NAME.upper(),
        "imgH": str(h),
        "imgW": str(w),
        "espdet.jpg": test_img_name,
        "espdet_jpg": os.path.splitext(test_img_name)[0] + "_jpg",
    }
    
    # Aplicar reemplazos
    rename_project(Path(custom_example_path), replacements)
    rename_project(Path(custom_model_path), replacements)
    
    # Copiar modelo cuantizado
    espdl_model_path = os.path.join(custom_model_path, "models/p4") if TARGET == "esp32p4" else os.path.join(custom_model_path, "models/s3")
    shutil.copy(ESPDL_MODEL_NAME, espdl_model_path)
    print(f"✅ Modelo copiado a: {espdl_model_path}")
    
    # Copiar imagen de prueba
    shutil.copy(TEST_IMAGE, os.path.join(custom_example_path, "main"))
    print(f"✅ Imagen de prueba copiada")
    
    # RESULTADO FINAL
    print("\n" + "🎉 " + "="*55)
    print("¡PROCESO COMPLETADO EXITOSAMENTE!")
    print("="*60)
    print(f"\n📁 Proyecto generado en: {custom_example_path}")
    print("\n📋 PASOS PARA FLASHEAR EN ESP32-S3:")
    print("="*60)
    print(f"1. cd {custom_example_path}")
    print(f"2. idf.py set-target {TARGET}")
    print("3. idf.py build")
    print("4. idf.py flash monitor")
    print("="*60)
    print("\n📊 ARCHIVOS GENERADOS:")
    print(f"  • Modelo PyTorch:    {model_path}")
    print(f"  • Modelo ONNX:       {onnx_path}")
    print(f"  • Modelo ESP-DL:     {ESPDL_MODEL_NAME}")
    print(f"  • Proyecto C++:      {custom_example_path}")
    print("\n✨ Tu modelo está listo para ser desplegado en ESP32-S3 ✨\n")


if __name__ == '__main__':
    try:
        run_wheelchair_detection()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\nPor favor, revisa la configuración y vuelve a intentar.")
        raise
