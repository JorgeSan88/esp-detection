"""
Script para exportar un modelo ya entrenado a ESP32-S3
Solo ejecuta los pasos de: Exportar ONNX → Cuantizar → Generar proyecto C++

USO RÁPIDO:
    python export_trained_model.py

Este script toma el modelo de runs/detect/train4/weights/best.pt
y genera el proyecto listo para ESP32-S3
"""

import os
import argparse
from pathlib import Path
import shutil
import subprocess
import re

from deploy.export import Export
from deploy.quantize import quant_espdet


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


def export_trained_model():
    """
    Exporta un modelo ya entrenado directamente a ESP32-S3
    """
    
    # ========== CONFIGURACIÓN ==========
    # Modelo ya entrenado
    MODEL_PATH = "runs/detect/train4/weights/best.pt"
    
    # Configuración de exportación
    CLASS_NAME = "wheelchair"
    IMG_SIZE = [224, 224]  # Debe coincidir con el entrenamiento
    TARGET = "esp32s3"
    
    # Paths
    ESPDL_MODEL_NAME = "wheelchair_detector.espdl"
    CALIB_DATA = "datasets/wheelchair_data/images/val"
    TEST_IMAGE = "datasets/wheelchair_data/images/val/1013_jpg.rf.6c51efff88fafdf7ff3ab95165537299.jpg"
    
    # ===================================
    
    print("\n" + "="*60)
    print("🔧 EXPORTACIÓN DE MODELO YA ENTRENADO")
    print("="*60)
    print(f"Modelo origen:       {MODEL_PATH}")
    print(f"Clase:               {CLASS_NAME}")
    print(f"Tamaño de imagen:    {IMG_SIZE[0]}x{IMG_SIZE[1]}")
    print(f"Target ESP32:        {TARGET.upper()}")
    print(f"Modelo de salida:    {ESPDL_MODEL_NAME}")
    print("="*60 + "\n")
    
    # Verificar que existe el modelo
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No se encontró el modelo en: {MODEL_PATH}")
    
    # Verificar carpeta de calibración
    if not os.path.exists(CALIB_DATA):
        raise FileNotFoundError(f"No se encontró la carpeta de calibración en: {CALIB_DATA}")
    
    # Verificar imagen de prueba
    if not os.path.exists(TEST_IMAGE):
        print(f"⚠️  ADVERTENCIA: No se encontró la imagen de prueba: {TEST_IMAGE}")
        val_images = list(Path(CALIB_DATA).glob("*.jpg"))
        if val_images:
            TEST_IMAGE = str(val_images[0])
            print(f"   Usando: {TEST_IMAGE}")
        else:
            raise FileNotFoundError("No se encontraron imágenes .jpg en la carpeta de validación")
    
    # PASO 1: EXPORTAR A ONNX
    print("\n" + "📦 " + "="*55)
    print("PASO 1/3: EXPORTAR MODELO A FORMATO ONNX")
    print("="*60)
    
    Export(MODEL_PATH, IMG_SIZE)
    onnx_path = MODEL_PATH.replace(".pt", ".onnx")
    
    if os.path.exists(onnx_path):
        print(f"✅ Modelo ONNX guardado en: {onnx_path}")
    else:
        raise FileNotFoundError(f"No se pudo generar el archivo ONNX: {onnx_path}")
    
    # PASO 2: CUANTIZACIÓN
    print("\n" + "⚙️  " + "="*55)
    print("PASO 2/3: CUANTIZAR MODELO A FORMATO ESP-DL")
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
    
    if os.path.exists(ESPDL_MODEL_NAME):
        print(f"✅ Modelo cuantizado guardado en: {ESPDL_MODEL_NAME}")
    else:
        raise FileNotFoundError(f"No se pudo generar el archivo ESPDL: {ESPDL_MODEL_NAME}")
    
    # PASO 3: GENERAR PROYECTO C++ PARA ESP32
    print("\n" + "🔨 " + "="*55)
    print("PASO 3/3: GENERAR PROYECTO C++ PARA ESP32")
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
    h, w = IMG_SIZE
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
    print("¡EXPORTACIÓN COMPLETADA EXITOSAMENTE!")
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
    print(f"  • Modelo PyTorch:    {MODEL_PATH}")
    print(f"  • Modelo ONNX:       {onnx_path}")
    print(f"  • Modelo ESP-DL:     {ESPDL_MODEL_NAME}")
    print(f"  • Proyecto C++:      {custom_example_path}")
    print("\n✨ Tu modelo está listo para ser desplegado en ESP32-S3 ✨\n")


if __name__ == '__main__':
    try:
        export_trained_model()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\nPor favor, revisa la configuración y vuelve a intentar.")
        raise
