"""
Script rápido para verificar si PyTorch detecta tu GPU

Ejecuta este script antes de entrenar para verificar que todo está configurado correctamente:
    python check_gpu.py
"""

import torch

print("\n" + "="*60)
print("🖥️  VERIFICACIÓN DE GPU PARA ENTRENAMIENTO")
print("="*60 + "\n")

# Verificar PyTorch
print(f"📦 PyTorch version: {torch.__version__}")

# Verificar CUDA
cuda_available = torch.cuda.is_available()

if cuda_available:
    print(f"\n✅ CUDA está disponible")
    print(f"✅ CUDA version: {torch.version.cuda}")
    print(f"✅ cuDNN version: {torch.backends.cudnn.version()}")
    print(f"✅ cuDNN habilitado: {torch.backends.cudnn.enabled}")
    
    # Información de GPUs
    gpu_count = torch.cuda.device_count()
    print(f"\n🎮 Número de GPUs detectadas: {gpu_count}")
    
    for i in range(gpu_count):
        print(f"\n   GPU {i}:")
        print(f"   └─ Nombre: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"   └─ Memoria total: {props.total_memory / 1024**3:.2f} GB")
        print(f"   └─ Compute capability: {props.major}.{props.minor}")
    
    # Probar crear un tensor en GPU
    print(f"\n🧪 Probando crear tensor en GPU...")
    try:
        test_tensor = torch.randn(100, 100).cuda()
        print(f"✅ ¡Tensor creado exitosamente en GPU!")
        print(f"   Device del tensor: {test_tensor.device}")
        del test_tensor  # Liberar memoria
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"❌ Error al crear tensor: {e}")
    
    print(f"\n" + "="*60)
    print("✅ TU SISTEMA ESTÁ LISTO PARA ENTRENAR EN GPU")
    print("="*60)
    print("\n💡 El entrenamiento usará GPU automáticamente")
    print("💡 Durante el entrenamiento verás mensajes como:")
    print("   - 'Transferring model to cuda:0'")
    print("   - Uso de VRAM en las métricas")
    print("\n")
    
else:
    print(f"\n⚠️  CUDA NO está disponible")
    print(f"\n❌ PyTorch no detecta una GPU compatible")
    print(f"\n📋 Posibles causas:")
    print(f"   1. No tienes una GPU NVIDIA")
    print(f"   2. Los drivers de NVIDIA no están instalados")
    print(f"   3. PyTorch fue instalado sin soporte CUDA")
    print(f"\n🔧 Para instalar PyTorch con CUDA:")
    print(f"   pip uninstall torch torchvision")
    print(f"   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    print(f"\n⚠️  El entrenamiento usará CPU (será MUCHO más lento)")
    print(f"\n")

# Información adicional sobre el sistema
print("="*60)
print("ℹ️  INFORMACIÓN DEL SISTEMA")
print("="*60)
import platform
print(f"Sistema operativo: {platform.system()} {platform.release()}")
print(f"Python version: {platform.python_version()}")
print(f"Arquitectura: {platform.machine()}")
print("="*60 + "\n")
