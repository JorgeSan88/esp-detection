"""
Script para instalar todas las dependencias necesarias para exportación ONNX

Ejecuta este script ANTES de entrenar:
    python install_dependencies.py
"""

import subprocess
import sys

def install_package(package_name, display_name=None):
    """Instala un paquete con pip"""
    display_name = display_name or package_name
    print(f"\n{'='*60}")
    print(f"Instalando {display_name}...")
    print('='*60)
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name, "--upgrade"])
        print(f"✅ {display_name} instalado correctamente")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando {display_name}: {e}")
        return False

def main():
    print("\n" + "="*60)
    print("🔧 INSTALACIÓN DE DEPENDENCIAS PARA ESP-DETECTION")
    print("="*60)
    
    # Lista de paquetes necesarios
    packages = [
        ("onnx>=1.14.0", "ONNX"),
        ("onnxscript", "ONNX Script"),
        ("onnxsim", "ONNX Simplifier"),
        ("onnxruntime", "ONNX Runtime (CPU)"),
    ]
    
    success_count = 0
    failed_packages = []
    
    for package, name in packages:
        if install_package(package, name):
            success_count += 1
        else:
            failed_packages.append(name)
    
    # Resumen
    print("\n" + "="*60)
    print("📊 RESUMEN DE INSTALACIÓN")
    print("="*60)
    print(f"✅ Paquetes instalados: {success_count}/{len(packages)}")
    
    if failed_packages:
        print(f"\n❌ Paquetes que fallaron:")
        for pkg in failed_packages:
            print(f"   - {pkg}")
        print("\n⚠️ Por favor, instala manualmente los paquetes que fallaron")
        print("   Cierra todos los terminales Python y vuelve a ejecutar este script")
        return False
    else:
        print("\n✅ ¡Todas las dependencias instaladas correctamente!")
        print("\n💡 Ahora puedes ejecutar:")
        print("   python espdet_run_wheelchair.py")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
