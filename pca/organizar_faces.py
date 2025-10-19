import os
import shutil
import zipfile

# --- 🔧 CONFIGURA AQUÍ LAS RUTAS ---
zip_path = r"C:\Users\alems\Downloads\ADA\pca\att_faces.zip"  # ✅ Ruta correcta del ZIP
base_dir = r"C:\Users\alems\Downloads\ADA\pca"
dst_gallery = os.path.join(base_dir, "Gallery", "Gallery")
dst_probe = os.path.join(base_dir, "Probe", "Probe")

# --- 🧹 LIMPIAR CARPETAS VIEJAS ---
for path in [dst_gallery, dst_probe]:
    if os.path.exists(os.path.dirname(path)):
        shutil.rmtree(os.path.dirname(path))
        print(f"🧹 Eliminada carpeta previa: {os.path.dirname(path)}")

# --- 📦 EXTRAER ZIP (si existe) ---
if zip_path.endswith(".zip"):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(base_dir)
        print(f"✅ Archivo ZIP extraído en: {base_dir}")

# --- 📂 RUTA DE CARPETAS DE SUJETOS (ej. s1, s2, ..., s40) ---
src = base_dir

if not os.path.exists(src):
    raise FileNotFoundError(f"No se encontró la carpeta de imágenes en: {src}")

# --- 🚀 CREAR NUEVAS CARPETAS ---
os.makedirs(dst_gallery, exist_ok=True)
os.makedirs(dst_probe, exist_ok=True)

# --- 🪄 DIVIDIR Y MOVER IMÁGENES ---
for folder in os.listdir(src):
    if folder.startswith("s"):
        os.makedirs(os.path.join(dst_gallery, folder), exist_ok=True)
        os.makedirs(os.path.join(dst_probe, folder), exist_ok=True)

        images = sorted(os.listdir(os.path.join(src, folder)), key=lambda x: int(os.path.splitext(x)[0]))
        for i, img in enumerate(images):
            src_path = os.path.join(src, folder, img)
            if i < 5:
                shutil.move(src_path, os.path.join(dst_gallery, folder, img))
            else:
                shutil.move(src_path, os.path.join(dst_probe, folder, img))

print("✅ División completa:")
print(f"→ Gallery: {dst_gallery}")
print(f"→ Probe:   {dst_probe}")
