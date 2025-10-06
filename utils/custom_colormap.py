# custom_colormap.py
from matplotlib.colors import ListedColormap
from colour import Color

# --- CONFIGURACIÓN ---

CANTIDAD_SUBTIPOS = 4  # Número de subtipos por categoría principal
# 2. Define los 10 colores base para cada categoría principal.
#    Puedes cambiar estos colores si lo deseas.
base_colors = [
    '#e6194b', # Rojo
    '#3cb44b', # Verde
    '#ffe119', # Amarillo
    '#4363d8', # Azul
    '#f58231', # Naranja
    '#911eb4', # Morado
    '#46f0f0', # Cian
    '#f032e6', # Magenta
    '#bcf60c', # Lima
    "#f59f8a", # Rosa claro
]

# --- GENERACIÓN DEL COLORMAL ---
final_colors_list = []
print("Generando gradientes de color para cada categoría...")

for category_id in range(10):
    num_subtypes = CANTIDAD_SUBTIPOS # Obtiene el número de subtipos
    
    # Crea un objeto Color a partir del color base
    base_color = Color(base_colors[category_id])
    
    # Crea un gradiente para esta categoría.
    # El gradiente irá desde una versión más oscura (luminosidad 0.25)
    # a una más clara (luminosidad 0.9) del color base.
    dark_version = Color(base_color)
    dark_version.set_luminance(0.20)
    
    light_version = Color(base_color)
    light_version.set_luminance(0.80)
    
    # Genera los N colores del gradiente
    gradient = list(dark_version.range_to(light_version, num_subtypes))
    
    # Añade los colores generados (en formato hex) a nuestra lista final
    final_colors_list.extend([c.hex for c in gradient])

    print(f"  - Categoría {category_id}: {num_subtypes} tonos de {base_color.hex}")

# Finalmente, crea el objeto Colormap de Matplotlib
# Este es el objeto que importarás en tu main.py
custom_cmap = ListedColormap(final_colors_list)
