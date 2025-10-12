import pandas as pd
import glob
import os

carpeta_csv = r"C:\Users\alems\Downloads\ADA\cluster\csv_subcategorias"

archivos = glob.glob(os.path.join(carpeta_csv, "*.csv"))

dataframes = []

for archivo in archivos:
    df = pd.read_csv(archivo)


    tipo = os.path.splitext(os.path.basename(archivo))[0]


    df["label_tipo"] = tipo


    columnas = ["label"] + [col for col in df.columns if col != "label"]
    df = df[columnas]

    dataframes.append(df)
    
df_final = pd.concat(dataframes, ignore_index=True)

df_final = df_final.sort_values(by=["label", "cluster"]).reset_index(drop=True)

combos_unicos = df_final[["label", "cluster"]].drop_duplicates().reset_index(drop=True)
combos_unicos["categoria"] = range(len(combos_unicos))

df_final = df_final.merge(combos_unicos, on=["label", "cluster"], how="left")

columnas_finales = ["label", "categoria"] + [col for col in df_final.columns if col not in ["label", "categoria"]]
df_final = df_final[columnas_finales]

ruta_salida = os.path.join(carpeta_csv, "fashion_mnist_categorizado_final.csv")
df_final.to_csv(ruta_salida, index=False)

print(f"✅ Archivo final guardado en: {ruta_salida}")
print(f"Total de filas: {len(df_final)}")
print(f"Total de categorías: {df_final['categoria'].nunique()}")
