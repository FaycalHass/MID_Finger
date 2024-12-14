import pandas as pd

# Chemin vers le fichier Parquet
chemin_fichier = r"C:\Users\hassa\Desktop\final_test\test\meta_GL.parquet"

# Lecture du fichier Parquet
try:
    df = pd.read_parquet(chemin_fichier)
    print("Fichier Parquet lu avec succès.")
    print(df.head())  # Affiche les premières lignes du DataFrame
except Exception as e:
    print(f"Erreur lors de la lecture du fichier Parquet : {e}")
