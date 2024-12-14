import pyopenms
import matplotlib.pyplot as plt
import pandas as pd
import os

# Dictionnaire de correspondance RT -> Annotations
reference_dict = {
    501.571: "PE O-42:5",
    508.8: "PE 40:1",
    417.135: "PE O-37:5",
    389.537: "PC 37:4",
    408.736: "PE O-33:2",
    460.703: "PC O-34:0",
    376.157: "PC 40:7",
    407.986: "PE O-38:6",
    422.685: "PE 40:5",
    536.909: "PC 44:2",
}

def extract_and_store_spectra(mzml_file, rt_targets, output_path):
    exp = pyopenms.MSExperiment()
    pyopenms.MzMLFile().load(mzml_file, exp)
    
    spectra = exp.getSpectra()
    if not spectra:
        print("No spectra found in the file.")
        return

    all_data_list = []
    all_mz_values = set()

    for target_rt in rt_targets:
        closest_spectrum = min(spectra, key=lambda spectrum: abs(spectrum.getRT() - target_rt))
        closest_rt = closest_spectrum.getRT()
        mz_values, intensities = closest_spectrum.get_peaks()
        
        # Ajouter les m/z au set global
        all_mz_values.update(mz_values)

        # Ajouter l'annotation en fonction du dictionnaire
        annotation = reference_dict.get(round(closest_rt, 3), "Unknown")
        if annotation == "Unknown":
            print(f"RT {closest_rt:.3f} not found in reference_dict.")
        
        # Stocker les données dans un DataFrame temporaire
        intensity_dict = dict(zip(mz_values, intensities))
        row_data = {
            "RT (s)": closest_rt,
            "File": os.path.basename(mzml_file),
            "Annotations": annotation,
        }
        row_data.update(intensity_dict)
        all_data_list.append(pd.DataFrame(row_data, index=[0]))

        # Afficher le spectre
        plt.figure(figsize=(10, 6))
        plt.plot(mz_values, intensities, label=f"Spectrum at RT {closest_rt:.2f}s (Target: {target_rt:.2f}s)")
        plt.xlabel("Mass-to-Charge Ratio (m/z)")
        plt.ylabel("Intensity")
        plt.title(f"Spectrum Closest to RT {target_rt:.2f}s")
        plt.legend()
        plt.show()

    # Concaténer toutes les données dans un DataFrame global
    all_data = pd.concat(all_data_list, ignore_index=True)

    # Définir l'ordre des colonnes
    desired_columns = ["RT (s)", "File", "Annotations"] + sorted(all_mz_values)
    all_data = all_data.reindex(columns=desired_columns, fill_value=0)

    # Remplacer les NaN par 0
    all_data = all_data.fillna(0)

    # Sauvegarder au format Parquet
    output_file = os.path.join(output_path, "spectra_data.parquet")
    all_data.to_parquet(output_file, index=False)
    print(f"Parquet file saved to: {output_file}")
    return all_data

if __name__ == "__main__":
    mzml_file = r"C:\\Users\\hassa\\Desktop\\final_test\\data\\20231017_S00068934_N-20231019_S00068934.mzML"
    target_rts = [501.571, 508.8, 417.135, 389.537, 408.736, 460.703, 376.157, 407.986, 422.685, 536.909]
    output_path = r"C:\\Users\\hassa\\Desktop\\final_test"
    print(f"Processing file: {mzml_file}")
    all_spectra_data = extract_and_store_spectra(mzml_file, target_rts, output_path)




