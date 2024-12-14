import pyopenms
import matplotlib.pyplot as plt

def extract_spectrum_from_chromatogram(mzml_file, rt_target):
    """
    Extract and plot the spectrum at a specific retention time (RT) from the chromatograms.
    """
    exp = pyopenms.MSExperiment()  # Create an MSExperiment object
    pyopenms.MzMLFile().load(mzml_file, exp)  # Load the mzML file into the experiment

    # Verify if chromatograms are present
    chromatograms = exp.getChromatograms()
    if not chromatograms:
        print("No chromatograms found in the file.")
        return

    # Find the chromatogram closest to the RT
    print(f"Searching for the chromatogram close to RT {rt_target:.2f}...")
    closest_chromatogram = None
    min_diff = float("inf")
    for chrom in chromatograms:
        rt_values = [point.getRT() for point in chrom]
        if rt_target in rt_values:
            closest_chromatogram = chrom
            break
        else:
            for rt in rt_values:
                diff = abs(rt - rt_target)
                if diff < min_diff:
                    min_diff = diff
                    closest_chromatogram = chrom

    if closest_chromatogram is None:
        print(f"No chromatogram found close to RT {rt_target:.2f}.")
        return

    # Retrieve the spectrum closest to the RT in the chromatogram
    print(f"Extracting spectrum close to RT {rt_target:.2f}...")
    spectra = exp.getSpectra()
    closest_spectrum = min(spectra, key=lambda spectrum: abs(spectrum.getRT() - rt_target))
    closest_rt = closest_spectrum.getRT()
    mz, intensity = closest_spectrum.get_peaks()

    # Plot the extracted spectrum
    plt.figure(figsize=(10, 6))
    plt.plot(mz, intensity, label=f"Spectrum at RT {closest_rt:.2f}s (Target: {rt_target:.2f}s)")
    plt.xlabel("Mass-to-Charge Ratio (m/z)")
    plt.ylabel("Intensity")
    plt.title(f"Spectrum Closest to RT {rt_target:.2f}s")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Replace this with the path to your mzML file
    mzml_file = r"C:\Users\hassa\Desktop\final_test\data\20231017_S00068934_N-20231019_S00068934.mzML"

    # Input the target RT (Retention Time)
    target_rt = float(input("Enter the retention time (RT) to display the spectrum: "))

    print(f"Processing file: {mzml_file}")
    extract_spectrum_from_chromatogram(mzml_file, target_rt)
