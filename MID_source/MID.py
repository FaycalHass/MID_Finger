import os
import pyopenms
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.signal import find_peaks
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr, spearmanr
from scipy.sparse import csr_matrix
import streamlit as st
import subprocess
from joblib import Parallel, delayed
import time

# Function to generate output mzML path from RAW file path
def generate_mzml_path(raw_file_path):
    base, _ = os.path.splitext(raw_file_path)
    return base + ".mzML"

# Function to convert RAW to mzML
def convert_raw_to_mzml(raw_file):
    mzml_file = generate_mzml_path(raw_file)
    subprocess.run(["msconvert", raw_file, "-o", os.path.dirname(mzml_file), "--mzML"])
    return mzml_file

# Function to analyze an mzML file
def analyze_mzml_file(mzml_file):
    exp = pyopenms.MSExperiment()
    pyopenms.MzMLFile().load(mzml_file, exp)

    chromatogram = exp.getChromatograms()[0]
    chrom_times, chrom_intensities = chromatogram.get_peaks()

    peaks, _ = find_peaks(chrom_intensities, height=1)
    max_peak_index = np.argmax(chrom_intensities[peaks])
    max_peak_time = chrom_times[peaks[max_peak_index]]
    max_peak_intensity = chrom_intensities[peaks[max_peak_index]]

    # Plot chromatogram and spectrum side by side using Plotly
    fig = go.Figure()

    # Add chromatogram trace
    fig.add_trace(go.Scatter(x=chrom_times, y=chrom_intensities, mode='lines', name='Chromatogram'))
    fig.add_trace(go.Scatter(x=[max_peak_time], y=[max_peak_intensity], mode='markers', name='Most Intense Peak', marker=dict(color='red', size=10)))

    # Configure the chromatogram plot
    fig.update_layout(
        title=f'Chromatography and spectrum of the imported file (Most Intense Peak at {max_peak_time:.2f} seconds)',
        xaxis_title='Time (s)',
        yaxis_title='Intensity',
        xaxis=dict(domain=[0, 0.48]),  # Set domain for chromatogram to take left 48% of the plot width
        yaxis=dict(domain=[0, 1])  # Use full height for chromatogram
    )

    # Add mass spectrum trace (plot the mass spectrum on the second axis)
    max_peak_spectrum = exp.getSpectra()[peaks[max_peak_index]]
    mz_values, mass_intensities = max_peak_spectrum.get_peaks()
    fig.add_trace(go.Scatter(x=mz_values, y=mass_intensities, mode='lines', name='Mass Spectrum', xaxis='x2', yaxis='y2'))

    # Configure the mass spectrum plot (second axis)
    fig.update_layout(
        xaxis2=dict(
            domain=[0.52, 1],  # Set domain for mass spectrum to take right 48% of the plot width
            anchor='y2',
            title='m/z'
        ),
        yaxis2=dict(
            domain=[0, 1],  # Use full height for spectrum
            anchor='x2',
            title='Intensity'
        ),
    )

    # Show the plot in Streamlit
    st.plotly_chart(fig)

    # Create a dataframe from the spectrum data for further processing
    df_import = create_spectrum_dataframe(max_peak_spectrum)
    return df_import


# Function to create a DataFrame from spectrum data
def create_spectrum_dataframe(spectrum):
    mz_values, mass_intensities = spectrum.get_peaks()
    return pd.DataFrame([mass_intensities], columns=mz_values)

# Function to adjust DataFrames by adding missing columns
def adjust_dataframes(df_import, df_subset):
    subset_columns = df_subset.columns
    import_columns = df_import.columns

    missing_columns_in_import = [col for col in subset_columns if col not in import_columns]
    missing_columns_in_subset = [col for col in import_columns if col not in subset_columns]

    df_import = pd.concat([df_import, pd.DataFrame(columns=missing_columns_in_import)], axis=1)
    df_subset = pd.concat([df_subset, pd.DataFrame(columns=missing_columns_in_subset)], axis=1)

    sorted_subset_columns = sorted(df_subset.columns, key=lambda x: float(x))
    sorted_import_columns = sorted(df_import.columns, key=lambda x: float(x))

    df_import = df_import.reindex(columns=sorted_import_columns).fillna(0)
    df_subset = df_subset.reindex(columns=sorted_subset_columns).fillna(0)

    return df_import, df_subset

# Function to convert DataFrame to sparse matrix
def to_sparse_matrix(df):
    return csr_matrix(df.values)

# Function to calculate similarities using parallel processing
def calculate_similarities(df_import, df_subset):
    # Ensure all column names are strings
    df_import.columns = df_import.columns.astype(str)
    df_subset.columns = df_subset.columns.astype(str)

    # Convert to sparse matrices
    df_import_sparse = to_sparse_matrix(df_import)
    df_subset_sparse = to_sparse_matrix(df_subset)

    cosine_similarities = cosine_similarity(df_import_sparse, df_subset_sparse)

    def calculate_pearson_spearman(i):
        pearson_scores = []
        spearman_scores = []
        for j in range(df_subset_sparse.shape[0]):
            pearson_score, _ = pearsonr(df_import_sparse[i].toarray()[0], df_subset_sparse[j].toarray()[0])
            spearman_score, _ = spearmanr(df_import_sparse[i].toarray()[0], df_subset_sparse[j].toarray()[0])
            pearson_scores.append(pearson_score)
            spearman_scores.append(spearman_score)
        return pearson_scores, spearman_scores

    results = Parallel(n_jobs=-1)(delayed(calculate_pearson_spearman)(i) for i in range(df_import_sparse.shape[0]))
    pearson_similarities, spearman_similarities = zip(*results)

    pearson_similarities = np.array(pearson_similarities)
    spearman_similarities = np.array(spearman_similarities)

    return cosine_similarities, pearson_similarities, spearman_similarities

# Function to display results based on similarity threshold
def display_results_threshold(similarities, threshold, metric_name, df):
    similar_rows = np.where(similarities >= threshold)
    if similar_rows[0].size == 0:
        st.warning(f"No results found with a similarity threshold of {threshold} for {metric_name}.")
    else:
        for i, j in zip(*similar_rows):
            st.write(f"### Results for the Most Similar Spectrum:{df.iloc[j]['Annotations']}")
            st.write("\n" + "-"*50 + "\n")
            st.write(f"For row {j} in the reference data:\n")
            st.write(f"- The {metric_name} is: {similarities[i, j]:.4f}")
            st.write(f"- Associated file: {df.iloc[j]['File']}")
            st.write(f"- Analysis date: {df.iloc[j]['Date']}")
            st.write(f"- Observed m/z: {df.iloc[j]['m/z']}")
            st.write(f"- Polarity: {df.iloc[j]['Polarité']}")
            st.write(f"- MS type: {df.iloc[j]['Type MS']}")
            st.write(f"- Tissue type: {df.iloc[j]['Type']}")
            st.write(f"- Tissue: {df.iloc[j]['Tissus']}")
            st.write(f"- Subtype: {df.iloc[j]['Sous-type']}")
            st.write(f"- Annotations: {df.iloc[j]['Annotations']}")
            st.write(f"- Sum of intensities: {df.iloc[j]['Sum']}\n")
            st.write("\n" + "-"*50 + "\n")

# Streamlit interface configuration (must be the first Streamlit command)
st.set_page_config(
    page_title="MID: Molecule_ID based MSMS fingerprint Similarity",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Function to load and display a logo image
def load_logo(image_filename):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current script
    image_path = os.path.join(current_dir, image_filename)    # Construct the relative path to the logo
    st.sidebar.image(image_path, use_column_width=True)       # Display the logo in the sidebar

# File name of the logo (place the image in the same folder as this script)
logo_filename = "prism-logo-full.png"  # Logo file in the same directory

# Load logo at the start of the app
load_logo(logo_filename)

st.title('🔬 MID: Molecule_ID based MSMS fingerprint Similarity')

with st.sidebar:
    st.header("Settings")
    folder_path = st.text_input("📁 Folder Path Containing RAW Files")
    similarity_threshold = st.slider('🔍 Similarity Threshold', min_value=0.0, max_value=1.0, step=0.01, value=0.5)

    if st.button("Import Positive Reference Data."):
        start_time = time.time()
        progress_text = "Importing Positive Reference Data..."
        my_bar = st.progress(0, text=progress_text)
        script_dir = os.path.dirname(os.path.realpath(__file__))
        positive_parquet_file_path = os.path.join(script_dir, 'data_ref_pos.parquet')# Path to the positive parquet file
        parquet_file = pd.read_parquet(positive_parquet_file_path)
        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1, text=progress_text)
        time.sleep(1)
        my_bar.empty()
        elapsed_time = (time.time() - start_time) / 60
        st.session_state['parquet_file'] = parquet_file  # Store positive data in session state
        st.success(f"Positive reference data successfully imported in {elapsed_time:.2f} minutes!")

    if st.button("Import Negative Reference Data"):
        start_time = time.time()
        progress_text = "Importing Negative Reference Data..."
        my_bar = st.progress(0, text=progress_text)
        script_dir = os.path.dirname(os.path.realpath(__file__))  # Répertoire du script
        parquet_file_path = os.path.join(script_dir, 'data_ref_pos_float32.parquet') # Path to the negative parquet file
        parquet_file = pd.read_parquet(parquet_file_path)
        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1, text=progress_text)
        time.sleep(1)
        my_bar.empty()
        elapsed_time = (time.time() - start_time) / 60
        st.session_state['parquet_file'] = parquet_file  # Store positive data in session state
        st.success(f"Reference data successfully imported in {elapsed_time:.2f} minutes!")
        
    if st.button("Import Metabolite reference base"):
        start_time = time.time()
        progress_text = "Importing Metabolite reference base..."
        my_bar = st.progress(0, text=progress_text)
        script_dir = os.path.dirname(os.path.realpath(__file__))  # Répertoire du script
        parquet_file_path = os.path.join(script_dir, 'metabolite.parquet.parquet') # Path to the negative parquet file
        parquet_file = pd.read_parquet(parquet_file_path)
        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1, text=progress_text)
        time.sleep(1)
        my_bar.empty()
        elapsed_time = (time.time() - start_time) / 60
        st.session_state['parquet_file'] = parquet_file  # Store positive data in session state
        st.success(f"Reference data successfully imported in {elapsed_time:.2f} minutes!")

if st.sidebar.button("Analyze RAW Files"):
    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)

    for percent_complete in range(100):
        time.sleep(0.01)  # Simulation de progression
        my_bar.progress(percent_complete + 1, text=progress_text)

    time.sleep(1)
    my_bar.empty()
    st.success("Operation Completed Successfully!")  # Message de succès après la progression

    start_time = time.time()
    parquet_file = st.session_state.get('parquet_file', None)  # Retrieve reference database from session state

    if parquet_file is not None:
        if folder_path:
            raw_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.raw')]

              
            if raw_files:
                st.write("## Reference Data Content")
                
                # Retirer une colonne si elle existe, puis réinitialiser l'index
                df = parquet_file.drop('column_name', axis=1, errors='ignore').reset_index(drop=True)
                
                # Afficher les 5 premières et 5 dernières lignes du DataFrame
                
                st.dataframe(df.iloc[:, :9])
                
                total_files = len(raw_files)
            
                # Fit PCA pipeline on the reference data after alignment
                df_subset = df.iloc[:, 10:]
            
                for raw_file in raw_files:
                    with st.spinner(f"Converting {raw_file} to mzML..."):
                        mzml_file = convert_raw_to_mzml(raw_file)
                        st.success(f"Conversion of {raw_file} to mzML successful!")
            
                    with st.spinner(f"Analyzing {mzml_file}..."):
                        df_import = analyze_mzml_file(mzml_file)
            
                    df_import, df_subset = adjust_dataframes(df_import, df_subset)
                    df_import.columns = df_import.columns.astype(str)
                    df_subset.columns = df_subset.columns.astype(str)
            
                    cosine_similarities, pearson_similarities, spearman_similarities = calculate_similarities(df_import, df_subset)
            
                    with st.expander("Cosine Similarity"):
                        display_results_threshold(cosine_similarities, similarity_threshold, "Cosine Similarity", df)
                    with st.expander("Pearson Correlation"):
                        display_results_threshold(pearson_similarities, similarity_threshold, "Pearson Correlation", df)
                    with st.expander("Spearman Correlation"):
                        display_results_threshold(spearman_similarities, similarity_threshold, "Spearman Correlation", df)
            
                elapsed_time = (time.time() - start_time) / 60
                st.success(f"RAW file analysis completed in {elapsed_time:.2f} minutes!")
            else:
                st.error("No RAW files found in the specified folder.")
        else:
            st.error("Please provide the path to the folder containing the RAW files.")
    else:
        st.error("Please import the reference data before analyzing the RAW files.")


