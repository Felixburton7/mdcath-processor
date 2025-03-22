# 🧪 mdCATH Dataset Processor

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![BioMolecular Analysis](https://img.shields.io/badge/Biomolecular-Analysis-green.svg)](https://github.com/Felixburton7/mdcath-processor)
[![Protein Dynamics](https://img.shields.io/badge/Protein-Dynamics-red.svg)](https://github.com/Felixburton7/mdcath-processor)

<p align="center">
  <img src="https://img.shields.io/badge/🔬%20Protein-Dynamics-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/🧠%20Machine-Learning-purple?style=for-the-badge" />
  <img src="https://img.shields.io/badge/🔄%20Molecular-Simulations-teal?style=for-the-badge" />
</p>

A comprehensive suite for processing mdCATH protein dynamics dataset to facilitate machine learning-based prediction of Root Mean Square Fluctuation (RMSF) from protein structures.

---

## 📑 Table of Contents

<div style="display: grid; grid-template-columns: 1fr 1fr; grid-gap: 20px; margin-bottom: 30px;">
<div>

### Getting Started
- [🌟 Overview](#-overview)
- [🧬 About mdCATH Dataset](#-about-mdcath-dataset)
- [🔬 Key Features](#-key-features)
- [⚙️ Installation](#️-installation)
- [🚀 Quick Start](#-quick-start)
  
### Core Documentation
- [🔄 Dataflow Pipeline](#-dataflow-pipeline)
- [🛠️ Configuration Options](#️-configuration-options)
- [📊 Output Examples](#-output-examples)

</div>
<div>

### Technical Details
- [📂 Project Structure](#-project-structure)
- [🧩 Modules Explained](#-modules-explained)
- [🔮 Advanced Usage](#-advanced-usage)

### Resources
- [📚 API Reference](#-api-reference)
- [🐞 Troubleshooting](#-troubleshooting)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)

</div>
</div>

---

## 🌟 Overview

The mdCATH Dataset Processor is designed to transform raw molecular dynamics data from the mdCATH dataset into structured, analysis-ready formats optimized for machine learning applications. This pipeline extracts, processes, and organizes protein dynamics data, with a focus on Root Mean Square Fluctuation (RMSF) prediction.

By providing a consistent framework for data preparation, this project enables researchers to develop machine learning models that can accurately predict protein dynamics from structural features alone, potentially accelerating drug discovery and protein engineering efforts.

## 🧬 About mdCATH Dataset

<div style="background-color: #f0f8ff; padding: 15px; border-radius: 8px; border-left: 5px solid #007bff; margin: 20px 0;">
<h3 style="margin-top: 0;">📚 mdCATH: A Large-Scale MD Dataset</h3>

<p>The mdCATH dataset is a comprehensive collection of molecular dynamics simulations designed for data-driven computational biophysics research. Created by Antonio Mirarchi, Toni Giorgino, and Gianni De Fabritiis, this dataset provides:</p>

<ul>
  <li>🧪 Multiple temperature simulations (320K-450K)</li>
  <li>🔄 Multiple replicas per temperature</li>
  <li>⏱️ Extensive trajectory data for protein domains from the CATH database</li>
  <li>📊 Rich structural and dynamics information including coordinates, RMSF, and DSSP data</li>
</ul>

<p>The original dataset is available on <a href="https://huggingface.co/datasets/compsciencelab/mdCATH">Hugging Face</a> and can be visualized on <a href="https://open.playmolecule.org/mdcath">PlayMolecule</a>.</p>

<p><i>Citation: Mirarchi, A., Giorgino, T., & De Fabritiis, G. (2024). mdCATH: A Large-Scale MD Dataset for Data-Driven Computational Biophysics. <a href="https://arxiv.org/abs/2407.14794">arXiv:2407.14794</a></i></p>
</div>

## 🔬 Key Features

<div style="display: grid; grid-template-columns: 1fr 1fr; grid-gap: 20px; margin-bottom: 30px;">
<div style="background-color: rgba(240, 248, 255, 0.6); padding: 15px; border-radius: 8px;">
  <h3>📊 Data Processing</h3>
  <ul>
    <li>📈 <b>Comprehensive Data Extraction</b>: Extract RMSF, DSSP, and coordinate data from H5 files</li>
    <li>🧹 <b>Sophisticated PDB Processing</b>: Clean and standardize PDB files</li>
    <li>🌡️ <b>Multi-temperature Analysis</b>: Process data across temperatures (320K-450K)</li>
    <li>🔄 <b>Multi-replica Support</b>: Handle multiple simulation replicas</li>
  </ul>
</div>

<div style="background-color: rgba(255, 248, 240, 0.6); padding: 15px; border-radius: 8px;">
  <h3>🧠 ML & Analysis Features</h3>
  <ul>
    <li>🧮 <b>ML-Ready Feature Generation</b>: Create optimized feature sets</li>
    <li>🎯 <b>Core/Exterior Classification</b>: Classify protein residues using solvent accessibility</li>
    <li>📊 <b>Insightful Visualizations</b>: Generate publication-quality plots</li>
    <li>🧩 <b>3D Voxelization</b>: Convert proteins to 3D grids for deep learning</li>
  </ul>
</div>
</div>

## ⚙️ Installation

### Prerequisites

<table>
<thead>
  <tr>
    <th>Requirement</th>
    <th>Purpose</th>
    <th>Installation</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Python 3.9+</td>
    <td>Base environment</td>
    <td><a href="https://www.python.org/downloads/">python.org</a></td>
  </tr>
  <tr>
    <td>H5py</td>
    <td>HDF5 file handling</td>
    <td><code>pip install h5py</code></td>
  </tr>
  <tr>
    <td>Biopython</td>
    <td>PDB handling & DSSP</td>
    <td><code>pip install biopython</code></td>
  </tr>
  <tr>
    <td>aposteriori</td>
    <td>Voxelization (optional)</td>
    <td><code>pip install aposteriori</code></td>
  </tr>
  <tr>
    <td>pdbUtils</td>
    <td>Enhanced PDB processing</td>
    <td><code>pip install pdbUtils</code></td>
  </tr>
</tbody>
</table>

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Felixburton7/mdcath-processor.git
cd mdcath-processor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install additional tools:
```bash
# For voxelization support
pip install aposteriori

# For enhanced PDB handling
pip install pdbUtils
```

4. Verify installation:
```bash
python check_environment.py
```

## 🚀 Quick Start

### Basic Usage

```bash
# Process default domains with standard settings
python main.py

# Process specific domains
python main.py --domain_ids 12asA00 153lA00

# Use custom configuration
python main.py --config my_config.yaml
```

### Example Workflow

```python
# Import necessary modules
from src.mdcath.core.data_loader import H5DataLoader, process_domains
from src.mdcath.processing import pdb, rmsf, features, visualization

# Load configuration
import yaml
with open('src/mdcath/config/default_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Adjust configuration as needed
config['input']['domain_ids'] = ['12asA00', '153lA00']
config['output']['base_dir'] = './custom_outputs'

# Process domains
domain_results = process_domains(config['input']['domain_ids'], 
                                config['input']['mdcath_folder'], 
                                config)

# Process PDB data
pdb_results = pdb.process_pdb_data(domain_results, config)

# Process RMSF data
rmsf_results = rmsf.process_rmsf_data(domain_results, config)

# Generate ML features
ml_results = features.process_ml_features(rmsf_results, pdb_results, domain_results, config)

# Generate visualizations
vis_results = visualization.generate_visualizations(rmsf_results, ml_results, domain_results, config)
```

## 🔄 Dataflow Pipeline

The mdCATH processor transforms raw H5 data through a sophisticated pipeline of processing steps:

<div align="center" style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0;">
  <img src="https://mermaid.ink/img/pako:eNqFksFuwjAMhl_F8mlI0MLYhgaHHSZNmrRJO1S9mMRQi6ZOlYQJId59aQutNm0bl8T-f_v3r2SAkmMGAZztjHbf2u6MHGFaFtZpA8El6BjWSpVEpLxZJg-ztODxYZ4OEEilc6TM62pGt7ZbPZHrvGu_WZOb9mlh1J5srb9pNcnrDmEwi6LoYjQjQ-56hKENNm8LhMGDxbgSxk0Moe95nuc3GJW6n-hFv8XQMmXKyGKoE_RxNkKYGbsxWBRcmgphE0MzYaPEw9n2f6m8wCUKSJmwGZIgONZ2oaQ07zx4QKuXymWmIxTvEApIl5JhwG9yITnPvTU-8P4y73XwC_hhzT4L6KxVgoBJXZb9-FweM19g7Uh0MIBu-byfBrngJQSiUUq2qEy6KOvLw-lGFcxX1oUA-2L3-IbQ8e0VZ71Y-R6TQoAjnrdbxu95Ng?type=png" alt="Data Processing Pipeline" width="700">
</div>

### Detailed Process Flow:

1. **Data Extraction** 🔍
   - Raw H5 files are parsed to extract multiple data types:
     - PDB structures for structural analysis
     - RMSF values for flexibility quantification
     - DSSP annotations for secondary structure
     - Atomic coordinates for 3D analysis

2. **PDB Processing** 🧹
   - PDB files are cleaned and standardized
   - Atom numbering is fixed
   - Unusual residue names are corrected
   - CRYST1 records are added for compatibility

3. **Coordinate Processing** 📍
   - Extract trajectories from the H5 files
   - Calculate RMSD and gyration radius metrics
   - Prepare coordinates for frame selection

4. **RMSF Analysis** 📈
   - Calculate per-residue RMSF values
   - Average across replicas and temperatures
   - Generate comprehensive flexibility profiles

5. **Structure Classification** 🧩
   - Classify residues as core or exterior
   - Analyze secondary structure with DSSP
   - Measure solvent accessibility

6. **Frame Selection** 🖼️
   - Extract representative frames using various methods:
     - RMSD clustering for diverse conformations
     - Gyration radius for size-based selection
     - Regular sampling for uniform coverage

7. **Feature Generation** 🧮
   - Combine RMSF, structural, and sequence information
   - Create ML-ready feature datasets
   - Normalize and encode features

8. **Visualization** 📊
   - Generate publication-quality plots
   - Visualize flexibility patterns
   - Create correlation analyses

9. **Voxelization** 🧊
   - Convert proteins to 3D voxel grids
   - Prepare data for 3D deep learning
   - Encode atom types in separate channels

## 🛠️ Configuration Options

The processing pipeline is highly configurable via a YAML configuration file. Here are the key configuration sections:

<div style="display: grid; grid-template-columns: 1fr 1fr; grid-gap: 20px; margin-bottom: 30px;">

<!-- Input/Output Configuration -->
<div style="background-color: #f0f7ff; padding: 15px; border-radius: 8px; border-left: 5px solid #3498db;">
<h3>📥 Input/Output Configuration</h3>

```yaml
input:
  mdcath_folder: "/mnt/datasets/MD_CATH/data"  # H5 files path
  domain_ids: [                           # Domains to process
    "12asA00",
    "153lA00",
    "16pkA02",
    # More domains...
  ]

temperatures: [320, 348, 379, 413, 450]  # Temps to process
num_replicas: 5                          # Replicas per temp

output:
  base_dir: "./outputs"                  # Output directory
```
</div>

<!-- Processing Parameters -->
<div style="background-color: #f0fff7; padding: 15px; border-radius: 8px; border-left: 5px solid #2ecc71;">
<h3>⚙️ Processing Parameters</h3>

```yaml
processing:
  frame_selection:
    method: "rmsd"        # regular/rmsd/gyration/random
    num_frames: 4         # Frames per domain/temperature
    cluster_method: "kmeans"  # For RMSD-based selection

  pdb_cleaning:
    replace_chain_0_with_A: true    # Replace chain '0' with 'A'
    fix_atom_numbering: true        # Fix atom numbering
    correct_unusual_residue_names: true  # Convert non-standard names
    add_cryst1_record: true         # Add CRYST1 record
    remove_hydrogens: false         # Remove H atoms
    remove_solvent_ions: true       # Remove water/ions
    stop_after_ter: true            # Stop at TER record
```
</div>

<!-- ML Feature Extraction -->
<div style="background-color: #fff7f0; padding: 15px; border-radius: 8px; border-left: 5px solid #e67e22;">
<h3>🧠 ML Feature Extraction</h3>

```yaml
processing:
  ml_feature_extraction:
    min_residues_per_domain: 0      # Min residues filter
    max_residues_per_domain: 50000  # Max residues filter
    normalize_features: true        # Normalize features
    include_secondary_structure: true  # Include DSSP
    include_core_exterior: true     # Core/exterior data
    include_dssp: true              # Per-residue DSSP

  core_exterior:
    method: "msms"                  # msms/biopython/fallback
    msms_executable_dir: "./msms_executables"
    ses_threshold: 1.0              # MSMS threshold (Å²)
    sasa_threshold: 20.0            # Biopython threshold (Å²)
```
</div>

<!-- Voxelization & Performance -->
<div style="background-color: #f7f0ff; padding: 15px; border-radius: 8px; border-left: 5px solid #9b59b6;">
<h3>🧊 Voxelization & Performance</h3>

```yaml
processing:
  voxelization:
    frame_edge_length: 12.0         # Grid size (Å)
    voxels_per_side: 21             # Grid resolution
    atom_encoder: "CNOCBCA"         # Atom types to include
    encode_cb: true                 # Include CB atoms
    compression_gzip: true          # Compress output
    voxelise_all_states: false      # Multiple states
    process_frames: false           # Voxelize frames
    
performance:
  num_cores: 10                     # 0 = auto-detect
  batch_size: 80                    # Batch size
  memory_limit_gb: 26               # 0 = no limit
  use_gpu: true                     # Use GPU if available
```
</div>
</div>

### Configuration Examples

<details>
<summary><b>🔍 Click to view example configurations</b></summary>

#### Minimal Configuration (Processing a Single Domain)

```yaml
input:
  mdcath_folder: "/path/to/mdcath/data"
  domain_ids: ["12asA00"]
temperatures: [320]
num_replicas: 1
output:
  base_dir: "./minimal_output"
```

#### Production Configuration (All Domains and Temperatures)

```yaml
input:
  mdcath_folder: "/path/to/mdcath/data"
  domain_ids: []  # Process all domains
temperatures: [320, 348, 379, 413, 450]
num_replicas: 5
output:
  base_dir: "./full_output"
performance:
  num_cores: 16
  memory_limit_gb: 32
  use_gpu: true
```

#### Advanced Frame Selection Configuration

```yaml
processing:
  frame_selection:
    method: "rmsd"               # Select frames based on RMSD clustering
    num_frames: 5                # Extract 5 representative frames
    cluster_method: "kmeans"     # Use k-means clustering

  # Additional processing options
  pdb_cleaning:
    remove_hydrogens: true       # Clean up PDB by removing hydrogens
    remove_solvent_ions: true    # Remove water molecules and ions
```
</details>

## 📊 Output Examples

The mdCATH processor generates multiple structured outputs for analysis and modeling:

<div style="display: grid; grid-template-columns: 1fr 1fr; grid-gap: 20px; margin-bottom: 30px;">

<!-- RMSF Analysis Results -->
<div style="background-color: #f0f7ff; padding: 15px; border-radius: 8px;">
<h3>📈 RMSF Analysis Results</h3>
<p><b>File:</b> <code>outputs/RMSF/replica_average/average/rmsf_all_temperatures_all_replicas.csv</code></p>

<div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; font-family: monospace; font-size: 0.85em; overflow-x: auto;">
domain_id,resid,resname,rmsf_320,rmsf_348,rmsf_379,rmsf_413,rmsf_450,rmsf_average<br>
12asA00,1,MET,1.243,1.321,1.467,1.589,1.723,1.469<br>
12asA00,2,LYS,1.103,1.174,1.256,1.392,1.532,1.291<br>
12asA00,3,ILE,0.936,0.987,1.075,1.156,1.267,1.084
</div>

<p>Average RMSF values across replicas and temperatures for each residue.</p>
</div>

<!-- ML-Ready Feature Datasets -->
<div style="background-color: #f0fff7; padding: 15px; border-radius: 8px;">
<h3>🧠 ML-Ready Feature Datasets</h3>
<p><b>File:</b> <code>outputs/ML_features/final_dataset_temperature_average.csv</code></p>

<div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; font-family: monospace; font-size: 0.85em; overflow-x: auto; white-space: nowrap;">
domain_id,resid,resname,rmsf_320,...,protein_size,normalized_resid,core_exterior,relative_accessibility,dssp,...<br>
12asA00,1,MET,1.243,...,330,0.000,exterior,0.85,C,13,1,2,0.00,0.00,1.469<br>
12asA00,2,LYS,1.103,...,330,0.003,exterior,0.63,T,12,1,2,-0.42,0.33,1.291<br>
</div>

<p>Comprehensive dataset with RMSF values and structural features.</p>
</div>
</div>

<div style="display: grid; grid-template-columns: 1fr 1fr; grid-gap: 20px; margin-bottom: 30px;">

<!-- PDB Files and Frames -->
<div style="background-color: #fff7f0; padding: 15px; border-radius: 8px;">
<h3>🧬 PDB Files and Frames</h3>

<p><b>Cleaned PDB:</b> <code>outputs/pdbs/12asA00.pdb</code></p>
<p><b>Frames:</b> <code>outputs/frames/replica_0/320/12asA00_frame_0.pdb</code></p>

<div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; font-family: monospace; font-size: 0.85em; overflow-x: auto;">
CRYST1  100.000  100.000  100.000  90.00  90.00  90.00 P 1           1<br>
ATOM      1  N   MET A   1      -9.152  25.423   4.759  1.00  0.00           N<br>
ATOM      2  CA  MET A   1      -9.446  24.674   3.532  1.00  0.00           C<br>
...
</div>

<p>Cleaned PDB structures and representative trajectory frames.</p>
</div>

<!-- Visualizations -->
<div style="background-color: #f7f0ff; padding: 15px; border-radius: 8px;">
<h3>📊 Visualizations</h3>

<p>The project generates multiple high-quality visualizations:</p>

<ul>
  <li>🌡️ <b>RMSF Distribution by Temperature</b></li>
  <li>🧬 <b>RMSF by Secondary Structure</b></li>
  <li>🔍 <b>Feature Correlations</b></li>
  <li>🌈 <b>Temperature Summary Heatmap</b></li>
  <li>🧪 <b>Amino Acid-specific RMSF Analysis</b></li>
</ul>

<p>Located in <code>outputs/visualizations/</code> directory.</p>
</div>
</div>

<details>
<summary><b>🖼️ Click to see example visualization previews</b></summary>

<div style="text-align: center;">
  <p><b>RMSF Distribution by Temperature:</b></p>
  <img src="https://i.imgur.com/UYsKg2P.png" alt="RMSF Violin Plot" width="500"/>
  
  <p><b>RMSF by Secondary Structure:</b></p>
  <img src="https://i.imgur.com/FWsZnc7.png" alt="DSSP-RMSF Correlation" width="500"/>
  
  <p><b>Amino Acid-specific RMSF Analysis:</b></p>
  <img src="https://i.imgur.com/JdEb4uQ.png" alt="Amino Acid RMSF" width="500"/>
</div>
</details>

## 📂 Project Structure

The mdCATH processor has the following structure:

<div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; font-family: monospace; font-size: 0.9em; overflow-x: auto; white-space: pre;">
mdcath-processor/
├── 📄 main.py                  <span style="color: #6a737d;">// Main entry point</span>
├── 📄 setup.py                 <span style="color: #6a737d;">// Installation setup</span>
├── 📄 setup.sh                 <span style="color: #6a737d;">// Setup script</span>
├── 📄 requirements.txt         <span style="color: #6a737d;">// Dependencies</span>
├── 📄 check_environment.py     <span style="color: #6a737d;">// Environment verification</span>
├── 📄 LICENSE                  <span style="color: #6a737d;">// MIT License</span>
├── 📄 README.md                <span style="color: #6a737d;">// This documentation</span>
├── 📄 all_domain_ids.txt       <span style="color: #6a737d;">// List of all available domains</span>
├── 📁 msms_executables/        <span style="color: #6a737d;">// Surface calculation tools</span>
├── 📁 src/                     <span style="color: #6a737d;">// Source code</span>
│   └── 📁 mdcath/              <span style="color: #6a737d;">// Main package</span>
│       ├── 📄 __init__.py      <span style="color: #6a737d;">// Package initialization</span>
│       ├── 📁 config/          <span style="color: #6a737d;">// Configuration handling</span>
│       ├── 📁 core/            <span style="color: #6a737d;">// Core functionality</span>
│       └── 📁 processing/      <span style="color: #6a737d;">// Processing modules</span>
└── 📁 outputs/                 <span style="color: #6a737d;">// Generated outputs</span>
    ├── 📁 frames/              <span style="color: #6a737d;">// Extracted frames</span>
    ├── 📁 ML_features/         <span style="color: #6a737d;">// Feature datasets</span>
    ├── 📁 pdbs/                <span style="color: #6a737d;">// Cleaned PDB files</span>
    ├── 📁 RMSF/                <span style="color: #6a737d;">// RMSF analysis</span>
    ├── 📁 visualizations/      <span style="color: #6a737d;">// Generated plots</span>
    └── 📁 voxelized/           <span style="color: #6a737d;">// Voxelized data</span>
</div>

## 🧩 Modules Explained

<table>
<thead>
  <tr>
    <th>Module</th>
    <th>Purpose</th>
    <th>Key Functions</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td><b>📦 data_loader.py</b></td>
    <td>Extracts data from H5 files</td>
    <td>
      • <code>H5DataLoader</code>: Main class for loading H5 files<br>
      • <code>extract_rmsf()</code>: Gets RMSF data<br>
      • <code>extract_pdb()</code>: Gets PDB structures<br>
      • <code>extract_coordinates()</code>: Gets atomic coordinates
    </td>
  </tr>
  <tr>
    <td><b>📦 core_exterior.py</b></td>
    <td>Classifies residue locations</td>
    <td>
      • <code>compute_core_exterior()</code>: Main classification<br>
      • <code>run_dssp_once()</code>: Runs DSSP with caching<br>
      • <code>compute_core_exterior_biopython()</code>: Uses Biopython
    </td>
  </tr>
  <tr>
    <td><b>📦 pdb.py</b></td>
    <td>Handles PDB cleaning & frames</td>
    <td>
      • <code>fix_pdb()</code>: Coordinates PDB cleaning<br>
      • <code>fix_pdb_with_pdbutils()</code>: PDB cleaning<br>
      • <code>extract_frames()</code>: Extracts frames from trajectories
    </td>
  </tr>
  <tr>
    <td><b>📦 rmsf.py</b></td>
    <td>Analyzes RMSF data</td>
    <td>
      • <code>calculate_replica_averages()</code>: Averages across replicas<br>
      • <code>calculate_temperature_average()</code>: Cross-temp averages<br>
      • <code>process_rmsf_data()</code>: High-level processing
    </td>
  </tr>
  <tr>
    <td><b>📦 features.py</b></td>
    <td>Generates ML-ready features</td>
    <td>
      • <code>generate_ml_features()</code>: Creates feature datasets<br>
      • <code>save_ml_features()</code>: Saves to CSV files<br>
      • <code>process_ml_features()</code>: High-level generation
    </td>
  </tr>
  <tr>
    <td><b>📦 visualization.py</b></td>
    <td>Creates visualizations</td>
    <td>
      • <code>create_temperature_summary_heatmap()</code>: Temperature analysis<br>
      • <code>create_amino_acid_rmsf_plot()</code>: AA-specific RMSF<br>
      • <code>create_dssp_rmsf_correlation_plot()</code>: Structure correlations
    </td>
  </tr>
  <tr>
    <td><b>📦 voxelizer.py</b></td>
    <td>Voxelizes protein structures</td>
    <td>
      • <code>voxelize_domains()</code>: Converts to voxel grids
    </td>
  </tr>
</tbody>
</table>

## 🔮 Advanced Usage

<div style="display: grid; grid-template-columns: 1fr; grid-gap: 20px; margin-bottom: 30px;">

<!-- Custom Data Processing -->
<div style="background-color: #f0f7ff; padding: 15px; border-radius: 8px; border-left: 5px solid #3498db;">
<h3>🔬 Custom Data Processing Workflow</h3>

```python
from src.mdcath.core.data_loader import H5DataLoader
from src.mdcath.processing import core_exterior, pdb, features
import yaml

# Load configuration
with open('src/mdcath/config/default_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Customize configuration
config['temperatures'] = [320, 348]  # Process only specific temperatures

# Custom data loading
h5_path = "/path/to/mdcath_dataset_12asA00.h5"
loader = H5DataLoader(h5_path, config)

# Extract specific data
pdb_data = loader.extract_pdb()
rmsf_data = loader.extract_rmsf("320", "0")
coords, resids, resnames, rmsd_data, gyration_data = loader.extract_coordinates("320", "0", frame=-1)
dssp_data = loader.extract_dssp("320", "0")

# Process PDB
pdb_path = "custom_output/12asA00.pdb"
pdb.save_pdb_file(pdb_data, pdb_path, config)

# Analyze core/exterior
ce_data = core_exterior.compute_core_exterior(pdb_path, config)

# Extract just one frame using RMSD clustering
pdb.extract_frames(coords, resids, resnames, "12asA00", 
                  "custom_output", "320", "0", config,
                  rmsd_data, gyration_data)
```
</div>

<!-- ML Model Building -->
<div style="background-color: #f0fff7; padding: 15px; border-radius: 8px; border-left: 5px solid #2ecc71;">
<h3>🧠 Using Extracted Features for Machine Learning</h3>

```python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the feature dataset
features_df = pd.read_csv("outputs/ML_features/final_dataset_temperature_average.csv")

# Select features and target
X = features_df[['normalized_resid', 'relative_accessibility', 
                'secondary_structure_encoded', 'core_exterior_encoded',
                'phi_norm', 'psi_norm']]
y = features_df['rmsf_average']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"RMSE: {rmse:.4f}")

# Feature importance
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)
print(importance)
```
</div>

<!-- Parallel Processing -->
<div style="background-color: #fff7f0; padding: 15px; border-radius: 8px; border-left: 5px solid #e67e22;">
<h3>⚡ Parallel Processing for Large Datasets</h3>

```python
from src.mdcath.core.data_loader import process_domains
import multiprocessing
import yaml

# Load configuration
with open('src/mdcath/config/default_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Set parallelization options
num_cores = max(1, multiprocessing.cpu_count() - 2)  # Use all but 2 cores
config['performance']['num_cores'] = num_cores
config['performance']['batch_size'] = 20  # Process 20 domains at a time

# Process multiple domains in parallel
domain_ids = ["12asA00", "153lA00", "16pkA02", "1a02F00", "1a15A00"]
results = process_domains(domain_ids, config['input']['mdcath_folder'], config, 
                         num_cores=num_cores)
```
</div>

<!-- PyTorch Integration -->
<div style="background-color: #f7f0ff; padding: 15px; border-radius: 8px; border-left: 5px solid #9b59b6;">
<h3>🔥 Using Voxelized Data with PyTorch</h3>

```python
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F

# Load voxelized data
voxel_file = "outputs/voxelized/mdcath_voxelized.h5"
with h5py.File(voxel_file, 'r') as f:
    # Assuming format matches aposteriori output structure
    domains = list(f.keys())
    domain = domains[0]  # First domain
    chain = list(f[domain].keys())[0]  # First chain
    
    # Load voxel data - shape should be [batch, channels, depth, height, width]
    voxels = torch.tensor(f[domain][chain][:])

# Simple 3D CNN for voxel data
class Voxel3DCNN(nn.Module):
    def __init__(self, in_channels=6):  # Default for 'CNOCBCA' encoding
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(2)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 5 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 1)  # Predict RMSF
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create model instance
model = Voxel3DCNN()
```
</div>
</div>

## 📚 API Reference

<div style="background-color: #f0f8ff; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
<h3 style="margin-top: 0;">🔍 Key Classes and Functions</h3>

<details>
<summary><b>📦 Data Loading API</b></summary>

```python
from src.mdcath.core.data_loader import H5DataLoader, process_domains

# Initialize a data loader
loader = H5DataLoader(h5_path, config)

# Extract data
pdb_data = loader.extract_pdb()
rmsf_data = loader.extract_rmsf(temp, replica)
coords, resids, resnames, rmsd_data, gyration_data = loader.extract_coordinates(temp, replica)
dssp_data = loader.extract_dssp(temp, replica)

# Process multiple domains in parallel
results = process_domains(domain_ids, data_dir, config, num_cores)
```
</details>

<details>
<summary><b>📦 PDB Processing API</b></summary>

```python
from src.mdcath.processing import pdb

# Clean and save PDB data
pdb.save_pdb_file(pdb_string, output_path, config)

# Process PDB data for all domains
pdb_results = pdb.process_pdb_data(domain_results, config)

# Extract frames from coordinates
pdb.extract_frames(coords, resids, resnames, domain_id, output_dir, 
                  temperature, replica, config, rmsd_data, gyration_data)
```
</details>

<details>
<summary><b>📦 RMSF Analysis API</b></summary>

```python
from src.mdcath.processing import rmsf

# Calculate average RMSF across replicas
replica_avg = rmsf.calculate_replica_averages(rmsf_data, temperature)

# Calculate average RMSF across temperatures
temp_avg = rmsf.calculate_temperature_average(replica_averages)

# Process all RMSF data
rmsf_results = rmsf.process_rmsf_data(domain_results, config)
```
</details>

<details>
<summary><b>📦 Structure Analysis API</b></summary>

```python
from src.mdcath.processing import core_exterior

# Classify residues as core or exterior
ce_data = core_exterior.compute_core_exterior(pdb_file, config)

# Run DSSP analysis
dssp_data = core_exterior.run_dssp_analysis(pdb_file)

# Get cached DSSP results
dssp_results = core_exterior.collect_dssp_data(pdb_file, domain_id, temp, replica)
```
</details>

<details>
<summary><b>📦 ML Features API</b></summary>

```python
from src.mdcath.processing import features

# Generate ML features
feature_dfs = features.generate_ml_features(rmsf_data, core_exterior_data, dssp_data, config)

# Save features to CSV
features.save_ml_features(feature_dfs, output_dir)

# Process all ML features
ml_results = features.process_ml_features(rmsf_results, pdb_results, domain_results, config)
```
</details>

<details>
<summary><b>📦 Visualization API</b></summary>

```python
from src.mdcath.processing import visualization

# Generate individual visualizations
visualization.create_temperature_summary_heatmap(replica_averages, output_dir)
visualization.create_rmsf_distribution_plots(replica_averages, output_dir)
visualization.create_amino_acid_rmsf_plot({"average": temperature_average}, output_dir)

# Generate all visualizations
vis_results = visualization.generate_visualizations(rmsf_results, ml_results, domain_results, config)
```
</details>

<details>
<summary><b>📦 Voxelization API</b></summary>

```python
from src.mdcath.processing import voxelizer

# Voxelize protein structures
voxel_results = voxelizer.voxelize_domains(pdb_results, config)
```
</details>
</div>

## 🐞 Troubleshooting

### Common Issues

<div style="display: grid; grid-template-columns: 1fr; grid-gap: 15px; margin-bottom: 30px;">

<!-- H5 File Validation Errors -->
<div style="background-color: #fff8f0; padding: 15px; border-radius: 8px; border-left: 5px solid #e74c3c;">
<h4>❌ H5 File Validation Errors</h4>

<div style="background-color: #f8d7da; padding: 10px; border-radius: 5px; color: #721c24; font-family: monospace; font-size: 0.85em;">
ERROR: Failed to validate H5 file for domain 12asA00
</div>

<h5>💡 Solution:</h5>
<ul>
  <li>Ensure the H5 file has the expected structure</li>
  <li>Check that the domain_id is correct and exists in the H5 file</li>
  <li>Validate that temperature and replica data are present</li>
</ul>

```python
# Manually check H5 file structure
import h5py
with h5py.File('/path/to/mdcath_dataset_12asA00.h5', 'r') as f:
    print(list(f.keys()))  # Should contain the domain ID
    domain = f['12asA00']  # Access domain group
    print(list(domain.keys()))  # Should contain temperature groups
```
</div>

<!-- DSSP Processing Failures -->
<div style="background-color: #fff8f0; padding: 15px; border-radius: 8px; border-left: 5px solid #e67e22;">
<h4>⚠️ DSSP Processing Failures</h4>

<div style="background-color: #fff3cd; padding: 10px; border-radius: 5px; color: #856404; font-family: monospace; font-size: 0.85em;">
WARNING: DSSP failed for domain 12asA00, using fallback
</div>

<h5>💡 Solution:</h5>
<ul>
  <li>Ensure DSSP is installed (mkdssp or dssp command should be available)</li>
  <li>Check that PDB files have proper CRYST1 records and atom formatting</li>
  <li>Verify PDBs have complete backbone atoms (N, CA, C)</li>
</ul>

```python
# Try manually running DSSP
import subprocess
result = subprocess.run(['mkdssp', 'outputs/pdbs/12asA00.pdb'], 
                      capture_output=True, text=True)
print(result.stdout)
print(result.stderr)
```
</div>

<!-- PDB Cleaning Errors -->
<div style="background-color: #fff8f0; padding: 15px; border-radius: 8px; border-left: 5px solid #e74c3c;">
<h4>❌ PDB Cleaning Errors</h4>

<div style="background-color: #f8d7da; padding: 10px; border-radius: 5px; color: #721c24; font-family: monospace; font-size: 0.85em;">
ERROR: Failed to clean PDB with pdbUtils: AttributeError: 'module' object has no attribute 'pdb2df'
</div>

<h5>💡 Solution:</h5>
<ul>
  <li>Install pdbUtils: <code>pip install pdbUtils</code></li>
  <li>Check that the PDB file is in a valid format</li>
  <li>If pdbUtils fails, the code will use the fallback method</li>
</ul>
</div>

<!-- Memory Errors -->
<div style="background-color: #fff8f0; padding: 15px; border-radius: 8px; border-left: 5px solid #e74c3c;">
<h4>❌ Memory Errors with Large Datasets</h4>

<div style="background-color: #f8d7da; padding: 10px; border-radius: 5px; color: #721c24; font-family: monospace; font-size: 0.85em;">
MemoryError: Unable to allocate array with shape (10000, 10000, 3)
</div>

<h5>💡 Solution:</h5>
<ul>
  <li>Adjust batch size to process fewer domains at once</li>
  <li>Set memory limits in the configuration</li>
  <li>Process domains in smaller batches</li>
</ul>

```yaml
performance:
  batch_size: 20  # Reduce batch size
  memory_limit_gb: 16  # Set memory limit
```
</div>
</div>

### Debugging Tips

<div style="background-color: #f0f8ff; padding: 15px; border-radius: 8px; margin: 20px 0;">
<h4 style="margin-top: 0;">🔍 Advanced Debugging Techniques</h4>

<ol>
  <li><b>Check logs</b>: Examine <code>mdcath_processing.log</code> for detailed error messages</li>
  
  <li><b>Enable verbose logging</b>:
  <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; font-family: monospace; font-size: 0.85em; margin: 10px 0;">
  logging:<br>
  &nbsp;&nbsp;verbose: true<br>
  &nbsp;&nbsp;level: "DEBUG"<br>
  &nbsp;&nbsp;console_level: "DEBUG"
  </div>
  </li>
  
  <li><b>Test individual components</b>:
  <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; font-family: monospace; font-size: 0.85em; margin: 10px 0;">
  # Test data loading<br>
  from src.mdcath.core.data_loader import H5DataLoader<br>
  loader = H5DataLoader("/path/to/mdcath_dataset_12asA00.h5", config)<br>
  print(loader._validate_h5())  # Should return True if file is valid<br>
  <br>
  # Test PDB cleaning<br>
  from src.mdcath.processing import pdb<br>
  pdb_data = loader.extract_pdb()<br>
  success = pdb.save_pdb_file(pdb_data, "test.pdb", config)<br>
  print(f"PDB cleaning success: {success}")
  </div>
  </li>
  
  <li><b>Process a single domain first</b>:
  <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; font-family: monospace; font-size: 0.85em; margin: 10px 0;">
  python main.py --domain_ids 12asA00 --output ./test_output
  </div>
  </li>
</ol>
</div>

## 🤝 Contributing

Contributions to the mdCATH processor are welcome! Here's how to get started:

<div style="display: grid; grid-template-columns: 1fr 1fr; grid-gap: 20px; margin-bottom: 30px;">

<!-- How to Contribute -->
<div style="background-color: #f0fff7; padding: 15px; border-radius: 8px; border-left: 5px solid #2ecc71;">
<h3>🚀 How to Contribute</h3>

<ol>
  <li>Fork the repository</li>
  <li>Create a feature branch: <code>git checkout -b feature/amazing-feature</code></li>
  <li>Make your changes</li>
  <li>Test your changes thoroughly</li>
  <li>Commit your changes: <code>git commit -m 'Add amazing feature'</code></li>
  <li>Push to the branch: <code>git push origin feature/amazing-feature</code></li>
  <li>Open a Pull Request</li>
</ol>
</div>

<!-- Development Setup -->
<div style="background-color: #f0f7ff; padding: 15px; border-radius: 8px; border-left: 5px solid #3498db;">
<h3>⚙️ Development Setup</h3>

```bash
# Clone repository
git clone https://github.com/Felixburton7/mdcath-processor.git
cd mdcath-processor

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development tools
pip install pytest pytest-cov black flake8

# Create a branch
git checkout -b feature/my-feature
```
</div>
</div>

### Code Style and Testing

We recommend following these guidelines:

- Use Black for code formatting
- Use type hints wherever possible
- Write comprehensive docstrings
- Add tests for new functionality
- Ensure all existing tests pass

## 📜 License

<div style="background-color: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0; border-left: 5px solid #17a2b8;">

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 mdCATH Processing Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```
</div>

---

<p align="center">
  <a href="https://github.com/Felixburton7/mdcath-processor">
    <img src="https://img.shields.io/badge/✨%20Star%20This%20Repo-If%20Useful-blue?style=for-the-badge" alt="Star This Repo">
  </a>
</p>

<p align="center">
  <em>Developed with ❤️ for the scientific community.</em>
</p>
