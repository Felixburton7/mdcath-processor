
#!/usr/bin/env python3
"""
Processing module for generating ML features.
"""

import os
import logging
import shutil
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple, Union
from tqdm import tqdm

from src.mdcath.processing.core_exterior import compute_core_exterior, collect_dssp_data


def generate_ml_features(rmsf_data: Dict[str, pd.DataFrame],
                         core_exterior_data: Dict[str, pd.DataFrame],
                         dssp_data: Dict[str, Dict[str, pd.DataFrame]],
                         config: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """
    Generate ML features for all domains with improved handling of missing values.
    """
    try:
        # Get list of all domains
        domain_ids = set()
        for temp, df in rmsf_data.items():
            domain_ids.update(df["domain_id"].unique())

        domain_ids = list(domain_ids)
        logging.info(f"Generating ML features for {len(domain_ids)} domains")

        # Create feature dataframes for each temperature
        temps = [t for t in rmsf_data.keys() if t != "average"]
        feature_dfs = {}

        for temp in temps:
            # Start with RMSF data
            if temp not in rmsf_data:
                logging.warning(f"RMSF data not found for temperature {temp}")
                continue

            df = rmsf_data[temp].copy()

            # Ensure RMSF column is numeric
            rmsf_col = f"rmsf_{temp}"
            if rmsf_col in df.columns:
                df[rmsf_col] = pd.to_numeric(df[rmsf_col], errors='coerce').fillna(0.0)

            # Add protein size (number of residues for each domain)
            df["protein_size"] = df.groupby("domain_id")["resid"].transform("count")

            # Add normalized residue position
            df["normalized_resid"] = df.groupby("domain_id")["resid"].transform(
                lambda x: (x - x.min()) / max(x.max() - x.min(), 1)
            )

            # Create core_exterior column with default value
            if "core_exterior" not in df.columns:
                df["core_exterior"] = "core"  # Default to core (more conservative)
                
            # Create empty columns for missing data with default values
            if "relative_accessibility" not in df.columns:
                df["relative_accessibility"] = 0.5  # Default to moderate accessibility

            if "dssp" not in df.columns:
                df["dssp"] = "C"  # Default to coil
                
            # Add default phi and psi columns if missing
            if "phi" not in df.columns:
                df["phi"] = 0.0  # Default phi angle
                
            if "psi" not in df.columns:
                df["psi"] = 0.0  # Default psi angle

            # ---------------------------
            # MERGE CORE/EXTERIOR DATA
            # ---------------------------
            for domain_id in df["domain_id"].unique():
                if domain_id in core_exterior_data:
                    core_ext_df = core_exterior_data[domain_id]
                    
                    # Use regular merge to avoid dimension mismatch
                    domain_df = df[df["domain_id"] == domain_id][["domain_id", "resid"]].copy()
                    merged = pd.merge(domain_df, core_ext_df, on="resid", how="left")
                    
                    # Create mappings from resid to values
                    ce_mapping = dict(zip(merged["resid"], merged["core_exterior"]))
                    
                    # Apply the mapping to the original dataframe
                    domain_mask = df["domain_id"] == domain_id
                    df.loc[domain_mask, "core_exterior"] = df.loc[domain_mask, "resid"].map(ce_mapping).fillna("core")
                    
                    # If core_exterior_data contains relative_accessibility, use it
                    if "relative_accessibility" in core_ext_df.columns:
                        ra_mapping = dict(zip(core_ext_df["resid"], core_ext_df["relative_accessibility"]))
                        df.loc[domain_mask, "relative_accessibility"] = df.loc[domain_mask, "resid"].map(ra_mapping).fillna(0.5)

            # Ensure no missing values in core_exterior
            df["core_exterior"] = df["core_exterior"].fillna("core")

            # ---------------------------
            # ADD DSSP DATA
            # ---------------------------
            if temp in dssp_data:
                for replica, replica_dssp in dssp_data[temp].items():
                    if not replica_dssp.empty:
                        for domain_id in df["domain_id"].unique():
                            domain_dssp = replica_dssp[replica_dssp["domain_id"] == domain_id]
                            if not domain_dssp.empty:
                                # Convert resid to numeric carefully
                                domain_dssp.loc[:, "resid"] = pd.to_numeric(domain_dssp["resid"], errors='coerce')
                                
                                # Create mappings
                                dssp_mapping = dict(zip(domain_dssp["resid"], domain_dssp["dssp"]))
                                
                                # Apply mappings
                                domain_mask = df["domain_id"] == domain_id
                                df.loc[domain_mask, "dssp"] = df.loc[domain_mask, "resid"].map(dssp_mapping).fillna("C")
                                
                                # Apply relative_accessibility if present
                                if "relative_accessibility" in domain_dssp.columns:
                                    ra_mapping = dict(zip(domain_dssp["resid"], domain_dssp["relative_accessibility"]))
                                    df.loc[domain_mask, "relative_accessibility"] = df.loc[domain_mask, "resid"].map(ra_mapping).fillna(0.5)
                                
                                # Apply phi angles if present
                                if "phi" in domain_dssp.columns:
                                    phi_mapping = dict(zip(domain_dssp["resid"], domain_dssp["phi"]))
                                    df.loc[domain_mask, "phi"] = df.loc[domain_mask, "resid"].map(phi_mapping).fillna(0.0)
                                
                                # Apply psi angles if present
                                if "psi" in domain_dssp.columns:
                                    psi_mapping = dict(zip(domain_dssp["resid"], domain_dssp["psi"]))
                                    df.loc[domain_mask, "psi"] = df.loc[domain_mask, "resid"].map(psi_mapping).fillna(0.0)
                                
                                # Break after first valid replica with data for this domain
                                break

            # Ensure no empty strings in DSSP
            df["dssp"] = df["dssp"].replace("", "C").replace(" ", "C").fillna("C")
            
            # Ensure relative_accessibility is numeric and not empty
            df["relative_accessibility"] = pd.to_numeric(df["relative_accessibility"], errors='coerce').fillna(0.5)
            
            # Ensure phi and psi are numeric and not empty
            df["phi"] = pd.to_numeric(df["phi"], errors='coerce').fillna(0.0)
            df["psi"] = pd.to_numeric(df["psi"], errors='coerce').fillna(0.0)

            # ---------------------------
            # ENCODE CATEGORICAL VARIABLES
            # ---------------------------
            # 1) Resname encoding: ensure all are strings
            if "resname" not in df.columns:
                # If resname is missing, create a placeholder
                df["resname"] = "UNK"

            # Convert to string and remove invalid placeholders
            df["resname"] = df["resname"].astype(str)
            filtered_resnames = [r for r in df["resname"].unique() if r not in ["nan", "None", ""]]
            unique_resnames = sorted(filtered_resnames)

            # Build mapping
            resname_mapping = {name: i+1 for i, name in enumerate(unique_resnames)}  # Start at 1
            df["resname_encoded"] = df["resname"].map(resname_mapping).fillna(0).astype(int)

            # 2) Core/Exterior encoding
            core_ext_mapping = {"core": 0, "exterior": 1, "unknown": 2}
            df["core_exterior_encoded"] = df["core_exterior"].map(core_ext_mapping).fillna(0).astype(int)

            # 3) DSSP encoding (3-state secondary structure)
            def encode_ss(ss):
                if ss in ["H", "G", "I"]:
                    return 0  # Helix
                elif ss in ["E", "B"]:
                    return 1  # Sheet
                else:
                    return 2  # Coil, Loop, or other

            df["secondary_structure_encoded"] = df["dssp"].apply(encode_ss)
            
            # 4) Add normalized phi/psi features (map to [-1, 1] range)
            df["phi_norm"] = df["phi"] / 180.0  # Normalize to [-1, 1] range
            df["psi_norm"] = df["psi"] / 180.0  # Normalize to [-1, 1] range

            # Log DSSP encoding distribution
            dssp_codes = df["dssp"].value_counts().to_dict()
            encoded_values = df["secondary_structure_encoded"].value_counts().to_dict()
            logging.info(f"DSSP distribution for temp {temp}: {dssp_codes}")
            logging.info(f"Encoded SS distribution for temp {temp}: {encoded_values}")

            # Reorder columns to put domain_id first
            cols = df.columns.tolist()
            if "domain_id" in cols:
                cols.remove("domain_id")
                df = df[["domain_id"] + cols]

            # Final validation - ensure no NaN or empty values
            for col in df.columns:
                if df[col].dtype == 'object':
                    # For string columns, fill empty strings and NaNs with appropriate defaults
                    if col == 'dssp':
                        df[col] = df[col].replace('', 'C').replace(' ', 'C').fillna('C')
                    elif col == 'core_exterior':
                        df[col] = df[col].replace('', 'core').fillna('core')
                    elif col == 'resname':
                        df[col] = df[col].replace('', 'UNK').fillna('UNK')
                    else:
                        df[col] = df[col].fillna('unknown')
                else:
                    # For numeric columns, fill NaNs with appropriate defaults
                    if col == 'relative_accessibility':
                        df[col] = df[col].fillna(0.5)
                    elif col in ['phi', 'psi', 'phi_norm', 'psi_norm']:
                        df[col] = df[col].fillna(0.0)
                    else:
                        df[col] = df[col].fillna(0)

            # Store the feature dataframe
            feature_dfs[temp] = df

        # --------------------------------
        # CALCULATE AVERAGE FEATURES ACROSS TEMPS
        # --------------------------------
        if temps:
            avg_df = feature_dfs[temps[0]].copy(deep=True)

            # Calculate average RMSF across temperatures
            rmsf_cols = [f"rmsf_{temp}" for temp in temps]
            if all(col in avg_df.columns for col in rmsf_cols):
                avg_df["rmsf_average"] = avg_df[rmsf_cols].mean(axis=1)
            else:
                # If missing some temperature data
                available_cols = [col for col in rmsf_cols if col in avg_df.columns]
                if available_cols:
                    avg_df["rmsf_average"] = avg_df[available_cols].mean(axis=1)
                else:
                    # No RMSF data available
                    avg_df["rmsf_average"] = 0.0

            feature_dfs["average"] = avg_df

        return feature_dfs

    except Exception as e:
        logging.error(f"Failed to generate ML features: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {}
    
def save_ml_features(feature_dfs: Dict[str, pd.DataFrame], output_dir: str) -> bool:
    """
    Save ML features to CSV files.

    Args:
        feature_dfs: Dictionary with ML feature dataframes
        output_dir: Directory to save CSV files

    Returns:
        Boolean indicating if saving was successful
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

        for temp, df in feature_dfs.items():
            if temp == "average":
                output_file = os.path.join(output_dir, "final_dataset_temperature_average.csv")
            else:
                output_file = os.path.join(output_dir, f"final_dataset_temperature_{temp}.csv")

            df.to_csv(output_file, index=False)
            logging.info(f"Saved ML features to {output_file}")

        return True
    except Exception as e:
        logging.error(f"Failed to save ML features: {e}")
        return False


def process_ml_features(rmsf_results: Dict[str, Any],
                        pdb_results: Dict[str, Any],
                        domain_results: Dict[str, Dict[str, Any]],
                        config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process ML features for all domains.
    """
    output_dir = config.get("output", {}).get("base_dir", "./outputs")

    # Extract RMSF data
    replica_averages = rmsf_results.get("replica_averages", {})
    temperature_average = rmsf_results.get("temperature_average")

    if not replica_averages:
        logging.error("No RMSF data available for ML feature generation")
        return {"success": False, "error": "No RMSF data available"}

    # Create dictionary with all RMSF data
    rmsf_data = replica_averages.copy()
    if temperature_average is not None:
        rmsf_data["average"] = temperature_average

    # Compute core/exterior data
    core_exterior_data = {}
    logging.info("Computing core/exterior classification for domains")
    for domain_id, result in tqdm(pdb_results.items(), desc="Core/exterior classification"):
        if not result.get("pdb_saved", False):
            continue

        pdb_path = result.get("pdb_path")
        if not pdb_path or not os.path.exists(pdb_path):
            logging.warning(f"PDB file not found for domain {domain_id}")
            continue

        core_ext_df = compute_core_exterior(pdb_path, config)
        if core_ext_df is not None:
            core_exterior_data[domain_id] = core_ext_df

    # Collect DSSP data - MODIFIED TO USE CACHED DSSP RESULTS
    dssp_data = {}
    temps = [str(t) for t in config.get("temperatures", [320, 348, 379, 413, 450])]
    replica = "0"  # Use first replica by default

    logging.info("Collecting DSSP data with secondary structure and torsion angles")
    for domain_id, result in tqdm(pdb_results.items(), desc="Processing DSSP data"):
        if not result.get("pdb_saved", False):
            continue

        pdb_path = result.get("pdb_path")
        if not pdb_path or not os.path.exists(pdb_path):
            logging.warning(f"PDB file not found for domain {domain_id}")
            continue

        # For each temperature, collect DSSP data
        for temp in temps:
            if temp not in dssp_data:
                dssp_data[temp] = {}
            
            if replica not in dssp_data[temp]:
                dssp_data[temp][replica] = []
            
            # Use the cached DSSP function
            dssp_df = collect_dssp_data(pdb_path, domain_id, temp, replica)
            if not dssp_df.empty:
                dssp_data[temp][replica].append(dssp_df)

    # Concatenate DSSP dataframes
    logging.info("Concatenating DSSP data")
    for temp in temps:
        if temp in dssp_data:
            for r in dssp_data[temp]:
                if dssp_data[temp][r]:
                    dssp_data[temp][r] = pd.concat(dssp_data[temp][r], ignore_index=True)
                    logging.info(f"DSSP data for temp {temp}, replica {r}: {len(dssp_data[temp][r])} rows")
                    
                    # Log phi/psi angle statistics
                    if 'phi' in dssp_data[temp][r].columns and 'psi' in dssp_data[temp][r].columns:
                        phi_stats = dssp_data[temp][r]['phi'].describe()
                        psi_stats = dssp_data[temp][r]['psi'].describe()
                        logging.info(f"Phi angle stats: mean={phi_stats['mean']:.2f}, std={phi_stats['std']:.2f}")
                        logging.info(f"Psi angle stats: mean={psi_stats['mean']:.2f}, std={psi_stats['std']:.2f}")

    # Generate ML features
    logging.info("Generating ML features with torsion angles")
    feature_dfs = generate_ml_features(rmsf_data, core_exterior_data, dssp_data, config)

    if not feature_dfs:
        logging.error("Failed to generate ML features")
        return {"success": False, "error": "Feature generation failed"}

    # Save ML features
    ml_dir = os.path.join(output_dir, "ML_features")
    save_success = save_ml_features(feature_dfs, ml_dir)

    return {
        "success": save_success,
        "feature_dfs": feature_dfs,
        "output_dir": ml_dir
    }

# #!/usr/bin/env python3
# """
# Processing module for generating ML features.
# """

# import os
# import logging
# import shutil  
# import numpy as np
# import pandas as pd
# from typing import Dict, Any, Optional, List, Tuple, Union
# from tqdm import tqdm

# from src.mdcath.processing.core_exterior import compute_core_exterior, collect_dssp_data


# def generate_ml_features(rmsf_data: Dict[str, pd.DataFrame],
#                          core_exterior_data: Dict[str, pd.DataFrame],
#                          dssp_data: Dict[str, Dict[str, pd.DataFrame]],
#                          config: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
#     """
#     Generate ML features for all domains with improved handling of missing values.
#     """
#     try:
#         # Get list of all domains
#         domain_ids = set()
#         for temp, df in rmsf_data.items():
#             domain_ids.update(df["domain_id"].unique())

#         domain_ids = list(domain_ids)
#         logging.info(f"Generating ML features for {len(domain_ids)} domains")

#         # Create feature dataframes for each temperature
#         temps = [t for t in rmsf_data.keys() if t != "average"]
#         feature_dfs = {}

#         for temp in temps:
#             # Start with RMSF data
#             if temp not in rmsf_data:
#                 logging.warning(f"RMSF data not found for temperature {temp}")
#                 continue

#             df = rmsf_data[temp].copy()

#             # Ensure RMSF column is numeric
#             rmsf_col = f"rmsf_{temp}"
#             if rmsf_col in df.columns:
#                 df[rmsf_col] = pd.to_numeric(df[rmsf_col], errors='coerce').fillna(0.0)

#             # Add protein size (number of residues for each domain)
#             df["protein_size"] = df.groupby("domain_id")["resid"].transform("count")

#             # Add normalized residue position
#             df["normalized_resid"] = df.groupby("domain_id")["resid"].transform(
#                 lambda x: (x - x.min()) / max(x.max() - x.min(), 1)
#             )

#             # Create core_exterior column with default value
#             if "core_exterior" not in df.columns:
#                 df["core_exterior"] = "core"  # Default to core (more conservative)
                
#             # Create empty columns for missing data with default values
#             if "relative_accessibility" not in df.columns:
#                 df["relative_accessibility"] = 0.5  # Default to moderate accessibility

#             if "dssp" not in df.columns:
#                 df["dssp"] = "C"  # Default to coil

#             # ---------------------------
#             # MERGE CORE/EXTERIOR DATA
#             # ---------------------------
#             for domain_id in df["domain_id"].unique():
#                 if domain_id in core_exterior_data:
#                     core_ext_df = core_exterior_data[domain_id]
                    
#                     # Use regular merge to avoid dimension mismatch
#                     domain_df = df[df["domain_id"] == domain_id][["domain_id", "resid"]].copy()
#                     merged = pd.merge(domain_df, core_ext_df, on="resid", how="left")
                    
#                     # Create mappings from resid to values
#                     ce_mapping = dict(zip(merged["resid"], merged["core_exterior"]))
                    
#                     # Apply the mapping to the original dataframe
#                     domain_mask = df["domain_id"] == domain_id
#                     df.loc[domain_mask, "core_exterior"] = df.loc[domain_mask, "resid"].map(ce_mapping).fillna("core")
                    
#                     # If core_exterior_data contains relative_accessibility, use it
#                     if "relative_accessibility" in core_ext_df.columns:
#                         ra_mapping = dict(zip(core_ext_df["resid"], core_ext_df["relative_accessibility"]))
#                         df.loc[domain_mask, "relative_accessibility"] = df.loc[domain_mask, "resid"].map(ra_mapping).fillna(0.5)

#             # Ensure no missing values in core_exterior
#             df["core_exterior"] = df["core_exterior"].fillna("core")

#             # ---------------------------
#             # ADD DSSP DATA
#             # ---------------------------
#             if temp in dssp_data:
#                 for replica, replica_dssp in dssp_data[temp].items():
#                     if not replica_dssp.empty:
#                         for domain_id in df["domain_id"].unique():
#                             domain_dssp = replica_dssp[replica_dssp["domain_id"] == domain_id]
#                             if not domain_dssp.empty:
#                                 # Convert resid to numeric carefully
#                                 domain_dssp.loc[:, "resid"] = pd.to_numeric(domain_dssp["resid"], errors='coerce')
                                
#                                 # Create mappings
#                                 dssp_mapping = dict(zip(domain_dssp["resid"], domain_dssp["dssp"]))
                                
#                                 # Apply mappings
#                                 domain_mask = df["domain_id"] == domain_id
#                                 df.loc[domain_mask, "dssp"] = df.loc[domain_mask, "resid"].map(dssp_mapping).fillna("C")
                                
#                                 # If relative_accessibility is present, use it
#                                 if "relative_accessibility" in domain_dssp.columns:
#                                     ra_mapping = dict(zip(domain_dssp["resid"], domain_dssp["relative_accessibility"]))
#                                     df.loc[domain_mask, "relative_accessibility"] = df.loc[domain_mask, "resid"].map(ra_mapping).fillna(0.5)
                                
#                                 # Break after first valid replica with data for this domain
#                                 break

#             # Ensure no empty strings in DSSP
#             df["dssp"] = df["dssp"].replace("", "C").replace(" ", "C").fillna("C")
            
#             # Ensure relative_accessibility is numeric and not empty
#             df["relative_accessibility"] = pd.to_numeric(df["relative_accessibility"], errors='coerce').fillna(0.5)

#             # ---------------------------
#             # ENCODE CATEGORICAL VARIABLES
#             # ---------------------------
#             # 1) Resname encoding: ensure all are strings
#             if "resname" not in df.columns:
#                 # If resname is missing, create a placeholder
#                 df["resname"] = "UNK"

#             # Convert to string and remove invalid placeholders
#             df["resname"] = df["resname"].astype(str)
#             filtered_resnames = [r for r in df["resname"].unique() if r not in ["nan", "None", ""]]
#             unique_resnames = sorted(filtered_resnames)

#             # Build mapping
#             resname_mapping = {name: i+1 for i, name in enumerate(unique_resnames)}  # Start at 1
#             df["resname_encoded"] = df["resname"].map(resname_mapping).fillna(0).astype(int)

#             # 2) Core/Exterior encoding
#             core_ext_mapping = {"core": 0, "exterior": 1, "unknown": 2}
#             df["core_exterior_encoded"] = df["core_exterior"].map(core_ext_mapping).fillna(0).astype(int)

#             # 3) DSSP encoding (3-state secondary structure)
#             def encode_ss(ss):
#                 if ss in ["H", "G", "I"]:
#                     return 0  # Helix
#                 elif ss in ["E", "B"]:
#                     return 1  # Sheet
#                 else:
#                     return 2  # Coil, Loop, or other

#             df["secondary_structure_encoded"] = df["dssp"].apply(encode_ss)

#             # Log DSSP encoding distribution
#             dssp_codes = df["dssp"].value_counts().to_dict()
#             encoded_values = df["secondary_structure_encoded"].value_counts().to_dict()
#             logging.info(f"DSSP distribution for temp {temp}: {dssp_codes}")
#             logging.info(f"Encoded SS distribution for temp {temp}: {encoded_values}")

#             # Reorder columns to put domain_id first
#             cols = df.columns.tolist()
#             if "domain_id" in cols:
#                 cols.remove("domain_id")
#                 df = df[["domain_id"] + cols]

#             # Final validation - ensure no NaN or empty values
#             for col in df.columns:
#                 if df[col].dtype == 'object':
#                     # For string columns, fill empty strings and NaNs with appropriate defaults
#                     if col == 'dssp':
#                         df[col] = df[col].replace('', 'C').replace(' ', 'C').fillna('C')
#                     elif col == 'core_exterior':
#                         df[col] = df[col].replace('', 'core').fillna('core')
#                     elif col == 'resname':
#                         df[col] = df[col].replace('', 'UNK').fillna('UNK')
#                     else:
#                         df[col] = df[col].fillna('unknown')
#                 else:
#                     # For numeric columns, fill NaNs with appropriate defaults
#                     if col == 'relative_accessibility':
#                         df[col] = df[col].fillna(0.5)
#                     else:
#                         df[col] = df[col].fillna(0)

#             # Store the feature dataframe
#             feature_dfs[temp] = df

#         # --------------------------------
#         # CALCULATE AVERAGE FEATURES ACROSS TEMPS
#         # --------------------------------
#         if temps:
#             avg_df = feature_dfs[temps[0]].copy(deep=True)

#             # Calculate average RMSF across temperatures
#             rmsf_cols = [f"rmsf_{temp}" for temp in temps]
#             if all(col in avg_df.columns for col in rmsf_cols):
#                 avg_df["rmsf_average"] = avg_df[rmsf_cols].mean(axis=1)
#             else:
#                 # If missing some temperature data
#                 available_cols = [col for col in rmsf_cols if col in avg_df.columns]
#                 if available_cols:
#                     avg_df["rmsf_average"] = avg_df[available_cols].mean(axis=1)
#                 else:
#                     # No RMSF data available
#                     avg_df["rmsf_average"] = 0.0

#             feature_dfs["average"] = avg_df

#         return feature_dfs

#     except Exception as e:
#         logging.error(f"Failed to generate ML features: {e}")
#         import traceback
#         logging.error(traceback.format_exc())
#         return {}
    
# def save_ml_features(feature_dfs: Dict[str, pd.DataFrame], output_dir: str) -> bool:
#     """
#     Save ML features to CSV files.

#     Args:
#         feature_dfs: Dictionary with ML feature dataframes
#         output_dir: Directory to save CSV files

#     Returns:
#         Boolean indicating if saving was successful
#     """
#     try:
#         os.makedirs(output_dir, exist_ok=True)

#         for temp, df in feature_dfs.items():
#             if temp == "average":
#                 output_file = os.path.join(output_dir, "final_dataset_temperature_average.csv")
#             else:
#                 output_file = os.path.join(output_dir, f"final_dataset_temperature_{temp}.csv")

#             df.to_csv(output_file, index=False)
#             logging.info(f"Saved ML features to {output_file}")

#         return True
#     except Exception as e:
#         logging.error(f"Failed to save ML features: {e}")
#         return False


# def process_ml_features(rmsf_results: Dict[str, Any],
#                         pdb_results: Dict[str, Any],
#                         domain_results: Dict[str, Dict[str, Any]],
#                         config: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     Process ML features for all domains.
#     """
#     output_dir = config.get("output", {}).get("base_dir", "./outputs")

#     # Extract RMSF data
#     replica_averages = rmsf_results.get("replica_averages", {})
#     temperature_average = rmsf_results.get("temperature_average")

#     if not replica_averages:
#         logging.error("No RMSF data available for ML feature generation")
#         return {"success": False, "error": "No RMSF data available"}

#     # Create dictionary with all RMSF data
#     rmsf_data = replica_averages.copy()
#     if temperature_average is not None:
#         rmsf_data["average"] = temperature_average

#     # Compute core/exterior data
#     core_exterior_data = {}
#     logging.info("Computing core/exterior classification for domains")
#     for domain_id, result in tqdm(pdb_results.items(), desc="Core/exterior classification"):
#         if not result.get("pdb_saved", False):
#             continue

#         pdb_path = result.get("pdb_path")
#         if not pdb_path or not os.path.exists(pdb_path):
#             logging.warning(f"PDB file not found for domain {domain_id}")
#             continue

#         core_ext_df = compute_core_exterior(pdb_path, config)
#         if core_ext_df is not None:
#             core_exterior_data[domain_id] = core_ext_df

#     # Collect DSSP data - MODIFIED TO USE CACHED DSSP RESULTS
#     dssp_data = {}
#     temps = [str(t) for t in config.get("temperatures", [320, 348, 379, 413, 450])]
#     replica = "0"  # Use first replica by default

#     logging.info("Collecting DSSP data with correct secondary structure")
#     for domain_id, result in tqdm(pdb_results.items(), desc="Processing DSSP data"):
#         if not result.get("pdb_saved", False):
#             continue

#         pdb_path = result.get("pdb_path")
#         if not pdb_path or not os.path.exists(pdb_path):
#             logging.warning(f"PDB file not found for domain {domain_id}")
#             continue

#         # For each temperature, collect DSSP data
#         for temp in temps:
#             if temp not in dssp_data:
#                 dssp_data[temp] = {}
            
#             if replica not in dssp_data[temp]:
#                 dssp_data[temp][replica] = []
            
#             # Use the cached DSSP function
#             dssp_df = collect_dssp_data(pdb_path, domain_id, temp, replica)
#             if not dssp_df.empty:
#                 dssp_data[temp][replica].append(dssp_df)

#     # Concatenate DSSP dataframes
#     logging.info("Concatenating DSSP data")
#     for temp in temps:
#         if temp in dssp_data:
#             for r in dssp_data[temp]:
#                 if dssp_data[temp][r]:
#                     dssp_data[temp][r] = pd.concat(dssp_data[temp][r], ignore_index=True)
#                     logging.info(f"DSSP data for temp {temp}, replica {r}: {len(dssp_data[temp][r])} rows, "
#                                f"secondary structure: {dssp_data[temp][r]['dssp'].value_counts().to_dict()}")

#     # Generate ML features
#     logging.info("Generating ML features")
#     feature_dfs = generate_ml_features(rmsf_data, core_exterior_data, dssp_data, config)

#     if not feature_dfs:
#         logging.error("Failed to generate ML features")
#         return {"success": False, "error": "Feature generation failed"}

#     # Save ML features
#     ml_dir = os.path.join(output_dir, "ML_features")
#     save_success = save_ml_features(feature_dfs, ml_dir)

#     return {
#         "success": save_success,
#         "feature_dfs": feature_dfs,
#         "output_dir": ml_dir
#     }