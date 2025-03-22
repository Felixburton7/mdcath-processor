

#!/usr/bin/env python3
"""
Core functionality for loading and processing H5 data from mdCATH dataset.
"""

import os
import h5py
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

class H5DataLoader:
    """
    Class for efficiently loading and extracting data from mdCATH H5 files.
    Uses chunking/streaming to handle large files.
    """

    def __init__(self, h5_path: str, config: Dict[str, Any]):
        """
        Initialize the H5 data loader.

        Args:
            h5_path: Path to H5 file
            config: Configuration dictionary
        """
        self.h5_path = h5_path
        self.config = config
        self.domain_id = os.path.basename(h5_path).replace("mdcath_dataset_", "").replace(".h5", "")
        valid = self._validate_h5()
        if valid:
            logging.info(f"Successfully validated H5 file for domain {self.domain_id}")
        else:
            logging.error(f"H5 file validation failed for domain {self.domain_id}")

    def _validate_h5(self) -> bool:
        """
        Validate that the H5 file has the expected structure.
        Enhanced with detailed logging and verification.
        
        Returns:
            Boolean indicating if the file is valid
        """
        try:
            with h5py.File(self.h5_path, 'r') as f:
                # Check if domain exists
                if self.domain_id not in f:
                    logging.error(f"Domain {self.domain_id} not found in {self.h5_path}")
                    return False
                
                # Log domain attributes
                domain_attrs = f[self.domain_id].attrs
                logging.info(f"Domain: {self.domain_id}")
                logging.info(f"Number of chains: {domain_attrs.get('numChains', 'N/A')}")
                logging.info(f"Number of protein atoms: {domain_attrs.get('numProteinAtoms', 'N/A')}")
                logging.info(f"Number of residues: {domain_attrs.get('numResidues', 'N/A')}")
                
                # Log atomic number information if available
                if 'z' in f[self.domain_id]:
                    z_data = f[self.domain_id]['z']
                    logging.info(f"z.shape: {z_data.shape}")
                    logging.info(f"First 10 z values: {z_data[:10]}")
                
                # Verify CA count matches numResidues if PDB data is available
                if 'pdbProteinAtoms' in f[self.domain_id]:
                    try:
                        pdb_data = f[self.domain_id]['pdbProteinAtoms'][()].decode('utf-8').split('\n')
                        ca_count = sum(1 for line in pdb_data if line.startswith('ATOM') and ' CA ' in line)
                        logging.info(f"Number of CA atoms in PDB: {ca_count}")
                        
                        if 'numResidues' in domain_attrs and ca_count != domain_attrs['numResidues']:
                            logging.warning(f"CA count ({ca_count}) does not match numResidues "
                                        f"({domain_attrs['numResidues']})")
                    except Exception as e:
                        logging.warning(f"Could not verify CA atom count: {e}")
                
                # Check for required metadata fields
                required_metadata = ["resid", "resname"]
                for field in required_metadata:
                    if field not in f[self.domain_id]:
                        logging.error(f"Required metadata field '{field}' not found for domain {self.domain_id}")
                        return False
                
                # Check for required temperature groups
                temps = [str(t) for t in self.config.get("temperatures", [320, 348, 379, 413, 450])]
                num_replicas = self.config.get("num_replicas", 5)
                
                temp_found = False
                for temp in temps:
                    if temp in f[self.domain_id]:
                        temp_found = True
                        logging.info(f"Found temperature group: {temp}K")
                        # Check for replica groups
                        replicas_found = 0
                        for r in range(num_replicas):
                            replica = str(r)
                            if replica in f[self.domain_id][temp]:
                                replicas_found += 1
                                # Check for specific datasets
                                # Removed DSSP from required datasets
                                required_datasets = ['rmsf', 'coords']
                                datasets_found = []
                                for dataset in required_datasets:
                                    if dataset in f[self.domain_id][temp][replica]:
                                        datasets_found.append(dataset)
                                    else:
                                        logging.warning(f"Dataset {dataset} not found for temperature {temp}, " 
                                                    f"replica {replica} in domain {self.domain_id}")
                                
                                # Also check for gyrationRadius and rmsd for frame selection
                                optional_datasets = ['gyrationRadius', 'rmsd']
                                for dataset in optional_datasets:
                                    if dataset in f[self.domain_id][temp][replica]:
                                        datasets_found.append(dataset)
                                
                                logging.info(f"Replica {replica} has datasets: {', '.join(datasets_found)}")
                            else:
                                logging.warning(f"Replica {replica} not found for temperature {temp} in domain {self.domain_id}")
                        
                        logging.info(f"Found {replicas_found}/{num_replicas} replicas for temperature {temp}K")
                    else:
                        logging.warning(f"Temperature {temp} not found for domain {self.domain_id}")
                
                if not temp_found:
                    logging.error(f"No valid temperature groups found for domain {self.domain_id}")
                    return False
                    
                return True
        except Exception as e:
            logging.error(f"Failed to validate H5 file {self.h5_path}: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return False

    def extract_rmsf(self, temperature: str, replica: str) -> Optional[pd.DataFrame]:
        """
        Extract RMSF data for a specific temperature and replica.
        RMSF is per-residue, so we build a unique residue-level list
        from the full 'resid'/'resname' arrays (which may be per-atom).
        
        Args:
            temperature: Temperature (e.g., "320")
            replica: Replica (e.g., "0")
        
        Returns:
            DataFrame with columns: [domain_id, resid, resname, rmsf_{temperature}]
            or None if extraction fails
        """
        try:
            with h5py.File(self.h5_path, 'r') as f:
                # Check if temperature and replica exist
                if (temperature not in f[self.domain_id]) or (replica not in f[self.domain_id][temperature]):
                    logging.warning(f"Temperature {temperature} or replica {replica} not found for domain {self.domain_id}")
                    return None
                
                if 'rmsf' not in f[self.domain_id][temperature][replica]:
                    logging.warning(f"RMSF data not found for domain {self.domain_id}, temperature {temperature}, replica {replica}")
                    return None
                
                # RMSF data is typically length = number_of_residues
                rmsf_data = f[self.domain_id][temperature][replica]['rmsf'][:]
                logging.info(f"Extracted RMSF data with shape {rmsf_data.shape} for domain {self.domain_id}, "
                            f"temperature {temperature}, replica {replica}")

                # Extract the full, per-atom arrays
                resids_all = f[self.domain_id]['resid'][:]
                resnames_all = f[self.domain_id]['resname'][:]

                # Convert bytes -> string if needed
                resnames_all = [
                    rn.decode("utf-8") if isinstance(rn, bytes) else str(rn)
                    for rn in resnames_all
                ]

                # Build unique residue-level list
                # Map resid -> resname (the first occurrence of that resid)
                # This ensures one row per residue
                residue_dict = {}
                for i, resid_val in enumerate(resids_all):
                    if resid_val not in residue_dict:
                        residue_dict[resid_val] = resnames_all[i]

                unique_resids = sorted(residue_dict.keys())
                unique_resnames = [residue_dict[rid] for rid in unique_resids]
                
                logging.info(f"Found {len(unique_resids)} unique residues for domain {self.domain_id}")

                # Check dimension mismatch
                if len(unique_resids) != len(rmsf_data):
                    logging.info(
                        f"Dimension mismatch: unique_resids {len(unique_resids)}, "
                        f"rmsf_data {len(rmsf_data)}"
                    )
                    # Attempt to align by length
                    if len(unique_resids) > len(rmsf_data):
                        logging.warning(
                            f"More unique residues ({len(unique_resids)}) than RMSF points ({len(rmsf_data)}) -- truncating residues"
                        )
                        unique_resids = unique_resids[:len(rmsf_data)]
                        unique_resnames = unique_resnames[:len(rmsf_data)]
                    else:
                        logging.warning(
                            f"Fewer unique residues ({len(unique_resids)}) than RMSF points ({len(rmsf_data)}) -- truncating RMSF"
                        )
                        rmsf_data = rmsf_data[:len(unique_resids)]
                    logging.info("Using unique residue-level alignment for RMSF data")

                # Create DataFrame with final 1:1 alignment
                df = pd.DataFrame({
                    'domain_id': self.domain_id,
                    'resid': unique_resids,
                    'resname': unique_resnames,
                    f'rmsf_{temperature}': rmsf_data
                })
                
                logging.info(f"Created RMSF DataFrame with {len(df)} rows")
                return df
        except Exception as e:
            logging.error(f"Failed to extract RMSF data: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return None

    def extract_pdb(self) -> Optional[str]:
        """
        Extract PDB data from the H5 file.

        Returns:
            PDB string or None if extraction fails
        """
        try:
            with h5py.File(self.h5_path, 'r') as f:
                if 'pdb' not in f[self.domain_id]:
                    logging.error(f"PDB data not found for domain {self.domain_id}")
                    return None
                    
                pdb_data = f[self.domain_id]['pdb'][()]
                if isinstance(pdb_data, bytes):
                    pdb_str = pdb_data.decode('utf-8')
                else:
                    pdb_str = str(pdb_data)
                    
                # Log PDB extraction
                num_lines = pdb_str.count('\n') + 1
                logging.info(f"Extracted PDB data with {num_lines} lines for domain {self.domain_id}")
                
                return pdb_str
        except Exception as e:
            logging.error(f"Failed to extract PDB data: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return None

    def extract_dssp(self, temperature: str, replica: str, frame: int = -1) -> Optional[pd.DataFrame]:
        """
        Extract DSSP data for a specific temperature, replica, and frame.
        DSSP is per-residue, so we build a unique residue-level list
        from the full 'resid'/'resname' arrays. Then align to DSSP codes.
        
        Args:
            temperature: Temperature (e.g., "320")
            replica: Replica (e.g., "0")
            frame: Frame index (default: -1 for last frame)

        Returns:
            DataFrame [domain_id, resid, resname, dssp] or None if extraction fails
        """
        try:
            with h5py.File(self.h5_path, 'r') as f:
                if (temperature not in f[self.domain_id]) or (replica not in f[self.domain_id][temperature]):
                    logging.warning(f"Temperature {temperature} or replica {replica} not found for domain {self.domain_id}")
                    return None

                if 'dssp' not in f[self.domain_id][temperature][replica]:
                    logging.warning(f"DSSP data not found for domain {self.domain_id}, temperature {temperature}, replica {replica}")
                    return None
                    
                dssp_dataset = f[self.domain_id][temperature][replica]['dssp']

                # Number of frames
                num_frames = dssp_dataset.shape[0] if len(dssp_dataset.shape) > 0 else 0
                if num_frames == 0:
                    logging.warning(f"Empty DSSP dataset for domain {self.domain_id}, temperature {temperature}, replica {replica}")
                    return None

                # Convert negative frame index
                if frame < 0:
                    frame = num_frames + frame
                if frame < 0 or frame >= num_frames:
                    logging.warning(f"Frame index {frame} out of bounds (0-{num_frames-1}) for {self.domain_id}")
                    frame = max(0, min(frame, num_frames-1))  # clamp

                dssp_data = dssp_dataset[frame]

                # Full, per-atom arrays
                resids_all = f[self.domain_id]['resid'][:]
                resnames_all = f[self.domain_id]['resname'][:]
                resnames_all = [
                    rn.decode("utf-8") if isinstance(rn, bytes) else str(rn)
                    for rn in resnames_all
                ]

                # Build unique residue-level list
                residue_dict = {}
                for i, resid_val in enumerate(resids_all):
                    if resid_val not in residue_dict:
                        residue_dict[resid_val] = resnames_all[i]

                unique_resids = sorted(residue_dict.keys())
                unique_resnames = [residue_dict[rid] for rid in unique_resids]

                # DSSP codes might already be length = # of residues
                dssp_codes = [
                    c.decode("utf-8") if isinstance(c, bytes) else str(c)
                    for c in dssp_data
                ]

                if len(unique_resids) != len(dssp_codes):
                    logging.info(
                        f"Dimension mismatch in DSSP: unique_resids {len(unique_resids)}, dssp_codes {len(dssp_codes)}"
                    )
                    if len(unique_resids) > len(dssp_codes):
                        logging.warning(
                            f"More unique residues ({len(unique_resids)}) than DSSP codes ({len(dssp_codes)}) -- truncating residues"
                        )
                        unique_resids = unique_resids[:len(dssp_codes)]
                        unique_resnames = unique_resnames[:len(dssp_codes)]
                    else:
                        logging.warning(
                            f"Fewer unique residues ({len(unique_resids)}) than DSSP codes ({len(dssp_codes)}) -- truncating DSSP codes"
                        )
                        dssp_codes = dssp_codes[:len(unique_resids)]
                    logging.info("Using unique residue-level alignment for DSSP data")

                # Create final DataFrame
                df = pd.DataFrame({
                    'domain_id': self.domain_id,
                    'resid': unique_resids,
                    'resname': unique_resnames,
                    'dssp': dssp_codes
                })

                return df

        except Exception as e:
            logging.error(f"Failed to extract DSSP data: {e}")
            return None


    def extract_coordinates(self, temperature: str, replica: str, frame: int = -1) -> Optional[Tuple[np.ndarray, List[int], List[str], np.ndarray, np.ndarray]]:
        """
        Extract coordinate data for a specific temperature, replica, and frame.
        Now includes additional data for frame selection (RMSD and gyration radius)

        Args:
            temperature: Temperature (e.g., "320")
            replica: Replica (e.g., "0")
            frame: Frame index (default: -1 for last frame, -999 for all frames)

        Returns:
            Tuple of (coords, resids, resnames, rmsd_data, gyration_data) where:
            - coords shape is (n_atoms, 3) or (n_frames, n_atoms, 3) if all frames requested
            - rmsd_data shape is (n_frames,)
            - gyration_data shape is (n_frames,)
            or None if extraction fails.
        """
        try:
            with h5py.File(self.h5_path, 'r') as f:
                # Enhanced validation and logging
                logging.info(f"Extracting coordinates for domain: {self.domain_id}")
                logging.info(f"Domain attributes: numChains={f[self.domain_id].attrs.get('numChains', 'N/A')}, "
                            f"numProteinAtoms={f[self.domain_id].attrs.get('numProteinAtoms', 'N/A')}, "
                            f"numResidues={f[self.domain_id].attrs.get('numResidues', 'N/A')}")
                
                # Log atomic numbers (z) information
                if 'z' in f[self.domain_id]:
                    z_data = f[self.domain_id]['z']
                    logging.info(f"z.shape: {z_data.shape}")
                    logging.info(f"First 10 z values: {z_data[:10]}")
                
                # Count CA atoms from PDB data to verify consistency
                if 'pdbProteinAtoms' in f[self.domain_id]:
                    pdb_data = f[self.domain_id]['pdbProteinAtoms'][()].decode('utf-8').split('\n')
                    ca_count = sum(1 for line in pdb_data if line.startswith('ATOM') and ' CA ' in line)
                    logging.info(f"Number of CA atoms in PDB: {ca_count}")
                    if 'numResidues' in f[self.domain_id].attrs:
                        if ca_count != f[self.domain_id].attrs['numResidues']:
                            logging.warning(f"CA count ({ca_count}) does not match numResidues "
                                        f"({f[self.domain_id].attrs['numResidues']})")
                
                # Check temperature and replica existence
                if (temperature not in f[self.domain_id]) or (replica not in f[self.domain_id][temperature]):
                    logging.warning(f"Temperature {temperature} or replica {replica} not found for domain {self.domain_id}")
                    return None

                # Check coordinates, RMSD and gyration radius availability
                req_datasets = ['coords']
                opt_datasets = ['rmsd', 'gyrationRadius']
                for dataset in req_datasets:
                    if dataset not in f[self.domain_id][temperature][replica]:
                        logging.warning(f"{dataset} data not found for domain {self.domain_id}, "
                                    f"temperature {temperature}, replica {replica}")
                        return None

                # Extract coordinates
                coords_dataset = f[self.domain_id][temperature][replica]['coords']
                num_frames = coords_dataset.shape[0] if coords_dataset.ndim > 0 else 0
                
                if num_frames == 0:
                    logging.warning(f"Empty coords dataset for domain {self.domain_id}, "
                                f"temperature {temperature}, replica {replica}")
                    return None

                # Extract RMSD and gyration radius data for frame selection if available
                rmsd_data = None
                gyration_data = None
                
                if 'rmsd' in f[self.domain_id][temperature][replica]:
                    rmsd_data = f[self.domain_id][temperature][replica]['rmsd'][:]
                else:
                    logging.info(f"RMSD data not available for {self.domain_id}, {temperature}, {replica}")
                    rmsd_data = np.zeros(num_frames)
                    
                if 'gyrationRadius' in f[self.domain_id][temperature][replica]:
                    gyration_data = f[self.domain_id][temperature][replica]['gyrationRadius'][:]
                else:
                    logging.info(f"Gyration radius data not available for {self.domain_id}, {temperature}, {replica}")
                    gyration_data = np.zeros(num_frames)
                
                logging.info(f"Available frames: {num_frames}, RMSD shape: {rmsd_data.shape if rmsd_data is not None else 'N/A'}, "
                            f"Gyration shape: {gyration_data.shape if gyration_data is not None else 'N/A'}")

                # Handle negative frame index or return all frames if requested
                if frame == -999:  # Special value to request all frames
                    coords = coords_dataset[:]
                    logging.info(f"Extracting all {num_frames} frames")
                else:
                    # Convert negative frame index
                    if frame < 0:
                        frame = num_frames + frame
                    if frame < 0 or frame >= num_frames:
                        logging.warning(f"Frame index {frame} out of bounds (0-{num_frames-1}) for domain {self.domain_id}")
                        frame = max(0, min(frame, num_frames - 1))
                    logging.info(f"Extracting single frame {frame}")
                    coords = coords_dataset[frame]  # shape (n_atoms, 3)
                    
                    # Validate coordinate shape
                    if coords.ndim != 2 or coords.shape[1] != 3:
                        logging.error(f"Unexpected coordinate shape: {coords.shape} for domain {self.domain_id}")
                        return None

                # Extract residue information
                resids_all = f[self.domain_id]['resid'][:].tolist()
                resnames_all = f[self.domain_id]['resname'][:]
                resnames_all = [
                    rn.decode("utf-8") if isinstance(rn, bytes) else str(rn)
                    for rn in resnames_all
                ]

                # Check shape alignment for single frame
                coord_atoms = coords.shape[0] if coords.ndim == 2 else coords.shape[1]
                if len(resids_all) != coord_atoms:
                    logging.warning(f"Mismatch between residue IDs ({len(resids_all)}) and "
                                f"coords ({coord_atoms})")
                    min_size = min(len(resids_all), coord_atoms)
                    resids_all = resids_all[:min_size]
                    resnames_all = resnames_all[:min_size]
                    if coords.ndim == 3:  # Multiple frames
                        coords = coords[:, :min_size, :]
                    else:  # Single frame
                        coords = coords[:min_size]

                logging.info(f"Successfully extracted coordinates with shape {coords.shape}")
                return coords, resids_all, resnames_all, rmsd_data, gyration_data

        except Exception as e:
            logging.error(f"Failed to extract coordinate data: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return None

def process_domains(domain_ids: List[str], data_dir: str, config: Dict[str, Any],
                    num_cores: int = 1) -> Dict[str, Any]:
    """
    Process multiple domains in parallel.
    Updated to remove DSSP extraction.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    max_cores = os.cpu_count() - 2 if os.cpu_count() > 2 else 1
    n_cores = min(num_cores if num_cores > 0 else max_cores, max_cores)

    results = {}
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        future_to_domain = {}
        for domain_id in domain_ids:
            h5_path = os.path.join(data_dir, f"mdcath_dataset_{domain_id}.h5")
            if not os.path.exists(h5_path):
                logging.warning(f"H5 file not found for domain {domain_id}")
                continue

            future = executor.submit(_process_single_domain, h5_path, config)
            future_to_domain[future] = domain_id

        for future in as_completed(future_to_domain):
            domain_id = future_to_domain[future]
            try:
                result = future.result()
                results[domain_id] = result
            except Exception as e:
                logging.error(f"Error processing domain {domain_id}: {e}")
                results[domain_id] = {"success": False, "error": str(e)}

    return results

def _process_single_domain(h5_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single domain (helper function for parallel processing).
    Updated to remove DSSP extraction.
    """
    loader = H5DataLoader(h5_path, config)
    domain_id = loader.domain_id

    results = {"domain_id": domain_id, "success": False}

    # Extract RMSF data
    temps = [str(t) for t in config.get("temperatures", [320, 348, 379, 413, 450])]
    num_replicas = config.get("num_replicas", 5)

    rmsf_data = {}
    for temp in temps:
        rmsf_data[temp] = {}
        for r in range(num_replicas):
            replica = str(r)
            df_rmsf = loader.extract_rmsf(temp, replica)
            if df_rmsf is not None:
                rmsf_data[temp][replica] = df_rmsf
    results["rmsf_data"] = rmsf_data

    # Extract PDB data
    pdb_str = loader.extract_pdb()
    if pdb_str:
        results["pdb_data"] = pdb_str


    results["success"] = True
    return results