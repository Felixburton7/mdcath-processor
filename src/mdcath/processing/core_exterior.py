#!/usr/bin/env python3
"""
Processing module for core/exterior classification and secondary structure assignment.
Uses Biopython's DSSP for both tasks, with optimized sharing of results.
"""

import os
import logging
import subprocess
import tempfile
import pandas as pd
import numpy as np
import shutil
from Bio.PDB import PDBParser, DSSP, ShrakeRupley
from typing import Dict, Any, Optional, List, Tuple, Union
from functools import lru_cache

# Global cache for DSSP results to avoid redundant processing
_dssp_cache = {}

def compute_core_exterior(pdb_file: str, config: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """
    Classify residues as 'core' or 'exterior' based on solvent accessibility.

    Args:
        pdb_file: Path to the cleaned PDB file
        config: Configuration dictionary

    Returns:
        DataFrame with columns 'resid' and 'core_exterior' or None if classification fails
    """
    # We'll use Biopython's DSSP directly now
    return compute_core_exterior_biopython(pdb_file, config)

def prepare_pdb_for_dssp(pdb_file: str) -> Optional[str]:
    """
    Prepare a PDB file for DSSP processing by ensuring it has a proper CRYST1 record
    and correctly formatted atom names.
    
    Args:
        pdb_file: Path to the PDB file
        
    Returns:
        Path to the temporary PDB file or None if preparation fails
    """
    try:
        # Verify file exists
        abs_pdb_file = os.path.abspath(pdb_file)
        if not os.path.exists(abs_pdb_file):
            logging.error(f"PDB file not found: {abs_pdb_file}")
            return None
            
        # Read original PDB file
        with open(abs_pdb_file, 'r') as f:
            lines = f.readlines()
        
        # Fix PDB format issues
        corrected_lines = []
        has_cryst1 = False
        
        for line in lines:
            if line.startswith("CRYST1"):
                has_cryst1 = True
                # Ensure properly formatted CRYST1 record
                corrected_lines.append("CRYST1  100.000  100.000  100.000  90.00  90.00  90.00 P 1           1\n")
            else:
                # Fix atom name formatting which can cause DSSP issues
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    # Ensure atom names are properly padded
                    if len(line) >= 16:
                        atom_name = line[12:16].strip()
                        # DSSP requires atom names to be properly spaced
                        # Left-justify atom names starting with C, N, O, S, P
                        # Right-justify other atom names
                        if atom_name and atom_name[0] in "CNOSP":
                            padded_atom = f"{atom_name:<4}"
                        else:
                            padded_atom = f"{atom_name:>4}"
                        line = line[:12] + padded_atom + line[16:]
                corrected_lines.append(line)
        
        # Add CRYST1 if missing
        if not has_cryst1:
            corrected_lines.insert(0, "CRYST1  100.000  100.000  100.000  90.00  90.00  90.00 P 1           1\n")
        
        # Write to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdb", mode="w") as tmp:
            tmp_pdb = tmp.name
            tmp.writelines(corrected_lines)
            
        return tmp_pdb
    
    except Exception as e:
        logging.error(f"Failed to prepare PDB for DSSP: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None

@lru_cache(maxsize=128)
def run_dssp_once(pdb_file: str) -> Optional[Dict]:
    """
    Run DSSP on a PDB file once and cache the results.
    This unified function handles both secondary structure and accessibility.
    
    Args:
        pdb_file: Path to the PDB file
        
    Returns:
        Dictionary with DSSP information or None if DSSP fails
    """
    # Prepare PDB file
    tmp_pdb = prepare_pdb_for_dssp(pdb_file)
    if not tmp_pdb:
        return None
    
    try:
        # Extract domain_id from filename
        domain_id = os.path.basename(pdb_file).split('.')[0]
        
        # Parse structure
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", tmp_pdb)
        model = structure[0]
        
        # Check for required backbone atoms
        has_backbone = False
        for chain in model:
            for res in chain:
                if res.has_id("CA") and res.has_id("C") and res.has_id("N"):
                    has_backbone = True
                    break
            if has_backbone:
                break
        
        if not has_backbone:
            logging.warning(f"Model lacks complete backbone atoms for {domain_id}, DSSP will likely fail")
        
        # Try to find and run DSSP
        dssp_found = False
        dssp_obj = None
        
        for dssp_exec in ["dssp", "mkdssp"]:
            dssp_path = shutil.which(dssp_exec)
            if not dssp_path:
                continue
                
            try:
                logging.info(f"Running DSSP using {dssp_path} on {domain_id}")
                dssp_obj = DSSP(model, tmp_pdb, dssp=dssp_path)
                
                # Validate results
                if len(dssp_obj) == 0:
                    logging.warning(f"DSSP returned empty results for {domain_id}")
                    continue
                
                # Check if we got valid secondary structure assignments
                ss_counts = {}
                for key in dssp_obj.keys():
                    ss = dssp_obj[key][2]  # Secondary structure code
                    if ss not in ss_counts:
                        ss_counts[ss] = 0
                    ss_counts[ss] += 1
                
                logging.info(f"DSSP results for {domain_id}: {ss_counts}")
                
                # Success
                dssp_found = True
                break
                
            except Exception as e:
                logging.warning(f"DSSP ({dssp_exec}) failed on {domain_id}: {e}")
                continue
        
        # If DSSP failed through Biopython, try direct execution as last resort
        if not dssp_found and dssp_path:
            try:
                logging.info(f"Attempting direct DSSP call for {domain_id}")
                result = subprocess.run([dssp_path, tmp_pdb], capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    # Parse DSSP output manually
                    logging.info(f"Direct DSSP call succeeded for {domain_id}")
                    
                    # Create a simplified DSSP-like object from output
                    dssp_data = {}
                    for line in result.stdout.splitlines():
                        # DSSP output format has the secondary structure at position 16
                        # and residue info starting at position 5
                        if not line.startswith("#") and len(line) > 16:
                            try:
                                chain_id = line[11]
                                res_num = int(line[5:10].strip())
                                ss_code = line[16]
                                if ss_code == " ":
                                    ss_code = "C"  # Default to coil
                                
                                # Get accessibility
                                acc_str = line[34:38].strip()
                                acc = float(acc_str) if acc_str else 0.0
                                # Normalize to 0-1 scale
                                rel_acc = min(1.0, acc / 100.0)
                                
                                # Extract phi and psi angles (columns 103-109 and 109-115 in DSSP output)
                                # These positions might vary slightly in different DSSP versions
                                try:
                                    phi_str = line[103:109].strip()
                                    psi_str = line[109:115].strip()
                                    phi = float(phi_str) if phi_str else 0.0
                                    psi = float(psi_str) if psi_str else 0.0
                                except (ValueError, IndexError):
                                    phi, psi = 0.0, 0.0
                                
                                key = (chain_id, (' ', res_num, ' '))
                                dssp_data[key] = {
                                    'ss': ss_code,
                                    'acc': rel_acc,
                                    'phi': phi,
                                    'psi': psi
                                }
                            except (ValueError, IndexError) as e:
                                continue
                    
                    if dssp_data:
                        logging.info(f"Manually parsed {len(dssp_data)} residues from DSSP output for {domain_id}")
                        return {
                            'domain_id': domain_id,
                            'dssp_data': dssp_data,
                            'method': 'manual'
                        }
            except Exception as e:
                logging.warning(f"Direct DSSP execution failed for {domain_id}: {e}")
        
        # Return results if we have them
        if dssp_found and dssp_obj:
            return {
                'domain_id': domain_id,
                'dssp_obj': dssp_obj,
                'method': 'biopython'
            }
        
        # If we get here, DSSP failed
        logging.warning(f"All DSSP methods failed for {domain_id}")
        return None
        
    except Exception as e:
        logging.error(f"DSSP processing error: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None
        
    finally:
        # Clean up temp file
        if tmp_pdb and os.path.exists(tmp_pdb):
            try:
                os.remove(tmp_pdb)
            except Exception as e:
                logging.warning(f"Failed to remove temporary file {tmp_pdb}: {e}")

def collect_dssp_data(pdb_file: str, domain_id: str, temp: str, replica: str = "0") -> pd.DataFrame:
    """
    Collect DSSP data for a specific domain, temperature and replica.
    Uses cached DSSP results to avoid redundant processing.
    
    Args:
        pdb_file: Path to the PDB file
        domain_id: Domain identifier
        temp: Temperature
        replica: Replica index
        
    Returns:
        DataFrame with DSSP data
    """
    # Get DSSP results from cache
    dssp_results = run_dssp_once(pdb_file)
    
    if not dssp_results:
        # Return fallback data if DSSP failed
        logging.warning(f"Using fallback DSSP data for {domain_id} at {temp}K (rep {replica})")
        return use_fallback_dssp(pdb_file)
    
    # Process based on how DSSP was run
    if dssp_results['method'] == 'biopython':
        dssp_obj = dssp_results['dssp_obj']
        
        # Extract secondary structure, accessibility, and torsion angles
        records = []
        for key in dssp_obj.keys():
            chain_id = key[0]
            resid = key[1][1]  # residue number
            dssp_tuple = dssp_obj[key]
            
            # Extract data
            ss_code = dssp_tuple[2]  # Secondary structure code
            rel_acc = dssp_tuple[3]  # Relative accessibility
            phi = dssp_tuple[4]  # Phi angle
            psi = dssp_tuple[5]  # Psi angle
            
            # Ensure secondary structure is never empty
            if not ss_code or ss_code == ' ' or ss_code == '-':
                ss_code = 'C'  # Default to coil
            
            records.append({
                "domain_id": domain_id,
                "resid": resid,
                "chain": chain_id,
                "dssp": ss_code,
                "relative_accessibility": rel_acc,
                "phi": phi,
                "psi": psi
            })
        
        if records:
            df = pd.DataFrame(records)
            logging.info(f"DSSP secondary structure for {domain_id}: {df['dssp'].value_counts().to_dict()}")
            return df
            
    elif dssp_results['method'] == 'manual':
        # Process manual DSSP results
        dssp_data = dssp_results['dssp_data']
        records = []
        
        for key, data in dssp_data.items():
            chain_id = key[0]
            resid = key[1][1]  # residue number
            ss_code = data['ss']
            rel_acc = data['acc']
            phi = data.get('phi', 0.0)  # Get phi angle if available
            psi = data.get('psi', 0.0)  # Get psi angle if available
            
            records.append({
                "domain_id": domain_id,
                "resid": resid,
                "chain": chain_id,
                "dssp": ss_code,
                "relative_accessibility": rel_acc,
                "phi": phi,
                "psi": psi
            })
        
        if records:
            df = pd.DataFrame(records)
            logging.info(f"DSSP secondary structure for {domain_id}: {df['dssp'].value_counts().to_dict()}")
            return df
    
    # If we get here, use fallback
    return use_fallback_dssp(pdb_file)

def compute_core_exterior_biopython(pdb_file: str, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Use DSSP to classify residues as 'core' or 'exterior' based on accessibility.
    Uses cached DSSP results if available.

    Args:
        pdb_file: Path to the cleaned PDB file
        config: Configuration dictionary

    Returns:
        DataFrame with columns 'resid' and 'core_exterior'
    """
    domain_id = os.path.basename(pdb_file).split('.')[0]
    sasa_threshold = config.get("core_exterior", {}).get("sasa_threshold", 20.0)
    
    try:
        # Get DSSP results (from cache if available)
        dssp_results = run_dssp_once(pdb_file)
        
        if dssp_results:
            # Process based on how DSSP was run
            if dssp_results['method'] == 'biopython':
                dssp_obj = dssp_results['dssp_obj']
                
                # Extract accessibility and classify as core/exterior
                results = []
                for key in dssp_obj.keys():
                    chain_id = key[0]
                    resid = key[1][1]  # residue number
                    dssp_tuple = dssp_obj[key]
                    
                    # Get relative accessibility
                    rel_acc = dssp_tuple[3]  # Relative accessibility (0-1 scale)
                    
                    # Classify based on threshold (20% is typical cutoff)
                    core_exterior = "exterior" if rel_acc > 0.2 else "core"
                    
                    results.append({
                        "resid": resid,
                        "chain": chain_id,
                        "core_exterior": core_exterior,
                        "relative_accessibility": rel_acc
                    })
                
                if results:
                    logging.info(f"Successfully classified {len(results)} residues using DSSP for {domain_id}")
                    return pd.DataFrame(results)
            
            elif dssp_results['method'] == 'manual':
                # Process manual DSSP results
                dssp_data = dssp_results['dssp_data']
                results = []
                
                for key, data in dssp_data.items():
                    chain_id = key[0]
                    resid = key[1][1]  # residue number
                    rel_acc = data['acc']
                    
                    # Classify based on threshold
                    core_exterior = "exterior" if rel_acc > 0.2 else "core"
                    
                    results.append({
                        "resid": resid,
                        "chain": chain_id,
                        "core_exterior": core_exterior,
                        "relative_accessibility": rel_acc
                    })
                
                if results:
                    logging.info(f"Successfully classified {len(results)} residues using manual DSSP for {domain_id}")
                    return pd.DataFrame(results)
        
        # If DSSP failed, try ShrakeRupley
        logging.info(f"DSSP failed for {domain_id}, using ShrakeRupley SASA")
        
        # Parse structure
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_file)
        model = structure[0]
        
        # Calculate SASA
        sr = ShrakeRupley()
        sr.compute(model, level="R")  # Compute at residue level
        
        # Extract results
        results = []
        for chain in model:
            for residue in chain:
                if residue.id[0] == " ":  # Standard residue
                    resid = residue.id[1]
                    sasa = residue.sasa if hasattr(residue, 'sasa') else 0.0
                    
                    # Normalize SASA
                    rel_acc = min(1.0, sasa / 100.0)
                    core_exterior = "exterior" if sasa > sasa_threshold else "core"
                    
                    results.append({
                        "resid": resid,
                        "chain": chain.id,
                        "core_exterior": core_exterior,
                        "relative_accessibility": rel_acc
                    })
        
        if results:
            logging.info(f"Successfully classified {len(results)} residues using ShrakeRupley for {domain_id}")
            return pd.DataFrame(results)
        
        # Final fallback
        logging.warning(f"All methods failed, using fallback classification for {domain_id}")
        return fallback_core_exterior(pdb_file)
        
    except Exception as e:
        logging.error(f"Core/exterior classification failed: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return fallback_core_exterior(pdb_file)

def fallback_core_exterior(pdb_file: str) -> pd.DataFrame:
    """
    Fallback method to classify residues when other methods fail.
    Classifies outer 1/3 of residues as exterior, inner 2/3 as core.

    Args:
        pdb_file: Path to the cleaned PDB file

    Returns:
        DataFrame with columns 'resid' and 'core_exterior'
    """
    try:
        # Verify file exists and use absolute path
        abs_pdb_file = os.path.abspath(pdb_file)
        if not os.path.exists(abs_pdb_file):
            logging.error(f"PDB file not found: {abs_pdb_file}")
            # Create dummy data when PDB file is missing
            return pd.DataFrame({
                "resid": list(range(1, 21)),  # Create 20 dummy residues
                "core_exterior": ["core"] * 13 + ["exterior"] * 7,  # 2/3 core, 1/3 exterior
                "relative_accessibility": [0.1] * 13 + [0.7] * 7  # Low for core, high for exterior
            })

        # Parse PDB to get residue information
        residue_df = parse_pdb_residues(pdb_file)
        if residue_df.empty:
            # Create empty DataFrame with required columns
            return pd.DataFrame({
                "resid": list(range(1, 21)),
                "core_exterior": ["core"] * 13 + ["exterior"] * 7,
                "relative_accessibility": [0.1] * 13 + [0.7] * 7
            })

        # Sort by residue ID
        residue_df = residue_df.sort_values("resid")

        # Simple classification: outer 1/3 of residues as exterior, inner 2/3 as core
        total_residues = len(residue_df)
        boundary = int(total_residues * 2/3)

        residue_df["core_exterior"] = ["core"] * total_residues
        residue_df.loc[boundary:, "core_exterior"] = "exterior"
        
        # Add relative accessibility values (0-1 scale)
        residue_df["relative_accessibility"] = 0.1  # Default for core
        residue_df.loc[boundary:, "relative_accessibility"] = 0.7  # Higher for exterior

        return residue_df[["resid", "core_exterior", "relative_accessibility"]]
    except Exception as e:
        logging.error(f"Fallback classification failed: {e}")
        return pd.DataFrame({
            "resid": list(range(1, 21)),
            "core_exterior": ["core"] * 13 + ["exterior"] * 7,
            "relative_accessibility": [0.1] * 13 + [0.7] * 7
        })

def parse_pdb_residues(pdb_file: str) -> pd.DataFrame:
    """
    Parse a PDB file to extract residue-level information.

    Args:
        pdb_file: Path to the PDB file

    Returns:
        DataFrame with residue information
    """
    try:
        from Bio.PDB import PDBParser

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_file)

        records = []
        for model in structure:
            for chain in model:
                chain_id = chain.id
                for residue in chain:
                    if residue.id[0] == " ":  # Standard residue
                        records.append({
                            "resid": residue.id[1],
                            "resname": residue.get_resname(),
                            "chain": chain_id
                        })

        return pd.DataFrame(records)
    except Exception as e:
        logging.error(f"Failed to parse PDB residues: {e}")
        return pd.DataFrame()

def parse_pdb_atoms(pdb_file: str) -> pd.DataFrame:
    """
    Parse a PDB file to extract atom-level information.

    Args:
        pdb_file: Path to the PDB file

    Returns:
        DataFrame with atom information
    """
    try:
        from Bio.PDB import PDBParser

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_file)

        records = []
        atom_idx = 0
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.id[0] == " ":  # Standard residue
                        res_id = residue.id[1]
                        res_name = residue.get_resname()
                        for atom in residue:
                            atom_idx += 1
                            records.append({
                                "atom_idx": atom_idx,
                                "resid": res_id,
                                "resname": res_name,
                                "atom_name": atom.get_name()
                            })

        return pd.DataFrame(records)
    except Exception as e:
        logging.error(f"Failed to parse PDB atoms: {e}")
        return pd.DataFrame()

def run_dssp_analysis(pdb_file: str) -> pd.DataFrame:
    """
    Run DSSP to get secondary structure assignments.
    Uses the cached DSSP results if available.
    
    Args:
        pdb_file: Path to the PDB file
        
    Returns:
        DataFrame with columns: domain_id, resid, chain, dssp, relative_accessibility, phi, psi
    """
    domain_id = os.path.basename(pdb_file).split('.')[0]
    
    try:
        # Get DSSP results (from cache if available)
        dssp_results = run_dssp_once(pdb_file)
        
        if dssp_results:
            # Process based on how DSSP was run
            if dssp_results['method'] == 'biopython':
                dssp_obj = dssp_results['dssp_obj']
                
                # Extract secondary structure, accessibility, and torsion angles
                records = []
                for key in dssp_obj.keys():
                    chain_id = key[0]
                    resid = key[1][1]  # residue number
                    dssp_tuple = dssp_obj[key]
                    
                    # Extract data
                    ss_code = dssp_tuple[2]  # Secondary structure code
                    rel_acc = dssp_tuple[3]  # Relative accessibility
                    phi = dssp_tuple[4]  # Phi angle
                    psi = dssp_tuple[5]  # Psi angle
                    
                    # Ensure secondary structure is never empty
                    if not ss_code or ss_code == ' ' or ss_code == '-':
                        ss_code = 'C'  # Default to coil
                    
                    records.append({
                        "domain_id": domain_id,
                        "resid": resid,
                        "chain": chain_id,
                        "dssp": ss_code,
                        "relative_accessibility": rel_acc,
                        "phi": phi,
                        "psi": psi
                    })
                
                if records:
                    df = pd.DataFrame(records)
                    logging.info(f"Successfully extracted DSSP data for {len(df)} residues in {domain_id}")
                    logging.info(f"DSSP codes distribution: {df['dssp'].value_counts().to_dict()}")
                    return df
            
            elif dssp_results['method'] == 'manual':
                # Process manual DSSP results
                dssp_data = dssp_results['dssp_data']
                records = []
                
                for key, data in dssp_data.items():
                    chain_id = key[0]
                    resid = key[1][1]  # residue number
                    ss_code = data['ss']
                    rel_acc = data['acc']
                    phi = data.get('phi', 0.0)  # Get phi angle if available
                    psi = data.get('psi', 0.0)  # Get psi angle if available
                    
                    records.append({
                        "domain_id": domain_id,
                        "resid": resid,
                        "chain": chain_id,
                        "dssp": ss_code,
                        "relative_accessibility": rel_acc,
                        "phi": phi,
                        "psi": psi
                    })
                
                if records:
                    df = pd.DataFrame(records)
                    logging.info(f"Successfully extracted manual DSSP data for {len(df)} residues in {domain_id}")
                    logging.info(f"DSSP codes distribution: {df['dssp'].value_counts().to_dict()}")
                    return df
        
        # If DSSP fails, use fallback
        logging.warning(f"DSSP failed for {domain_id}, using fallback")
        return use_fallback_dssp(pdb_file)
        
    except Exception as e:
        logging.error(f"Failed to run DSSP analysis for {domain_id}: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return use_fallback_dssp(pdb_file)

def use_fallback_dssp(pdb_file: str) -> pd.DataFrame:
    """
    Fallback method when DSSP fails.
    Provides default secondary structure and accessibility values.
    
    Args:
        pdb_file: Path to the PDB file
        
    Returns:
        DataFrame with columns: domain_id, resid, chain, dssp, relative_accessibility, phi, psi
    """
    # Extract domain_id from filename
    domain_id = os.path.basename(pdb_file).split('.')[0]
    
    logging.info(f"Using fallback secondary structure prediction for {domain_id}")
    
    try:
        # First check if the PDB file exists
        abs_pdb_file = os.path.abspath(pdb_file)
        if not os.path.exists(abs_pdb_file):
            # Create dummy data for missing PDB
            dummy_df = pd.DataFrame({
                "domain_id": [domain_id] * 20,
                "resid": list(range(1, 21)),  # 20 dummy residues
                "chain": ["A"] * 20,
                "dssp": ["C"] * 20,
                "relative_accessibility": [0.5] * 20,  # Medium accessibility
                "phi": [0.0] * 20,  # Default phi angle
                "psi": [0.0] * 20   # Default psi angle
            })
            logging.warning(f"PDB file not found for {domain_id}, using dummy data with {len(dummy_df)} residues")
            return dummy_df
        
        # Parse PDB to get residue info
        try:
            from Bio.PDB import PDBParser
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("protein", abs_pdb_file)
            
            records = []
            for model in structure:
                for chain in model:
                    chain_id = chain.id
                    for residue in chain:
                        if residue.id[0] == " ":  # Standard residue
                            resid = residue.id[1]
                            records.append({
                                "domain_id": domain_id,
                                "resid": resid,
                                "chain": chain_id,
                                "dssp": "C",  # Default to coil
                                "relative_accessibility": 0.5,  # Default to moderate accessibility
                                "phi": 0.0,  # Default phi angle
                                "psi": 0.0   # Default psi angle
                            })
            
            if records:
                result_df = pd.DataFrame(records)
                logging.info(f"Created fallback DSSP data for {domain_id} with {len(result_df)} residues")
                return result_df
        except Exception as e:
            logging.warning(f"Failed to parse PDB structure for {domain_id}: {e}")
        
        # If we get here, we couldn't parse the PDB, so create dummy data
        dummy_df = pd.DataFrame({
            "domain_id": [domain_id] * 20,
            "resid": list(range(1, 21)),
            "chain": ["A"] * 20,
            "dssp": ["C"] * 20,
            "relative_accessibility": [0.5] * 20,
            "phi": [0.0] * 20,
            "psi": [0.0] * 20
        })
        logging.warning(f"Failed to parse PDB for {domain_id}, using dummy data with {len(dummy_df)} residues")
        return dummy_df
        
    except Exception as e:
        logging.error(f"Fallback DSSP also failed for {domain_id}: {e}")
        import traceback
        logging.error(traceback.format_exc())
        
        # Return minimal dataframe with required columns
        dummy_df = pd.DataFrame({
            "domain_id": [domain_id] * 20,
            "resid": list(range(1, 21)),
            "chain": ["A"] * 20,
            "dssp": ["C"] * 20,
            "relative_accessibility": [0.5] * 20,
            "phi": [0.0] * 20,
            "psi": [0.0] * 20
        })
        logging.warning(f"Critical failure in DSSP processing for {domain_id}, using emergency dummy data")
        return dummy_df



# # #!/usr/bin/env python3
# # """
# # Processing module for core/exterior classification.
# # """

# # import os
# # import logging
# # import subprocess
# # import tempfile
# # import pandas as pd
# # import numpy as np
# # import Bio
# # import shutil
# # from Bio.PDB import PDBParser, DSSP, ShrakeRupley
# # from typing import Dict, Any, Optional, List, Tuple


# # def compute_core_exterior(pdb_file: str, config: Dict[str, Any]) -> Optional[pd.DataFrame]:
# #     """
# #     Classify residues as 'core' or 'exterior' based on solvent accessibility.

# #     Args:
# #         pdb_file: Path to the cleaned PDB file
# #         config: Configuration dictionary

# #     Returns:
# #         DataFrame with columns 'resid' and 'core_exterior' or None if classification fails
# #     """
# #     method = config.get("core_exterior", {}).get("method", "msms")

# #     if method == "msms":
# #         return compute_core_exterior_msms(pdb_file, config)
# #     else:
# #         return compute_core_exterior_biopython(pdb_file, config)

# # def compute_core_exterior_msms(pdb_file: str, config: Dict[str, Any]) -> Optional[pd.DataFrame]:
# #     """
# #     Use MSMS to classify residues as 'core' or 'exterior'.

# #     Args:
# #         pdb_file: Path to the cleaned PDB file
# #         config: Configuration dictionary

# #     Returns:
# #         DataFrame with columns 'resid' and 'core_exterior' or None if MSMS fails
# #     """
# #     msms_dir = config.get("core_exterior", {}).get("msms_executable_dir", "./msms_executables")
# #     # Convert to absolute path
# #     msms_dir = os.path.abspath(msms_dir)
# #     ses_threshold = config.get("core_exterior", {}).get("ses_threshold", 1.0)
# #     protein_name = os.path.basename(pdb_file).split('.')[0]

# #     try:
# #         # Create temporary directory for MSMS files
# #         with tempfile.TemporaryDirectory() as tmp_dir:
# #             # Paths to MSMS executables and output files
# #             pdb2xyzr_exe = os.path.join(msms_dir, "pdb_to_xyzr")
# #             msms_exe = os.path.join(msms_dir, "msms.x86_64Linux2.2.6.1")
# #             xyzr_file = os.path.join(tmp_dir, f"{protein_name}.xyzr")
# #             area_base = os.path.join(tmp_dir, f"{protein_name}")
# #             area_file = f"{area_base}.area"

# #             # Ensure executables have proper permissions
# #             try:
# #                 os.chmod(pdb2xyzr_exe, 0o755)  # rwxr-xr-x
# #                 os.chmod(msms_exe, 0o755)      # rwxr-xr-x
# #             except Exception as e:
# #                 logging.warning(f"Failed to set executable permissions: {e}")

# #             # Check MSMS executables
# #             if not os.path.exists(pdb2xyzr_exe) or not os.path.exists(msms_exe):
# #                 logging.warning(f"MSMS executables not found in {msms_dir}, falling back to Biopython")
# #                 return compute_core_exterior_biopython(pdb_file, config)

# #             # Absolute path for input PDB
# #             abs_pdb_file = os.path.abspath(pdb_file)
# #             if not os.path.exists(abs_pdb_file):
# #                 logging.warning(f"PDB file not found: {abs_pdb_file}")
# #                 return compute_core_exterior_biopython(pdb_file, config)

# #             # Run pdb_to_xyzr with bash shell explicitly
# #             cmd_xyzr = f"bash {pdb2xyzr_exe} {abs_pdb_file} > {xyzr_file}"
# #             logging.info(f"Running command: {cmd_xyzr}")
# #             result = subprocess.run(cmd_xyzr, shell=True, check=False,
# #                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# #             if result.returncode != 0 or not os.path.exists(xyzr_file) or os.path.getsize(xyzr_file) == 0:
# #                 logging.warning(f"pdb_to_xyzr failed: {result.stderr.decode()}, falling back to Biopython")
# #                 return compute_core_exterior_biopython(pdb_file, config)

# #             # Run MSMS with bash shell explicitly
# #             cmd_msms = f"bash {msms_exe} -if {xyzr_file} -af {area_base}"
# #             logging.info(f"Running command: {cmd_msms}")
# #             result = subprocess.run(cmd_msms, shell=True, check=False,
# #                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# #             if result.returncode != 0 or not os.path.exists(area_file):
# #                 logging.warning(f"MSMS failed: {result.stderr.decode()}, falling back to Biopython")
# #                 return compute_core_exterior_biopython(pdb_file, config)

# #             # Rest of the function unchanged...
# #             # Parse atom-level PDB data
# #             per_atom_df = parse_pdb_atoms(pdb_file)
# #             if per_atom_df.empty:
# #                 logging.warning(f"Failed to parse atoms from PDB, falling back to Biopython")
# #                 return compute_core_exterior_biopython(pdb_file, config)

# #             # Parse MSMS area file
# #             area_df = parse_area_file(area_file)
# #             if area_df.empty:
# #                 logging.warning(f"Failed to parse area file, falling back to Biopython")
# #                 return compute_core_exterior_biopython(pdb_file, config)

# #             # Combine atom data with MSMS results
# #             if len(area_df) != len(per_atom_df):
# #                 logging.warning(f"Atom count mismatch: {len(area_df)} vs {len(per_atom_df)}, falling back to Biopython")
# #                 return compute_core_exterior_biopython(pdb_file, config)

# #             # Merge data
# #             per_atom_df = pd.concat([per_atom_df.reset_index(drop=True),
# #                                     area_df.reset_index(drop=True)], axis=1)

# #             # Calculate mean SES per residue
# #             mean_ses_per_res = per_atom_df.groupby("resid")["SES"].mean()

# #             # Classify residues as core or exterior
# #             exterior_residues = mean_ses_per_res[mean_ses_per_res > ses_threshold].index
# #             resids = mean_ses_per_res.index.tolist()
# #             core_exterior = ["exterior" if r in exterior_residues else "core" for r in resids]

# #             # Create final dataframe
# #             result_df = pd.DataFrame({
# #                 "resid": resids,
# #                 "core_exterior": core_exterior
# #             })

# #             return result_df
# #     except Exception as e:
# #         logging.warning(f"MSMS processing failed: {e}, falling back to Biopython")
# #         return compute_core_exterior_biopython(pdb_file, config)
    

# # def compute_core_exterior_biopython(pdb_file: str, config: Dict[str, Any]) -> pd.DataFrame:
# #     """
# #     Use Biopython's SASA calculation to classify residues as 'core' or 'exterior'.

# #     Args:
# #         pdb_file: Path to the cleaned PDB file
# #         config: Configuration dictionary

# #     Returns:
# #         DataFrame with columns 'resid' and 'core_exterior'
# #     """
# #     try:
# #         from Bio.PDB import PDBParser, Selection
# #         from Bio.PDB.SASA import ShrakeRupley

# #         # Set SASA threshold
# #         sasa_threshold = config.get("core_exterior", {}).get("sasa_threshold", 20.0)

# #         # Parse PDB - ensure CRYST1 record first
# #         try:
# #             # Fix PDB file to ensure proper CRYST1 record
# #             corrected_lines = []
# #             with open(pdb_file, 'r') as f:
# #                 lines = f.readlines()
            
# #             has_cryst1 = False
# #             for line in lines:
# #                 if line.startswith("CRYST1"):
# #                     has_cryst1 = True
# #                     corrected_lines.append("CRYST1  100.000  100.000  100.000  90.00  90.00  90.00 P 1           1\n")
# #                 else:
# #                     corrected_lines.append(line)
            
# #             if not has_cryst1:
# #                 corrected_lines.insert(0, "CRYST1  100.000  100.000  100.000  90.00  90.00  90.00 P 1           1\n")
            
# #             # Write to temporary file
# #             with tempfile.NamedTemporaryFile(delete=False, suffix=".pdb", mode="w") as tmp:
# #                 tmp_pdb = tmp.name
# #                 tmp.writelines(corrected_lines)
            
# #             parser = PDBParser(QUIET=True)
# #             structure = parser.get_structure("protein", tmp_pdb)
# #             model = structure[0]
        
# #         except Exception as e:
# #             logging.warning(f"Failed to fix CRYST1 record: {e}")
# #             parser = PDBParser(QUIET=True)
# #             structure = parser.get_structure("protein", pdb_file)
# #             model = structure[0]
        
# #         # Try DSSP first for better solvent accessibility calculation
# #         try:
# #             # Get the location of the DSSP executable
# #             dssp_executable = shutil.which("dssp") or shutil.which("mkdssp")
# #             if dssp_executable:
# #                 logging.info(f"Using DSSP executable: {dssp_executable}")
                
# #                 # Write fixed PDB to temporary file
# #                 with tempfile.NamedTemporaryFile(delete=False, suffix=".pdb", mode="w") as tmp:
# #                     tmp_pdb = tmp.name
                    
# #                     # Add CRYST1 record if needed
# #                     with open(pdb_file, 'r') as f:
# #                         content = f.readlines()
                    
# #                     has_cryst1 = False
# #                     for i, line in enumerate(content):
# #                         if line.startswith("CRYST1"):
# #                             has_cryst1 = True
# #                             content[i] = "CRYST1  100.000  100.000  100.000  90.00  90.00  90.00 P 1           1\n"
# #                             break
                    
# #                     if not has_cryst1:
# #                         content.insert(0, "CRYST1  100.000  100.000  100.000  90.00  90.00  90.00 P 1           1\n")
                    
# #                     tmp.writelines(content)
                
# #                 # Run DSSP on fixed PDB
# #                 from Bio.PDB import DSSP
# #                 dssp = DSSP(model, tmp_pdb, dssp=dssp_executable)
                
# #                 # Extract results - DSSP gives accessibility values directly
# #                 results = []
# #                 for chain in model:
# #                     for residue in chain:
# #                         if residue.id[0] == " ":  # Standard residue
# #                             resid = residue.id[1]
# #                             try:
# #                                 # DSSP key is (chain_id, residue_id)
# #                                 key = (chain.id, (' ', residue.id[1], ' '))
# #                                 if key in dssp:
# #                                     # Get relative solvent accessibility
# #                                     rel_acc = dssp[key][3]
# #                                     # A value > 0.2 (20%) is generally considered accessible
# #                                     core_exterior = "exterior" if rel_acc > 0.2 else "core"
# #                                     results.append({
# #                                         "resid": resid, 
# #                                         "core_exterior": core_exterior,
# #                                         "relative_accessibility": rel_acc
# #                                     })
# #                                 else:
# #                                     # If residue not found in DSSP, use default
# #                                     results.append({
# #                                         "resid": resid, 
# #                                         "core_exterior": "core",
# #                                         "relative_accessibility": 0.0
# #                                     })
# #                             except Exception as e:
# #                                 logging.warning(f"Error processing DSSP for residue {resid}: {e}")
# #                                 results.append({
# #                                     "resid": resid, 
# #                                     "core_exterior": "core",
# #                                     "relative_accessibility": 0.0
# #                                 })
                
# #                 # Clean up temp file
# #                 if os.path.exists(tmp_pdb):
# #                     os.remove(tmp_pdb)
                
# #                 if results:
# #                     logging.info("Successfully used DSSP for core/exterior classification")
# #                     return pd.DataFrame(results)
            
# #             # If DSSP fails or no results, fall back to ShrakeRupley
# #             logging.info("DSSP failed or not available, falling back to ShrakeRupley SASA")
        
# #         except Exception as e:
# #             logging.warning(f"DSSP calculation failed: {e}, falling back to ShrakeRupley")
        
# #         # Fall back to ShrakeRupley SASA calculation
# #         sr = ShrakeRupley()
# #         sr.compute(model, level="R")  # Compute at residue level

# #         # Extract results
# #         results = []
# #         for chain in model:
# #             for residue in chain:
# #                 if residue.id[0] == " ":  # Standard residue
# #                     resid = residue.id[1]
# #                     sasa = residue.sasa if hasattr(residue, 'sasa') else 0.0
# #                     # Normalize SASA to get approximation of relative accessibility
# #                     # Assuming max SASA is around 100 Å²
# #                     rel_acc = min(1.0, sasa / 100.0)
# #                     core_exterior = "exterior" if sasa > sasa_threshold else "core"
# #                     results.append({
# #                         "resid": resid, 
# #                         "core_exterior": core_exterior,
# #                         "relative_accessibility": rel_acc
# #                     })

# #         return pd.DataFrame(results)
# #     except Exception as e:
# #         logging.error(f"Biopython SASA calculation failed: {e}")
# #         import traceback
# #         logging.error(traceback.format_exc())
# #         return fallback_core_exterior(pdb_file)
    
# # def fallback_core_exterior(pdb_file: str) -> pd.DataFrame:
# #     """
# #     Fallback method to classify residues when other methods fail.
# #     Classifies outer 1/3 of residues as exterior, inner 2/3 as core.

# #     Args:
# #         pdb_file: Path to the cleaned PDB file

# #     Returns:
# #         DataFrame with columns 'resid' and 'core_exterior'
# #     """
# #     try:
# #         # Verify file exists and use absolute path
# #         abs_pdb_file = os.path.abspath(pdb_file)
# #         if not os.path.exists(abs_pdb_file):
# #             logging.error(f"PDB file not found: {abs_pdb_file}")
# #             # Create dummy data when PDB file is missing
# #             return pd.DataFrame({
# #                 "resid": list(range(1, 21)),  # Create 20 dummy residues
# #                 "core_exterior": ["core"] * 13 + ["exterior"] * 7,  # 2/3 core, 1/3 exterior
# #                 "relative_accessibility": [0.1] * 13 + [0.7] * 7  # Low for core, high for exterior
# #             })

# #         # Parse PDB to get residue information
# #         residue_df = parse_pdb_residues(pdb_file)
# #         if residue_df.empty:
# #             # Create empty DataFrame with required columns
# #             return pd.DataFrame({
# #                 "resid": list(range(1, 21)),
# #                 "core_exterior": ["core"] * 13 + ["exterior"] * 7,
# #                 "relative_accessibility": [0.1] * 13 + [0.7] * 7
# #             })

# #         # Sort by residue ID
# #         residue_df = residue_df.sort_values("resid")

# #         # Simple classification: outer 1/3 of residues as exterior, inner 2/3 as core
# #         total_residues = len(residue_df)
# #         boundary = int(total_residues * 2/3)

# #         residue_df["core_exterior"] = ["core"] * total_residues
# #         residue_df.loc[boundary:, "core_exterior"] = "exterior"
        
# #         # Add relative accessibility values (0-1 scale)
# #         residue_df["relative_accessibility"] = 0.1  # Default for core
# #         residue_df.loc[boundary:, "relative_accessibility"] = 0.7  # Higher for exterior

# #         return residue_df[["resid", "core_exterior", "relative_accessibility"]]
# #     except Exception as e:
# #         logging.error(f"Fallback classification failed: {e}")
# #         return pd.DataFrame({
# #             "resid": list(range(1, 21)),
# #             "core_exterior": ["core"] * 13 + ["exterior"] * 7,
# #             "relative_accessibility": [0.1] * 13 + [0.7] * 7
# #         })
        
        
# # def parse_pdb_residues(pdb_file: str) -> pd.DataFrame:
# #     """
# #     Parse a PDB file to extract residue-level information.

# #     Args:
# #         pdb_file: Path to the PDB file

# #     Returns:
# #         DataFrame with residue information
# #     """
# #     try:
# #         from Bio.PDB import PDBParser

# #         parser = PDBParser(QUIET=True)
# #         structure = parser.get_structure("protein", pdb_file)

# #         records = []
# #         for model in structure:
# #             for chain in model:
# #                 chain_id = chain.id
# #                 for residue in chain:
# #                     if residue.id[0] == " ":  # Standard residue
# #                         records.append({
# #                             "resid": residue.id[1],
# #                             "resname": residue.get_resname(),
# #                             "chain": chain_id
# #                         })

# #         return pd.DataFrame(records)
# #     except Exception as e:
# #         logging.error(f"Failed to parse PDB residues: {e}")
# #         return pd.DataFrame()

# # def parse_pdb_atoms(pdb_file: str) -> pd.DataFrame:
# #     """
# #     Parse a PDB file to extract atom-level information.

# #     Args:
# #         pdb_file: Path to the PDB file

# #     Returns:
# #         DataFrame with atom information
# #     """
# #     try:
# #         from Bio.PDB import PDBParser

# #         parser = PDBParser(QUIET=True)
# #         structure = parser.get_structure("protein", pdb_file)

# #         records = []
# #         atom_idx = 0
# #         for model in structure:
# #             for chain in model:
# #                 for residue in chain:
# #                     if residue.id[0] == " ":  # Standard residue
# #                         res_id = residue.id[1]
# #                         res_name = residue.get_resname()
# #                         for atom in residue:
# #                             atom_idx += 1
# #                             records.append({
# #                                 "atom_idx": atom_idx,
# #                                 "resid": res_id,
# #                                 "resname": res_name,
# #                                 "atom_name": atom.get_name()
# #                             })

# #         return pd.DataFrame(records)
# #     except Exception as e:
# #         logging.error(f"Failed to parse PDB atoms: {e}")
# #         return pd.DataFrame()

# # def parse_area_file(area_file: str) -> pd.DataFrame:
# #     """
# #     Parse an MSMS .area file to extract SES values per atom.

# #     Args:
# #         area_file: Path to the MSMS .area file

# #     Returns:
# #         DataFrame with SES values
# #     """
# #     try:
# #         atom_idx = []
# #         ses = []

# #         with open(area_file, "r") as f:
# #             for line in f:
# #                 if "Atom" in line or not line.strip():
# #                     continue

# #                 cols = line.split()
# #                 if len(cols) >= 2:
# #                     atom_idx.append(int(cols[0]))
# #                     ses.append(float(cols[1]))

# #         return pd.DataFrame({"atom_idx": atom_idx, "SES": ses})
# #     except Exception as e:
# #         logging.error(f"Failed to parse area file: {e}")
# #         return pd.DataFrame()

# # def run_dssp_analysis(pdb_file: str) -> pd.DataFrame:
# #     """
# #     Run DSSP using a temporary PDB file with correct CRYST1 record,
# #     then parse the resulting DSSP object.
# #     """
# #     logging.info(f"Running DSSP on {pdb_file}")
    
# #     # Extract domain_id from filename
# #     domain_id = os.path.basename(pdb_file).split('.')[0]
    
# #     # Verify PDB file exists
# #     abs_pdb_file = os.path.abspath(pdb_file)
# #     if not os.path.exists(abs_pdb_file):
# #         logging.error(f"PDB file not found: {abs_pdb_file}")
# #         return use_fallback_dssp(pdb_file)
    
# #     try:
# #         # Prepare PDB file with proper formatting for DSSP
# #         corrected_lines = []
        
# #         # Read original PDB file
# #         with open(abs_pdb_file, 'r') as f:
# #             lines = f.readlines()
        
# #         # Add/fix CRYST1 record and ensure proper PDB format
# #         has_cryst1 = False
# #         for i, line in enumerate(lines):
# #             if line.startswith("CRYST1"):
# #                 has_cryst1 = True
# #                 corrected_lines.append("CRYST1  100.000  100.000  100.000  90.00  90.00  90.00 P 1           1\n")
# #             else:
# #                 # Fix common PDB issues that can cause DSSP failures
# #                 if line.startswith("ATOM") or line.startswith("HETATM"):
# #                     # Ensure atom names are properly padded
# #                     if len(line) >= 16:
# #                         atom_name = line[12:16].strip()
# #                         # DSSP requires atom names to be properly spaced
# #                         # Left-justify atom names starting with C, N, O, S, P
# #                         # Right-justify other atom names
# #                         if atom_name and atom_name[0] in "CNOSP":
# #                             padded_atom = f"{atom_name:<4}"
# #                         else:
# #                             padded_atom = f"{atom_name:>4}"
# #                         line = line[:12] + padded_atom + line[16:]
# #                 corrected_lines.append(line)
        
# #         if not has_cryst1:
# #             corrected_lines.insert(0, "CRYST1  100.000  100.000  100.000  90.00  90.00  90.00 P 1           1\n")
        
# #         # Write corrected PDB to temporary file
# #         tmp_pdb = None
# #         try:
# #             with tempfile.NamedTemporaryFile(delete=False, suffix=".pdb", mode="w") as tmp_file:
# #                 tmp_pdb = tmp_file.name
# #                 tmp_file.writelines(corrected_lines)
            
# #             # Try different DSSP executables with enhanced diagnostics
# #             from Bio.PDB import PDBParser, DSSP
# #             parser = PDBParser(QUIET=True)
# #             structure = parser.get_structure("protein", tmp_pdb)
# #             model = structure[0]
            
# #             dssp_found = False
# #             dssp_obj = None
            
# #             # Check if required backbone atoms exist in the structure
# #             has_backbone = False
# #             for chain in model:
# #                 for res in chain:
# #                     if res.has_id("CA") and res.has_id("C") and res.has_id("N"):
# #                         has_backbone = True
# #                         break
# #                 if has_backbone:
# #                     break
            
# #             if not has_backbone:
# #                 logging.warning(f"Model lacks complete backbone atoms, DSSP will likely fail")
            
# #             for dssp_exec in ["dssp", "mkdssp"]:
# #                 dssp_path = shutil.which(dssp_exec)
# #                 if dssp_path:
# #                     try:
# #                         logging.info(f"Trying DSSP executable: {dssp_path}")
                        
# #                         # Test DSSP executable directly first
# #                         try:
# #                             # Run the DSSP executable with -h flag to check if it works
# #                             test_cmd = [dssp_path, "-h"]
# #                             test_result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=5)
# #                             logging.info(f"DSSP test exit code: {test_result.returncode}")
                            
# #                             if test_result.returncode != 0:
# #                                 logging.warning(f"DSSP test failed: {test_result.stderr}")
# #                                 continue
# #                         except Exception as e:
# #                             logging.warning(f"DSSP executable test failed: {e}")
# #                             continue
                        
# #                         # Now try running DSSP via Biopython
# #                         dssp_obj = DSSP(model, tmp_pdb, dssp=dssp_path)
# #                         dssp_found = True
                        
# #                         # Validate DSSP results
# #                         if len(dssp_obj) == 0:
# #                             logging.warning(f"DSSP ran successfully but returned no results")
# #                             dssp_found = False
# #                             continue
                        
# #                         # Check for secondary structure diversity
# #                         ss_counts = {}
# #                         for key in dssp_obj.keys():
# #                             ss = dssp_obj[key][2]  # Secondary structure code
# #                             if ss not in ss_counts:
# #                                 ss_counts[ss] = 0
# #                             ss_counts[ss] += 1
                        
# #                         logging.info(f"DSSP secondary structure counts: {ss_counts}")
                        
# #                         # If we only have undefined structures, DSSP might have failed silently
# #                         if set(ss_counts.keys()).issubset({' ', '-'}):
# #                             logging.warning("DSSP only returned undefined structures, trying different approach")
# #                             dssp_found = False
# #                             continue
                            
# #                         break
# #                     except Exception as e:
# #                         logging.warning(f"Failed with {dssp_exec}: {str(e)}")
# #                         import traceback
# #                         logging.warning(f"Traceback: {traceback.format_exc()}")
            
# #             if not dssp_found or dssp_obj is None:
# #                 logging.warning(f"No working DSSP executable found for {domain_id}")
                
# #                 # Try direct DSSP command-line call as a last resort
# #                 try:
# #                     if dssp_path:
# #                         cmd = [dssp_path, tmp_pdb]
# #                         logging.info(f"Attempting direct DSSP call: {' '.join(cmd)}")
# #                         result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                        
# #                         if result.returncode == 0:
# #                             logging.info("Direct DSSP call succeeded, parsing output")
                            
# #                             # Parse DSSP output manually
# #                             ss_map = {}
# #                             for line in result.stdout.splitlines():
# #                                 # Skip header lines
# #                                 if line.startswith("#") or len(line) < 20 or not line[5:10].strip():
# #                                     continue
                                
# #                                 try:
# #                                     chain_id = line[11]
# #                                     res_num = int(line[5:10])
# #                                     ss_code = line[16]
                                    
# #                                     if ss_code == " ":
# #                                         ss_code = "C"  # Convert space to coil
                                    
# #                                     key = (chain_id, (' ', res_num, ' '))
# #                                     ss_map[key] = ss_code
# #                                 except (ValueError, IndexError):
# #                                     continue
                            
# #                             if ss_map:
# #                                 logging.info(f"Extracted {len(ss_map)} secondary structure assignments manually")
# #                                 records = []
                                
# #                                 # Create records from manual parsing
# #                                 for key, ss_code in ss_map.items():
# #                                     chain_id, (_, resid, _) = key
# #                                     records.append({
# #                                         "domain_id": domain_id,
# #                                         "resid": resid,
# #                                         "chain": chain_id,
# #                                         "dssp": ss_code,
# #                                         "relative_accessibility": 0.5  # Default since we're not parsing this
# #                                     })
                                
# #                                 if records:
# #                                     dssp_df = pd.DataFrame(records)
# #                                     logging.info(f"Created DSSP data manually for {domain_id} with {len(dssp_df)} residues")
# #                                     return dssp_df
# #                         else:
# #                             logging.warning(f"Direct DSSP call failed: {result.stderr}")
# #                 except Exception as e:
# #                     logging.warning(f"Manual DSSP parsing failed: {e}")
                
# #                 return use_fallback_dssp(pdb_file)
            
# #             # Extract DSSP results
# #             records = []
# #             for key in dssp_obj.keys():
# #                 chain_id = key[0]
# #                 resid = key[1][1]  # residue number
# #                 dssp_tuple = dssp_obj[key]
                
# #                 # Extract secondary structure and relative accessibility
# #                 ss_code = dssp_tuple[2]  # Secondary structure code
# #                 rel_acc = dssp_tuple[3]  # Relative accessibility
                
# #                 # Ensure secondary structure is never empty
# #                 if not ss_code or ss_code == ' ' or ss_code == '-':
# #                     ss_code = 'C'  # Default to coil
                
# #                 records.append({
# #                     "domain_id": domain_id,
# #                     "resid": resid,
# #                     "chain": chain_id,
# #                     "dssp": ss_code,
# #                     "relative_accessibility": rel_acc
# #                 })
            
# #             if not records:
# #                 logging.warning(f"DSSP returned no records for {domain_id}")
# #                 return use_fallback_dssp(pdb_file)
                
# #             dssp_df = pd.DataFrame(records)
# #             logging.info(f"DSSP successfully extracted data for {len(dssp_df)} residues in {domain_id}")
# #             logging.info(f"DSSP codes distribution: {dssp_df['dssp'].value_counts().to_dict()}")
            
# #             return dssp_df
        
# #         finally:
# #             # Clean up temporary file
# #             if tmp_pdb and os.path.exists(tmp_pdb):
# #                 try:
# #                     os.remove(tmp_pdb)
# #                 except Exception as e:
# #                     logging.warning(f"Failed to remove temporary file {tmp_pdb}: {e}")
    
# #     except Exception as e:
# #         logging.error(f"Failed to run DSSP analysis for {domain_id}: {e}")
# #         import traceback
# #         logging.error(traceback.format_exc())
# #         return use_fallback_dssp(pdb_file)


# # def use_fallback_dssp(pdb_file: str) -> pd.DataFrame:
# #     """
# #     Fallback method when DSSP fails.
# #     Provides default secondary structure and accessibility values.
    
# #     Args:
# #         pdb_file: Path to the PDB file
        
# #     Returns:
# #         DataFrame with columns: domain_id, resid, chain, dssp, relative_accessibility
# #     """
# #     # Extract domain_id from filename
# #     domain_id = os.path.basename(pdb_file).split('.')[0]
    
# #     logging.info(f"Using fallback secondary structure prediction for {domain_id}")
    
# #     try:
# #         # First check if the PDB file exists
# #         abs_pdb_file = os.path.abspath(pdb_file)
# #         if not os.path.exists(abs_pdb_file):
# #             # Create dummy data for missing PDB
# #             dummy_df = pd.DataFrame({
# #                 "domain_id": [domain_id] * 20,
# #                 "resid": list(range(1, 21)),  # 20 dummy residues
# #                 "chain": ["A"] * 20,
# #                 "dssp": ["C"] * 20,
# #                 "relative_accessibility": [0.5] * 20  # Medium accessibility
# #             })
# #             logging.warning(f"PDB file not found for {domain_id}, using dummy data with {len(dummy_df)} residues")
# #             return dummy_df
        
# #         # Parse PDB to get residue info
# #         try:
# #             from Bio.PDB import PDBParser
# #             parser = PDBParser(QUIET=True)
# #             structure = parser.get_structure("protein", abs_pdb_file)
            
# #             records = []
# #             for model in structure:
# #                 for chain in model:
# #                     chain_id = chain.id
# #                     for residue in chain:
# #                         if residue.id[0] == " ":  # Standard residue
# #                             resid = residue.id[1]
# #                             records.append({
# #                                 "domain_id": domain_id,
# #                                 "resid": resid,
# #                                 "chain": chain_id,
# #                                 "dssp": "C",  # Default to coil
# #                                 "relative_accessibility": 0.5  # Default to moderate accessibility
# #                             })
            
# #             if records:
# #                 result_df = pd.DataFrame(records)
# #                 logging.info(f"Created fallback DSSP data for {domain_id} with {len(result_df)} residues")
# #                 return result_df
# #         except Exception as e:
# #             logging.warning(f"Failed to parse PDB structure for {domain_id}: {e}")
        
# #         # If we get here, we couldn't parse the PDB, so create dummy data
# #         dummy_df = pd.DataFrame({
# #             "domain_id": [domain_id] * 20,
# #             "resid": list(range(1, 21)),
# #             "chain": ["A"] * 20,
# #             "dssp": ["C"] * 20,
# #             "relative_accessibility": [0.5] * 20
# #         })
# #         logging.warning(f"Failed to parse PDB for {domain_id}, using dummy data with {len(dummy_df)} residues")
# #         return dummy_df
        
# #     except Exception as e:
# #         logging.error(f"Fallback DSSP also failed for {domain_id}: {e}")
# #         import traceback
# #         logging.error(traceback.format_exc())
        
# #         # Return minimal dataframe with required columns
# #         dummy_df = pd.DataFrame({
# #             "domain_id": [domain_id] * 20,
# #             "resid": list(range(1, 21)),
# #             "chain": ["A"] * 20,
# #             "dssp": ["C"] * 20,
# #             "relative_accessibility": [0.5] * 20
# #         })
# #         logging.warning(f"Critical failure in DSSP processing for {domain_id}, using emergency dummy data")
# #         return dummy_df



# #!/usr/bin/env python3
# """
# Processing module for core/exterior classification and secondary structure assignment.
# Uses Biopython's DSSP for both tasks, with optimized sharing of results.
# """

# import os
# import logging
# import subprocess
# import tempfile
# import pandas as pd
# import numpy as np
# import shutil
# from Bio.PDB import PDBParser, DSSP, ShrakeRupley
# from typing import Dict, Any, Optional, List, Tuple, Union
# from functools import lru_cache

# # Global cache for DSSP results to avoid redundant processing
# _dssp_cache = {}

# def compute_core_exterior(pdb_file: str, config: Dict[str, Any]) -> Optional[pd.DataFrame]:
#     """
#     Classify residues as 'core' or 'exterior' based on solvent accessibility.

#     Args:
#         pdb_file: Path to the cleaned PDB file
#         config: Configuration dictionary

#     Returns:
#         DataFrame with columns 'resid' and 'core_exterior' or None if classification fails
#     """
#     # We'll use Biopython's DSSP directly now
#     return compute_core_exterior_biopython(pdb_file, config)

# # MSMS functions are commented out as requested
# """
# def compute_core_exterior_msms(pdb_file: str, config: Dict[str, Any]) -> Optional[pd.DataFrame]:
#     # MSMS implementation commented out as requested
#     pass
# """

# def prepare_pdb_for_dssp(pdb_file: str) -> Optional[str]:
#     """
#     Prepare a PDB file for DSSP processing by ensuring it has a proper CRYST1 record
#     and correctly formatted atom names.
    
#     Args:
#         pdb_file: Path to the PDB file
        
#     Returns:
#         Path to the temporary PDB file or None if preparation fails
#     """
#     try:
#         # Verify file exists
#         abs_pdb_file = os.path.abspath(pdb_file)
#         if not os.path.exists(abs_pdb_file):
#             logging.error(f"PDB file not found: {abs_pdb_file}")
#             return None
            
#         # Read original PDB file
#         with open(abs_pdb_file, 'r') as f:
#             lines = f.readlines()
        
#         # Fix PDB format issues
#         corrected_lines = []
#         has_cryst1 = False
        
#         for line in lines:
#             if line.startswith("CRYST1"):
#                 has_cryst1 = True
#                 # Ensure properly formatted CRYST1 record
#                 corrected_lines.append("CRYST1  100.000  100.000  100.000  90.00  90.00  90.00 P 1           1\n")
#             else:
#                 # Fix atom name formatting which can cause DSSP issues
#                 if line.startswith("ATOM") or line.startswith("HETATM"):
#                     # Ensure atom names are properly padded
#                     if len(line) >= 16:
#                         atom_name = line[12:16].strip()
#                         # DSSP requires atom names to be properly spaced
#                         # Left-justify atom names starting with C, N, O, S, P
#                         # Right-justify other atom names
#                         if atom_name and atom_name[0] in "CNOSP":
#                             padded_atom = f"{atom_name:<4}"
#                         else:
#                             padded_atom = f"{atom_name:>4}"
#                         line = line[:12] + padded_atom + line[16:]
#                 corrected_lines.append(line)
        
#         # Add CRYST1 if missing
#         if not has_cryst1:
#             corrected_lines.insert(0, "CRYST1  100.000  100.000  100.000  90.00  90.00  90.00 P 1           1\n")
        
#         # Write to temporary file
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".pdb", mode="w") as tmp:
#             tmp_pdb = tmp.name
#             tmp.writelines(corrected_lines)
            
#         return tmp_pdb
    
#     except Exception as e:
#         logging.error(f"Failed to prepare PDB for DSSP: {e}")
#         import traceback
#         logging.error(traceback.format_exc())
#         return None

# @lru_cache(maxsize=128)
# def run_dssp_once(pdb_file: str) -> Optional[Dict]:
#     """
#     Run DSSP on a PDB file once and cache the results.
#     This unified function handles both secondary structure and accessibility.
    
#     Args:
#         pdb_file: Path to the PDB file
        
#     Returns:
#         Dictionary with DSSP information or None if DSSP fails
#     """
#     # Prepare PDB file
#     tmp_pdb = prepare_pdb_for_dssp(pdb_file)
#     if not tmp_pdb:
#         return None
    
#     try:
#         # Extract domain_id from filename
#         domain_id = os.path.basename(pdb_file).split('.')[0]
        
#         # Parse structure
#         parser = PDBParser(QUIET=True)
#         structure = parser.get_structure("protein", tmp_pdb)
#         model = structure[0]
        
#         # Check for required backbone atoms
#         has_backbone = False
#         for chain in model:
#             for res in chain:
#                 if res.has_id("CA") and res.has_id("C") and res.has_id("N"):
#                     has_backbone = True
#                     break
#             if has_backbone:
#                 break
        
#         if not has_backbone:
#             logging.warning(f"Model lacks complete backbone atoms for {domain_id}, DSSP will likely fail")
        
#         # Try to find and run DSSP
#         dssp_found = False
#         dssp_obj = None
        
#         for dssp_exec in ["dssp", "mkdssp"]:
#             dssp_path = shutil.which(dssp_exec)
#             if not dssp_path:
#                 continue
                
#             try:
#                 logging.info(f"Running DSSP using {dssp_path} on {domain_id}")
#                 dssp_obj = DSSP(model, tmp_pdb, dssp=dssp_path)
                
#                 # Validate results
#                 if len(dssp_obj) == 0:
#                     logging.warning(f"DSSP returned empty results for {domain_id}")
#                     continue
                
#                 # Check if we got valid secondary structure assignments
#                 ss_counts = {}
#                 for key in dssp_obj.keys():
#                     ss = dssp_obj[key][2]  # Secondary structure code
#                     if ss not in ss_counts:
#                         ss_counts[ss] = 0
#                     ss_counts[ss] += 1
                
#                 logging.info(f"DSSP results for {domain_id}: {ss_counts}")
                
#                 # Success
#                 dssp_found = True
#                 break
                
#             except Exception as e:
#                 logging.warning(f"DSSP ({dssp_exec}) failed on {domain_id}: {e}")
#                 continue
        
#         # If DSSP failed through Biopython, try direct execution as last resort
#         if not dssp_found and dssp_path:
#             try:
#                 logging.info(f"Attempting direct DSSP call for {domain_id}")
#                 result = subprocess.run([dssp_path, tmp_pdb], capture_output=True, text=True, timeout=30)
                
#                 if result.returncode == 0:
#                     # Parse DSSP output manually
#                     logging.info(f"Direct DSSP call succeeded for {domain_id}")
                    
#                     # Create a simplified DSSP-like object from output
#                     dssp_data = {}
#                     for line in result.stdout.splitlines():
#                         # DSSP output format has the secondary structure at position 16
#                         # and residue info starting at position 5
#                         if not line.startswith("#") and len(line) > 16:
#                             try:
#                                 chain_id = line[11]
#                                 res_num = int(line[5:10].strip())
#                                 ss_code = line[16]
#                                 if ss_code == " ":
#                                     ss_code = "C"  # Default to coil
                                
#                                 # Get accessibility
#                                 acc_str = line[34:38].strip()
#                                 acc = float(acc_str) if acc_str else 0.0
#                                 # Normalize to 0-1 scale
#                                 rel_acc = min(1.0, acc / 100.0)
                                
#                                 key = (chain_id, (' ', res_num, ' '))
#                                 dssp_data[key] = {
#                                     'ss': ss_code,
#                                     'acc': rel_acc
#                                 }
#                             except (ValueError, IndexError) as e:
#                                 continue
                    
#                     if dssp_data:
#                         logging.info(f"Manually parsed {len(dssp_data)} residues from DSSP output for {domain_id}")
#                         return {
#                             'domain_id': domain_id,
#                             'dssp_data': dssp_data,
#                             'method': 'manual'
#                         }
#             except Exception as e:
#                 logging.warning(f"Direct DSSP execution failed for {domain_id}: {e}")
        
#         # Return results if we have them
#         if dssp_found and dssp_obj:
#             return {
#                 'domain_id': domain_id,
#                 'dssp_obj': dssp_obj,
#                 'method': 'biopython'
#             }
        
#         # If we get here, DSSP failed
#         logging.warning(f"All DSSP methods failed for {domain_id}")
#         return None
        
#     except Exception as e:
#         logging.error(f"DSSP processing error: {e}")
#         import traceback
#         logging.error(traceback.format_exc())
#         return None
        
#     finally:
#         # Clean up temp file
#         if tmp_pdb and os.path.exists(tmp_pdb):
#             try:
#                 os.remove(tmp_pdb)
#             except Exception as e:
#                 logging.warning(f"Failed to remove temporary file {tmp_pdb}: {e}")

# def collect_dssp_data(pdb_file: str, domain_id: str, temp: str, replica: str = "0") -> pd.DataFrame:
#     """
#     Collect DSSP data for a specific domain, temperature and replica.
#     Uses cached DSSP results to avoid redundant processing.
    
#     Args:
#         pdb_file: Path to the PDB file
#         domain_id: Domain identifier
#         temp: Temperature
#         replica: Replica index
        
#     Returns:
#         DataFrame with DSSP data
#     """
#     # Get DSSP results from cache
#     dssp_results = run_dssp_once(pdb_file)
    
#     if not dssp_results:
#         # Return fallback data if DSSP failed
#         logging.warning(f"Using fallback DSSP data for {domain_id} at {temp}K (rep {replica})")
#         return use_fallback_dssp(pdb_file)
    
#     # Process based on how DSSP was run
#     if dssp_results['method'] == 'biopython':
#         dssp_obj = dssp_results['dssp_obj']
        
#         # Extract secondary structure and accessibility
#         records = []
#         for key in dssp_obj.keys():
#             chain_id = key[0]
#             resid = key[1][1]  # residue number
#             dssp_tuple = dssp_obj[key]
            
#             # Extract data
#             ss_code = dssp_tuple[2]  # Secondary structure code
#             rel_acc = dssp_tuple[3]  # Relative accessibility
            
#             # Ensure secondary structure is never empty
#             if not ss_code or ss_code == ' ' or ss_code == '-':
#                 ss_code = 'C'  # Default to coil
            
#             records.append({
#                 "domain_id": domain_id,
#                 "resid": resid,
#                 "chain": chain_id,
#                 "dssp": ss_code,
#                 "relative_accessibility": rel_acc
#             })
        
#         if records:
#             df = pd.DataFrame(records)
#             logging.info(f"DSSP secondary structure for {domain_id}: {df['dssp'].value_counts().to_dict()}")
#             return df
            
#     elif dssp_results['method'] == 'manual':
#         # Process manual DSSP results
#         dssp_data = dssp_results['dssp_data']
#         records = []
        
#         for key, data in dssp_data.items():
#             chain_id = key[0]
#             resid = key[1][1]  # residue number
#             ss_code = data['ss']
#             rel_acc = data['acc']
            
#             records.append({
#                 "domain_id": domain_id,
#                 "resid": resid,
#                 "chain": chain_id,
#                 "dssp": ss_code,
#                 "relative_accessibility": rel_acc
#             })
        
#         if records:
#             df = pd.DataFrame(records)
#             logging.info(f"DSSP secondary structure for {domain_id}: {df['dssp'].value_counts().to_dict()}")
#             return df
    
#     # If we get here, use fallback
#     return use_fallback_dssp(pdb_file)

# def compute_core_exterior_biopython(pdb_file: str, config: Dict[str, Any]) -> pd.DataFrame:
#     """
#     Use DSSP to classify residues as 'core' or 'exterior' based on accessibility.
#     Uses cached DSSP results if available.

#     Args:
#         pdb_file: Path to the cleaned PDB file
#         config: Configuration dictionary

#     Returns:
#         DataFrame with columns 'resid' and 'core_exterior'
#     """
#     domain_id = os.path.basename(pdb_file).split('.')[0]
#     sasa_threshold = config.get("core_exterior", {}).get("sasa_threshold", 20.0)
    
#     try:
#         # Get DSSP results (from cache if available)
#         dssp_results = run_dssp_once(pdb_file)
        
#         if dssp_results:
#             # Process based on how DSSP was run
#             if dssp_results['method'] == 'biopython':
#                 dssp_obj = dssp_results['dssp_obj']
                
#                 # Extract accessibility and classify as core/exterior
#                 results = []
#                 for key in dssp_obj.keys():
#                     chain_id = key[0]
#                     resid = key[1][1]  # residue number
#                     dssp_tuple = dssp_obj[key]
                    
#                     # Get relative accessibility
#                     rel_acc = dssp_tuple[3]  # Relative accessibility (0-1 scale)
                    
#                     # Classify based on threshold (20% is typical cutoff)
#                     core_exterior = "exterior" if rel_acc > 0.2 else "core"
                    
#                     results.append({
#                         "resid": resid,
#                         "chain": chain_id,
#                         "core_exterior": core_exterior,
#                         "relative_accessibility": rel_acc
#                     })
                
#                 if results:
#                     logging.info(f"Successfully classified {len(results)} residues using DSSP for {domain_id}")
#                     return pd.DataFrame(results)
            
#             elif dssp_results['method'] == 'manual':
#                 # Process manual DSSP results
#                 dssp_data = dssp_results['dssp_data']
#                 results = []
                
#                 for key, data in dssp_data.items():
#                     chain_id = key[0]
#                     resid = key[1][1]  # residue number
#                     rel_acc = data['acc']
                    
#                     # Classify based on threshold
#                     core_exterior = "exterior" if rel_acc > 0.2 else "core"
                    
#                     results.append({
#                         "resid": resid,
#                         "chain": chain_id,
#                         "core_exterior": core_exterior,
#                         "relative_accessibility": rel_acc
#                     })
                
#                 if results:
#                     logging.info(f"Successfully classified {len(results)} residues using manual DSSP for {domain_id}")
#                     return pd.DataFrame(results)
        
#         # If DSSP failed, try ShrakeRupley
#         logging.info(f"DSSP failed for {domain_id}, using ShrakeRupley SASA")
        
#         # Parse structure
#         parser = PDBParser(QUIET=True)
#         structure = parser.get_structure("protein", pdb_file)
#         model = structure[0]
        
#         # Calculate SASA
#         sr = ShrakeRupley()
#         sr.compute(model, level="R")  # Compute at residue level
        
#         # Extract results
#         results = []
#         for chain in model:
#             for residue in chain:
#                 if residue.id[0] == " ":  # Standard residue
#                     resid = residue.id[1]
#                     sasa = residue.sasa if hasattr(residue, 'sasa') else 0.0
                    
#                     # Normalize SASA
#                     rel_acc = min(1.0, sasa / 100.0)
#                     core_exterior = "exterior" if sasa > sasa_threshold else "core"
                    
#                     results.append({
#                         "resid": resid,
#                         "chain": chain.id,
#                         "core_exterior": core_exterior,
#                         "relative_accessibility": rel_acc
#                     })
        
#         if results:
#             logging.info(f"Successfully classified {len(results)} residues using ShrakeRupley for {domain_id}")
#             return pd.DataFrame(results)
        
#         # Final fallback
#         logging.warning(f"All methods failed, using fallback classification for {domain_id}")
#         return fallback_core_exterior(pdb_file)
        
#     except Exception as e:
#         logging.error(f"Core/exterior classification failed: {e}")
#         import traceback
#         logging.error(traceback.format_exc())
#         return fallback_core_exterior(pdb_file)

# def fallback_core_exterior(pdb_file: str) -> pd.DataFrame:
#     """
#     Fallback method to classify residues when other methods fail.
#     Classifies outer 1/3 of residues as exterior, inner 2/3 as core.

#     Args:
#         pdb_file: Path to the cleaned PDB file

#     Returns:
#         DataFrame with columns 'resid' and 'core_exterior'
#     """
#     try:
#         # Verify file exists and use absolute path
#         abs_pdb_file = os.path.abspath(pdb_file)
#         if not os.path.exists(abs_pdb_file):
#             logging.error(f"PDB file not found: {abs_pdb_file}")
#             # Create dummy data when PDB file is missing
#             return pd.DataFrame({
#                 "resid": list(range(1, 21)),  # Create 20 dummy residues
#                 "core_exterior": ["core"] * 13 + ["exterior"] * 7,  # 2/3 core, 1/3 exterior
#                 "relative_accessibility": [0.1] * 13 + [0.7] * 7  # Low for core, high for exterior
#             })

#         # Parse PDB to get residue information
#         residue_df = parse_pdb_residues(pdb_file)
#         if residue_df.empty:
#             # Create empty DataFrame with required columns
#             return pd.DataFrame({
#                 "resid": list(range(1, 21)),
#                 "core_exterior": ["core"] * 13 + ["exterior"] * 7,
#                 "relative_accessibility": [0.1] * 13 + [0.7] * 7
#             })

#         # Sort by residue ID
#         residue_df = residue_df.sort_values("resid")

#         # Simple classification: outer 1/3 of residues as exterior, inner 2/3 as core
#         total_residues = len(residue_df)
#         boundary = int(total_residues * 2/3)

#         residue_df["core_exterior"] = ["core"] * total_residues
#         residue_df.loc[boundary:, "core_exterior"] = "exterior"
        
#         # Add relative accessibility values (0-1 scale)
#         residue_df["relative_accessibility"] = 0.1  # Default for core
#         residue_df.loc[boundary:, "relative_accessibility"] = 0.7  # Higher for exterior

#         return residue_df[["resid", "core_exterior", "relative_accessibility"]]
#     except Exception as e:
#         logging.error(f"Fallback classification failed: {e}")
#         return pd.DataFrame({
#             "resid": list(range(1, 21)),
#             "core_exterior": ["core"] * 13 + ["exterior"] * 7,
#             "relative_accessibility": [0.1] * 13 + [0.7] * 7
#         })

# def parse_pdb_residues(pdb_file: str) -> pd.DataFrame:
#     """
#     Parse a PDB file to extract residue-level information.

#     Args:
#         pdb_file: Path to the PDB file

#     Returns:
#         DataFrame with residue information
#     """
#     try:
#         from Bio.PDB import PDBParser

#         parser = PDBParser(QUIET=True)
#         structure = parser.get_structure("protein", pdb_file)

#         records = []
#         for model in structure:
#             for chain in model:
#                 chain_id = chain.id
#                 for residue in chain:
#                     if residue.id[0] == " ":  # Standard residue
#                         records.append({
#                             "resid": residue.id[1],
#                             "resname": residue.get_resname(),
#                             "chain": chain_id
#                         })

#         return pd.DataFrame(records)
#     except Exception as e:
#         logging.error(f"Failed to parse PDB residues: {e}")
#         return pd.DataFrame()

# def parse_pdb_atoms(pdb_file: str) -> pd.DataFrame:
#     """
#     Parse a PDB file to extract atom-level information.

#     Args:
#         pdb_file: Path to the PDB file

#     Returns:
#         DataFrame with atom information
#     """
#     try:
#         from Bio.PDB import PDBParser

#         parser = PDBParser(QUIET=True)
#         structure = parser.get_structure("protein", pdb_file)

#         records = []
#         atom_idx = 0
#         for model in structure:
#             for chain in model:
#                 for residue in chain:
#                     if residue.id[0] == " ":  # Standard residue
#                         res_id = residue.id[1]
#                         res_name = residue.get_resname()
#                         for atom in residue:
#                             atom_idx += 1
#                             records.append({
#                                 "atom_idx": atom_idx,
#                                 "resid": res_id,
#                                 "resname": res_name,
#                                 "atom_name": atom.get_name()
#                             })

#         return pd.DataFrame(records)
#     except Exception as e:
#         logging.error(f"Failed to parse PDB atoms: {e}")
#         return pd.DataFrame()

# def run_dssp_analysis(pdb_file: str) -> pd.DataFrame:
#     """
#     Run DSSP to get secondary structure assignments.
#     Uses the cached DSSP results if available.
    
#     Args:
#         pdb_file: Path to the PDB file
        
#     Returns:
#         DataFrame with columns: domain_id, resid, chain, dssp, relative_accessibility
#     """
#     domain_id = os.path.basename(pdb_file).split('.')[0]
    
#     try:
#         # Get DSSP results (from cache if available)
#         dssp_results = run_dssp_once(pdb_file)
        
#         if dssp_results:
#             # Process based on how DSSP was run
#             if dssp_results['method'] == 'biopython':
#                 dssp_obj = dssp_results['dssp_obj']
                
#                 # Extract secondary structure and accessibility
#                 records = []
#                 for key in dssp_obj.keys():
#                     chain_id = key[0]
#                     resid = key[1][1]  # residue number
#                     dssp_tuple = dssp_obj[key]
                    
#                     # Extract data
#                     ss_code = dssp_tuple[2]  # Secondary structure code
#                     rel_acc = dssp_tuple[3]  # Relative accessibility
                    
#                     # Ensure secondary structure is never empty
#                     if not ss_code or ss_code == ' ' or ss_code == '-':
#                         ss_code = 'C'  # Default to coil
                    
#                     records.append({
#                         "domain_id": domain_id,
#                         "resid": resid,
#                         "chain": chain_id,
#                         "dssp": ss_code,
#                         "relative_accessibility": rel_acc
#                     })
                
#                 if records:
#                     df = pd.DataFrame(records)
#                     logging.info(f"Successfully extracted DSSP data for {len(df)} residues in {domain_id}")
#                     logging.info(f"DSSP codes distribution: {df['dssp'].value_counts().to_dict()}")
#                     return df
            
#             elif dssp_results['method'] == 'manual':
#                 # Process manual DSSP results
#                 dssp_data = dssp_results['dssp_data']
#                 records = []
                
#                 for key, data in dssp_data.items():
#                     chain_id = key[0]
#                     resid = key[1][1]  # residue number
#                     ss_code = data['ss']
#                     rel_acc = data['acc']
                    
#                     records.append({
#                         "domain_id": domain_id,
#                         "resid": resid,
#                         "chain": chain_id,
#                         "dssp": ss_code,
#                         "relative_accessibility": rel_acc
#                     })
                
#                 if records:
#                     df = pd.DataFrame(records)
#                     logging.info(f"Successfully extracted manual DSSP data for {len(df)} residues in {domain_id}")
#                     logging.info(f"DSSP codes distribution: {df['dssp'].value_counts().to_dict()}")
#                     return df
        
#         # If DSSP fails, use fallback
#         logging.warning(f"DSSP failed for {domain_id}, using fallback")
#         return use_fallback_dssp(pdb_file)
        
#     except Exception as e:
#         logging.error(f"Failed to run DSSP analysis for {domain_id}: {e}")
#         import traceback
#         logging.error(traceback.format_exc())
#         return use_fallback_dssp(pdb_file)
    


# def use_fallback_dssp(pdb_file: str) -> pd.DataFrame:
#     """
#     Fallback method when DSSP fails.
#     Provides default secondary structure and accessibility values.
    
#     Args:
#         pdb_file: Path to the PDB file
        
#     Returns:
#         DataFrame with columns: domain_id, resid, chain, dssp, relative_accessibility
#     """
#     # Extract domain_id from filename
#     domain_id = os.path.basename(pdb_file).split('.')[0]
    
#     logging.info(f"Using fallback secondary structure prediction for {domain_id}")
    
#     try:
#         # First check if the PDB file exists
#         abs_pdb_file = os.path.abspath(pdb_file)
#         if not os.path.exists(abs_pdb_file):
#             # Create dummy data for missing PDB
#             dummy_df = pd.DataFrame({
#                 "domain_id": [domain_id] * 20,
#                 "resid": list(range(1, 21)),  # 20 dummy residues
#                 "chain": ["A"] * 20,
#                 "dssp": ["C"] * 20,
#                 "relative_accessibility": [0.5] * 20  # Medium accessibility
#             })
#             logging.warning(f"PDB file not found for {domain_id}, using dummy data with {len(dummy_df)} residues")
#             return dummy_df
        
#         # Parse PDB to get residue info
#         try:
#             from Bio.PDB import PDBParser
#             parser = PDBParser(QUIET=True)
#             structure = parser.get_structure("protein", abs_pdb_file)
            
#             records = []
#             for model in structure:
#                 for chain in model:
#                     chain_id = chain.id
#                     for residue in chain:
#                         if residue.id[0] == " ":  # Standard residue
#                             resid = residue.id[1]
#                             records.append({
#                                 "domain_id": domain_id,
#                                 "resid": resid,
#                                 "chain": chain_id,
#                                 "dssp": "C",  # Default to coil
#                                 "relative_accessibility": 0.5  # Default to moderate accessibility
#                             })
            
#             if records:
#                 result_df = pd.DataFrame(records)
#                 logging.info(f"Created fallback DSSP data for {domain_id} with {len(result_df)} residues")
#                 return result_df
#         except Exception as e:
#             logging.warning(f"Failed to parse PDB structure for {domain_id}: {e}")
        
#         # If we get here, we couldn't parse the PDB, so create dummy data
#         dummy_df = pd.DataFrame({
#             "domain_id": [domain_id] * 20,
#             "resid": list(range(1, 21)),
#             "chain": ["A"] * 20,
#             "dssp": ["C"] * 20,
#             "relative_accessibility": [0.5] * 20
#         })
#         logging.warning(f"Failed to parse PDB for {domain_id}, using dummy data with {len(dummy_df)} residues")
#         return dummy_df
        
#     except Exception as e:
#         logging.error(f"Fallback DSSP also failed for {domain_id}: {e}")
#         import traceback
#         logging.error(traceback.format_exc())
        
#         # Return minimal dataframe with required columns
#         dummy_df = pd.DataFrame({
#             "domain_id": [domain_id] * 20,
#             "resid": list(range(1, 21)),
#             "chain": ["A"] * 20,
#             "dssp": ["C"] * 20,
#             "relative_accessibility": [0.5] * 20
#         })
#         logging.warning(f"Critical failure in DSSP processing for {domain_id}, using emergency dummy data")
#         return dummy_df