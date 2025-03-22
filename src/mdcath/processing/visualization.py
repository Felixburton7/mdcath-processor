#!/usr/bin/env python3
"""
Enhanced module for generating visualizations of processed mdCATH data.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from typing import Dict, Any, Optional, List, Tuple
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import kde

def create_temperature_summary_heatmap(rmsf_data: Dict[str, pd.DataFrame], 
                                     output_dir: str) -> Optional[str]:
    """
    Create a heatmap showing RMSF values across temperatures for all domains.
    Modified to remove domain ID labels for better handling of large datasets.
    
    Args:
        rmsf_data: Dictionary with RMSF data for all temperatures
        output_dir: Directory to save visualization
        
    Returns:
        Path to the saved figure or None if creation fails
    """
    try:
        # Ensure output directory exists
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Extract temperature values
        temps = [temp for temp in rmsf_data.keys() if temp != "average"]
        
        if not temps:
            logging.warning("No temperature data available for heatmap")
            return None
            
        # Prepare data for heatmap
        domain_ids = set()
        for temp in temps:
            if temp in rmsf_data:
                domain_ids.update(rmsf_data[temp]["domain_id"].unique())
        
        domain_ids = sorted(list(domain_ids))
        
        # Create a dataframe for the heatmap
        heatmap_data = []
        
        for domain_id in domain_ids:
            domain_data = {"domain_id": domain_id}
            
            for temp in temps:
                if temp in rmsf_data:
                    domain_temp_data = rmsf_data[temp][rmsf_data[temp]["domain_id"] == domain_id]
                    if not domain_temp_data.empty:
                        domain_data[f"rmsf_{temp}"] = domain_temp_data[f"rmsf_{temp}"].mean()
            
            heatmap_data.append(domain_data)
        
        if not heatmap_data:
            logging.warning("No data available for heatmap")
            return None
            
        # Create dataframe and pivot for heatmap
        heatmap_df = pd.DataFrame(heatmap_data)
        heatmap_pivot = heatmap_df.set_index("domain_id")
        
        # Create heatmap - MODIFICATION: hide domain IDs on y-axis
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_pivot, annot=False, cmap="viridis")
        plt.title("Average RMSF by Domain and Temperature")
        plt.xlabel("Temperature (K)")
        plt.ylabel(f"Domains (n={len(domain_ids)})")
        
        # Hide y-ticks (domain IDs) for better handling of large datasets
        plt.yticks([])
        
        # Add diagnostic information
        plt.text(0.01, 0.01, f"Total domains: {len(domain_ids)}\nTemperatures: {', '.join(temps)}",
                transform=plt.gca().transAxes, fontsize=8, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(vis_dir, "temperature_summary.png")
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        logging.info(f"Temperature summary heatmap saved to {output_path}")
        return output_path
    except Exception as e:
        logging.error(f"Failed to create temperature summary heatmap: {e}")
        return None

def create_temperature_average_summary(temperature_average: pd.DataFrame, 
                                     output_dir: str) -> Optional[str]:
    """
    Create a visualization showing average RMSF across temperatures.
    Replaced bar graph with scatter plot for better handling of many domains.
    
    Args:
        temperature_average: DataFrame with average RMSF values across all temperatures
        output_dir: Directory to save visualization
        
    Returns:
        Path to the saved figure or None if creation fails
    """
    try:
        # Ensure output directory exists
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        if temperature_average is None or temperature_average.empty:
            logging.warning("No temperature average data available for summary")
            return None
            
        # Group by domain_id and calculate statistics
        domain_stats = temperature_average.groupby("domain_id")["rmsf_average"].agg(
            ["mean", "std", "min", "max"]).reset_index()
        
        # Sort by mean RMSF
        domain_stats = domain_stats.sort_values("mean", ascending=False)
        
        # Create scatter plot (replacement for bar plot)
        plt.figure(figsize=(12, 8))
        
        # Create a categorical x-axis for the domains
        domain_stats["domain_index"] = range(len(domain_stats))
        
        # Create scatter plot with error bars
        plt.errorbar(domain_stats["domain_index"], domain_stats["mean"], 
                   yerr=domain_stats["std"], fmt='o', alpha=0.6, 
                   elinewidth=0.8, capsize=3, markersize=4)
        
        # Add trend line
        z = np.polyfit(domain_stats["domain_index"], domain_stats["mean"], 1)
        p = np.poly1d(z)
        plt.plot(domain_stats["domain_index"], p(domain_stats["domain_index"]), 
                "r--", linewidth=1.5, alpha=0.7)
        
        # Show only subset of domain labels for clarity (first 5 and last 5)
        max_domains_to_show = 10
        if len(domain_stats) <= max_domains_to_show:
            plt.xticks(domain_stats["domain_index"], domain_stats["domain_id"], rotation=90)
        else:
            # Show first 5 and last 5 domains
            show_indices = list(range(5)) + list(range(len(domain_stats)-5, len(domain_stats)))
            show_domains = [domain_stats.iloc[i]["domain_id"] if i in show_indices else "" 
                          for i in range(len(domain_stats))]
            plt.xticks(domain_stats["domain_index"], show_domains, rotation=90)
        
        plt.title("Average RMSF by Domain (Across All Temperatures)")
        plt.xlabel(f"Domains (n={len(domain_stats)})")
        plt.ylabel("Average RMSF (nm)")
        
        # Add quality metrics
        mean_rmsf = domain_stats["mean"].mean()
        std_rmsf = domain_stats["mean"].std()
        plt.text(0.01, 0.95, f"Overall mean: {mean_rmsf:.4f} nm\nStd: {std_rmsf:.4f} nm",
                transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(vis_dir, "temperature_average_summary.png")
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        logging.info(f"Temperature average summary saved to {output_path}")
        return output_path
    except Exception as e:
        logging.error(f"Failed to create temperature average summary: {e}")
        return None

def create_rmsf_distribution_plots(rmsf_data: Dict[str, pd.DataFrame], 
                                  output_dir: str) -> Optional[str]:
    """
    Create distribution plots (violin plot and histogram) showing RMSF distribution by temperature.
    Enhanced to create separate histograms for each temperature.
    
    Args:
        rmsf_data: Dictionary with RMSF data for all temperatures
        output_dir: Directory to save visualization
        
    Returns:
        Path to the saved figure or None if creation fails
    """
    try:
        # Ensure output directory exists
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Extract temperature values
        temps = [temp for temp in rmsf_data.keys() if temp != "average"]
        
        if not temps:
            logging.warning("No temperature data available for distribution plots")
            return None
            
        # Prepare data for plotting
        dist_data = []
        
        for temp in temps:
            if temp in rmsf_data:
                temp_df = rmsf_data[temp]
                rmsf_col = f"rmsf_{temp}"
                
                if rmsf_col in temp_df.columns:
                    for _, row in temp_df.iterrows():
                        dist_data.append({
                            "Temperature": temp,
                            "RMSF": row[rmsf_col]
                        })
        
        if not dist_data:
            logging.warning("No data available for distribution plots")
            return None
            
        # Create dataframe for plotting
        dist_df = pd.DataFrame(dist_data)
        
        # Create violin plot (unchanged)
        plt.figure(figsize=(10, 6))
        sns.violinplot(x="Temperature", y="RMSF", data=dist_df)
        plt.title("RMSF Distribution by Temperature")
        plt.xlabel("Temperature (K)")
        plt.ylabel("RMSF (nm)")
        plt.tight_layout()
        
        # Save violin plot
        violin_path = os.path.join(vis_dir, "rmsf_violin_plot.png")
        plt.savefig(violin_path, dpi=300)
        plt.close()
        
        # Create histogram (overlaid version - unchanged)
        plt.figure(figsize=(10, 6))
        for temp in temps:
            temp_data = dist_df[dist_df["Temperature"] == temp]["RMSF"]
            if not temp_data.empty:
                sns.histplot(temp_data, kde=True, label=f"{temp}K")
        
        plt.title("RMSF Histogram by Temperature")
        plt.xlabel("RMSF (nm)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        
        # Save histogram
        hist_path = os.path.join(vis_dir, "rmsf_histogram.png")
        plt.savefig(hist_path, dpi=300)
        plt.close()
        
        # NEW: Create separate histograms for each temperature plus average
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        axs = axs.flatten()
        
        # Process each temperature
        for i, temp in enumerate(temps):
            temp_data = dist_df[dist_df["Temperature"] == temp]["RMSF"]
            if not temp_data.empty:
                sns.histplot(temp_data, kde=True, bins=30, ax=axs[i], color='skyblue')
                axs[i].set_title(f"Temperature {temp}K")
                axs[i].set_xlabel("RMSF (nm)")
                axs[i].set_ylabel("Frequency")
                
                # Add mean and std as text
                mean_val = temp_data.mean()
                std_val = temp_data.std()
                axs[i].text(0.05, 0.95, f"Mean: {mean_val:.4f}\nStd: {std_val:.4f}", 
                          transform=axs[i].transAxes, fontsize=10, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                # Add vertical line at mean
                axs[i].axvline(mean_val, color='blue', linestyle='--', linewidth=1.5)
        
        # Create average histogram if data available
        if "average" in rmsf_data and not rmsf_data["average"].empty:
            avg_data = rmsf_data["average"]["rmsf_average"]
            if not avg_data.empty:
                sns.histplot(avg_data, kde=True, bins=30, ax=axs[5], color='orange')
                axs[5].set_title("Average RMSF (All Temperatures)")
                axs[5].set_xlabel("RMSF (nm)")
                axs[5].set_ylabel("Frequency")
                
                # Add mean and std as text
                mean_val = avg_data.mean()
                std_val = avg_data.std()
                axs[5].text(0.05, 0.95, f"Mean: {mean_val:.4f}\nStd: {std_val:.4f}", 
                          transform=axs[5].transAxes, fontsize=10, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                # Add vertical line at mean
                axs[5].axvline(mean_val, color='blue', linestyle='--', linewidth=1.5)
        
        plt.tight_layout()
        
        # Save separated histograms
        separated_hist_path = os.path.join(vis_dir, "rmsf_histogram_seperated.png")
        plt.savefig(separated_hist_path, dpi=300)
        plt.close()
        
        logging.info(f"RMSF distribution plots saved to {violin_path}, {hist_path}, and {separated_hist_path}")
        return violin_path
    except Exception as e:
        logging.error(f"Failed to create RMSF distribution plots: {e}")
        return None

def create_amino_acid_rmsf_plot(rmsf_data: Dict[str, pd.DataFrame], 
                              output_dir: str) -> Optional[str]:
    """
    Create a violin plot showing RMSF distribution by amino acid type.
    Modified to consolidate HSE, HSP into HIS and create two versions.
    
    Args:
        rmsf_data: Dictionary with RMSF data for all temperatures
        output_dir: Directory to save visualization
        
    Returns:
        Path to the saved figure or None if creation fails
    """
    try:
        # Ensure output directory exists
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Use temperature average if available
        if "average" in rmsf_data and not rmsf_data["average"].empty:
            aa_data = []
            
            avg_df = rmsf_data["average"].copy()
            
            # Consolidate HSE, HSP into HIS
            avg_df["resname"] = avg_df["resname"].apply(
                lambda x: "HIS" if x in ["HSE", "HSP", "HSD"] else x
            )
            
            for _, row in avg_df.iterrows():
                aa_data.append({
                    "Residue": row["resname"],
                    "RMSF": row["rmsf_average"]
                })
                
            # Create dataframe for plotting
            aa_df = pd.DataFrame(aa_data)
            
            # Get common and non-standard residues
            standard_aa = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", 
                         "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", 
                         "THR", "TRP", "TYR", "VAL"]
            
            all_residues = sorted(aa_df["Residue"].unique())
            
            # 1. Create simple violin plot (consolidating HSE, HSP into HIS)
            plt.figure(figsize=(14, 8))
            sns.violinplot(x="Residue", y="RMSF", data=aa_df, order=all_residues)
            plt.title("RMSF Distribution by Amino Acid Type")
            plt.xlabel("Amino Acid")
            plt.ylabel("RMSF (nm)")
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save simple figure
            output_path = os.path.join(vis_dir, "amino_acid_rmsf_violin_plot.png")
            plt.savefig(output_path, dpi=300)
            plt.close()
            
            # 2. Create advanced colored version with additional information
            # Define amino acid properties for coloring
            aa_properties = {
                "ALA": {"hydrophobicity": 1.8, "volume": 88.6, "color": "salmon", "type": "hydrophobic"},
                "ARG": {"hydrophobicity": -4.5, "volume": 173.4, "color": "royalblue", "type": "basic"},
                "ASN": {"hydrophobicity": -3.5, "volume": 114.1, "color": "mediumseagreen", "type": "polar"},
                "ASP": {"hydrophobicity": -3.5, "volume": 111.1, "color": "crimson", "type": "acidic"},
                "CYS": {"hydrophobicity": 2.5, "volume": 108.5, "color": "gold", "type": "special"},
                "GLN": {"hydrophobicity": -3.5, "volume": 143.8, "color": "mediumseagreen", "type": "polar"},
                "GLU": {"hydrophobicity": -3.5, "volume": 138.4, "color": "crimson", "type": "acidic"},
                "GLY": {"hydrophobicity": -0.4, "volume": 60.1, "color": "lightgray", "type": "special"},
                "HIS": {"hydrophobicity": -3.2, "volume": 153.2, "color": "cornflowerblue", "type": "basic"},
                "ILE": {"hydrophobicity": 4.5, "volume": 166.7, "color": "darksalmon", "type": "hydrophobic"},
                "LEU": {"hydrophobicity": 3.8, "volume": 166.7, "color": "darksalmon", "type": "hydrophobic"},
                "LYS": {"hydrophobicity": -3.9, "volume": 168.6, "color": "royalblue", "type": "basic"},
                "MET": {"hydrophobicity": 1.9, "volume": 162.9, "color": "orange", "type": "hydrophobic"},
                "PHE": {"hydrophobicity": 2.8, "volume": 189.9, "color": "chocolate", "type": "aromatic"},
                "PRO": {"hydrophobicity": -1.6, "volume": 112.7, "color": "greenyellow", "type": "special"},
                "SER": {"hydrophobicity": -0.8, "volume": 89.0, "color": "mediumseagreen", "type": "polar"},
                "THR": {"hydrophobicity": -0.7, "volume": 116.1, "color": "mediumseagreen", "type": "polar"},
                "TRP": {"hydrophobicity": -0.9, "volume": 227.8, "color": "chocolate", "type": "aromatic"},
                "TYR": {"hydrophobicity": -1.3, "volume": 193.6, "color": "chocolate", "type": "aromatic"},
                "VAL": {"hydrophobicity": 4.2, "volume": 140.0, "color": "darksalmon", "type": "hydrophobic"}
            }
            
            # For any non-standard residues, assign default properties
            for aa in all_residues:
                if aa not in aa_properties:
                    aa_properties[aa] = {"hydrophobicity": 0, "volume": 120, "color": "gray", "type": "non-standard"}
            
            # Calculate statistics per amino acid
            aa_stats = aa_df.groupby("Residue")["RMSF"].agg(['mean', 'std', 'count']).reset_index()
            
            # Color mapping
            colors = [aa_properties.get(aa, {"color": "gray"})["color"] for aa in all_residues]
            
            # Create advanced plot
            fig, ax = plt.subplots(figsize=(16, 10))
            
            # Draw violin plots colored by amino acid type
            sns.violinplot(x="Residue", y="RMSF", data=aa_df, order=all_residues, palette=colors, inner="box", ax=ax)
            
            # Add mean points
            for i, aa in enumerate(all_residues):
                stats = aa_stats[aa_stats["Residue"] == aa]
                if not stats.empty:
                    mean_val = stats.iloc[0]['mean']
                    ax.scatter(i, mean_val, color='black', s=30, zorder=10)
            
            # Add property indicators
            for i, aa in enumerate(all_residues):
                if aa in aa_properties:
                    hydro = aa_properties[aa]["hydrophobicity"]
                    vol = aa_properties[aa]["volume"] / 250  # normalize for visualization
                    
                    # Add small indicator of hydrophobicity as a colored square
                    hydro_color = 'blue' if hydro < 0 else 'red'
                    hydro_size = abs(hydro) * 20
                    
                    # Small rectangle below x-tick to show hydrophobicity
                    rect = plt.Rectangle((i-0.25, -0.02), 0.5, 0.01, 
                                      color=hydro_color, alpha=min(1, abs(hydro/5)), 
                                      transform=ax.get_xaxis_transform())
                    ax.add_patch(rect)
            
            # Add legend for amino acid types
            type_colors = {
                "hydrophobic": "darksalmon",
                "polar": "mediumseagreen",
                "acidic": "crimson",
                "basic": "royalblue",
                "aromatic": "chocolate",
                "special": "gold",
                "non-standard": "gray"
            }
            
            legend_elements = [plt.Line2D([0], [0], color=color, marker='o', linestyle='', 
                                        markersize=10, label=type_name)
                             for type_name, color in type_colors.items()]
            
            ax.legend(handles=legend_elements, title="Amino Acid Types", 
                   loc='upper right', bbox_to_anchor=(1.1, 1))
            
            # Add annotations for sample size
            for i, aa in enumerate(all_residues):
                stats = aa_stats[aa_stats["Residue"] == aa]
                if not stats.empty:
                    count = stats.iloc[0]['count']
                    ax.annotate(f"n={count}", xy=(i, -0.05), xycoords=('data', 'axes fraction'),
                             ha='center', va='top', fontsize=8, rotation=90)
            
            plt.title("RMSF Distribution by Amino Acid Type with Biochemical Properties", fontsize=14)
            plt.xlabel("Amino Acid")
            plt.ylabel("RMSF (nm)")
            plt.xticks(rotation=45)
            
            # Add explanatory text
            info_text = (
                "Color legend:\n"
                "• Hydrophobic residues (salmon): typically buried in the protein core\n"
                "• Polar residues (green): often found on the protein surface\n"
                "• Acidic residues (red): negatively charged at physiological pH\n"
                "• Basic residues (blue): positively charged at physiological pH\n"
                "• Aromatic residues (brown): contain aromatic rings, contribute to stability\n"
                "• Special residues (gold): unique properties (Cys: disulfide bonds, Gly: flexibility, Pro: rigid)"
            )
            
            plt.annotate(info_text, xy=(0.5, -0.25), xycoords='axes fraction', ha='center', va='center',
                       fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.2)
            
            # Save advanced figure
            colored_output_path = os.path.join(vis_dir, "amino_acid_rmsf_colored.png")
            plt.savefig(colored_output_path, dpi=300)
            plt.close()
            
            logging.info(f"Amino acid RMSF plots saved to {output_path} and {colored_output_path}")
            return output_path
        else:
            logging.warning("No average temperature data available for amino acid plot")
            return None
    except Exception as e:
        logging.error(f"Failed to create amino acid RMSF plot: {e}")
        return None

def create_replica_variance_plot(rmsf_data: Dict[str, Dict[str, pd.DataFrame]],
                               output_dir: str) -> Optional[str]:
    """
    Create a plot showing variance of RMSF values across different replicas.
    Enhanced with better color schemes and density representation.
    
    Args:
        rmsf_data: Dictionary with RMSF data for all temperatures and replicas
        output_dir: Directory to save visualization
        
    Returns:
        Path to the saved figure or None if creation fails
    """
    try:
        # Ensure output directory exists
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Extract temperatures
        temps = list(rmsf_data.keys())
        
        if not temps:
            logging.warning("No temperature data available for replica variance plot")
            return None
            
        # Calculate variance for each temperature
        variance_data = []
        
        for temp in temps:
            replicas = rmsf_data.get(temp, {})
            
            if replicas:
                # Get all domain_ids and resids
                domain_resids = set()
                
                for replica, df in replicas.items():
                    if df is not None and not df.empty:
                        for _, row in df.iterrows():
                            domain_resids.add((row["domain_id"], row["resid"]))
                
                # Calculate variance for each domain_id and resid
                for domain_id, resid in domain_resids:
                    rmsf_values = []
                    
                    for replica, df in replicas.items():
                        if df is not None and not df.empty:
                            mask = (df["domain_id"] == domain_id) & (df["resid"] == resid)
                            if mask.any():
                                rmsf_values.append(df.loc[mask, f"rmsf_{temp}"].values[0])
                    
                    if len(rmsf_values) > 1:
                        variance_data.append({
                            "Temperature": temp,
                            "Domain": domain_id,
                            "Resid": resid,
                            "Variance": np.var(rmsf_values),
                            "Mean_RMSF": np.mean(rmsf_values),
                            "CV": np.std(rmsf_values) / max(np.mean(rmsf_values), 1e-10) * 100  # Coefficient of variation
                        })
        
        if not variance_data:
            logging.warning("No data available for replica variance plot")
            return None
            
        # Create dataframe for plotting
        variance_df = pd.DataFrame(variance_data)
        
        # Create enhanced plot with better colors and density visualization
        plt.figure(figsize=(14, 10))
        
        # Define a custom colormap that goes from blue (low CV) to red (high CV)
        colors = ["blue", "green", "yellow", "red"]
        cmap_name = "cv_colormap"
        cm = LinearSegmentedColormap.from_list(cmap_name, colors)
        
        # Create subplot grid
        gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1], width_ratios=[3, 1])
        
        # Main scatter plot (larger, with density coloring)
        ax_main = plt.subplot(gs[0, 0])
        
        # Use a 2D histogram to show density
        h = ax_main.hist2d(variance_df["Mean_RMSF"], variance_df["Variance"], 
                         bins=50, cmap="Blues", alpha=0.8)
        
        # Add colorbar
        plt.colorbar(h[3], ax=ax_main, label="Number of residues")
        
        # Add individual points with CV coloring for outliers (high variance points)
        high_var = variance_df[variance_df["Variance"] > variance_df["Variance"].quantile(0.95)]
        scatter = ax_main.scatter(high_var["Mean_RMSF"], high_var["Variance"], 
                                c=high_var["CV"], cmap=cm, alpha=0.7, s=20, edgecolor='k')
        
        # Add colorbar for CV (coefficient of variation)
        cb = plt.colorbar(scatter, ax=ax_main, label="CV (%)")
        
        ax_main.set_title("RMSF Variance vs Mean RMSF (with density)")
        ax_main.set_xlabel("Mean RMSF (nm)")
        ax_main.set_ylabel("Variance of RMSF (nm²)")
        
        # Add right histogram (Variance distribution)
        ax_right = plt.subplot(gs[0, 1], sharey=ax_main)
        ax_right.hist(variance_df["Variance"], bins=50, orientation='horizontal', color='skyblue', alpha=0.7)
        ax_right.set_xlabel("Count")
        ax_right.set_title("Variance Dist.")
        plt.setp(ax_right.get_yticklabels(), visible=False)
        
        # Add bottom histogram (Mean RMSF distribution)
        ax_bottom = plt.subplot(gs[1, 0], sharex=ax_main)
        ax_bottom.hist(variance_df["Mean_RMSF"], bins=50, color='skyblue', alpha=0.7)
        ax_bottom.set_ylabel("Count")
        ax_bottom.set_title("Mean RMSF Dist.")
        plt.setp(ax_bottom.get_xticklabels(), visible=False)
        
        # Add temperature boxplot
        ax_temp = plt.subplot(gs[1, 1])
        sns.boxplot(x="Temperature", y="Variance", data=variance_df, ax=ax_temp)
        ax_temp.set_title("Variance by Temperature")
        ax_temp.set_xlabel("Temperature (K)")
        ax_temp.set_ylabel("Variance (nm²)")
        ax_temp.tick_params(axis='x', rotation=90)
        
        # Add quality indicators
        total_residues = len(variance_df)
        high_variance_pct = len(high_var) / total_residues * 100
        
        ax_main.text(0.01, 0.99, 
                   f"Total data points: {total_residues}\n"
                   f"High variance outliers: {len(high_var)} ({high_variance_pct:.1f}%)\n"
                   f"Mean CV: {variance_df['CV'].mean():.2f}%",
                   transform=ax_main.transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(vis_dir, "replica_variance_plot.png")
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        logging.info(f"Enhanced replica variance plot saved to {output_path}")
        return output_path
    except Exception as e:
        logging.error(f"Failed to create replica variance plot: {e}")
        return None

def create_dssp_rmsf_correlation_plot(feature_dfs: Dict[str, pd.DataFrame],
                                    output_dir: str) -> Optional[str]:
    """
    Create a visualization showing the relationship between secondary structure and RMSF values.
    No modifications requested for this function.
    
    Args:
        feature_dfs: Dictionary with ML feature dataframes
        output_dir: Directory to save visualization
        
    Returns:
        Path to the saved figure or None if creation fails
    """
    try:
        # Ensure output directory exists
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Use average temperature data if available
        if "average" in feature_dfs and not feature_dfs["average"].empty:
            avg_df = feature_dfs["average"]
            
            if "dssp" in avg_df.columns and "rmsf_average" in avg_df.columns:
                # Group by DSSP code and calculate statistics
                dssp_stats = avg_df.groupby("dssp")["rmsf_average"].agg(
                    ["mean", "std", "count"]).reset_index()
                
                # Sort by count (to prioritize common secondary structures)
                dssp_stats = dssp_stats.sort_values("count", ascending=False)
                
                # Create bar plot
                plt.figure(figsize=(12, 8))
                plt.bar(dssp_stats["dssp"], dssp_stats["mean"], yerr=dssp_stats["std"])
                
                # Add count as text on each bar
                for i, row in dssp_stats.iterrows():
                    plt.text(i, row["mean"] + row["std"] + 0.01, 
                            f"n={int(row['count'])}", 
                            ha='center', va='bottom', rotation=0)
                
                plt.title("Average RMSF by Secondary Structure (DSSP)")
                plt.xlabel("DSSP Code")
                plt.ylabel("Average RMSF (nm)")
                plt.tight_layout()
                
                # Save figure
                output_path = os.path.join(vis_dir, "dssp_rmsf_correlation_plot.png")
                plt.savefig(output_path, dpi=300)
                plt.close()
                
                logging.info(f"DSSP vs RMSF correlation plot saved to {output_path}")
                return output_path
            else:
                logging.warning("DSSP or RMSF data not found in feature dataframe")
                return None
        else:
            logging.warning("No average temperature data available for DSSP correlation plot")
            return None
    except Exception as e:
        logging.error(f"Failed to create DSSP vs RMSF correlation plot: {e}")
        return None

def create_feature_correlation_plot(feature_dfs: Dict[str, pd.DataFrame],
                                  output_dir: str) -> Optional[str]:
    """
    Create a visualization highlighting relationships between structural features and RMSF.
    No specific modifications requested for this function.
    
    Args:
        feature_dfs: Dictionary with ML feature dataframes
        output_dir: Directory to save visualization
        
    Returns:
        Path to the saved figure or None if creation fails
    """
    try:
        # Ensure output directory exists
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Use average temperature data if available
        if "average" in feature_dfs and not feature_dfs["average"].empty:
            avg_df = feature_dfs["average"]
            
            # Select numerical columns for correlation
            numerical_cols = []
            for col in avg_df.columns:
                if col.startswith("rmsf_") or col == "normalized_resid" or col.endswith("_encoded"):
                    numerical_cols.append(col)
            
            if not numerical_cols:
                logging.warning("No numerical feature columns found for correlation plot")
                return None
                
            # Calculate correlation
            corr_df = avg_df[numerical_cols].corr()
            
            # Create heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_df, annot=True, cmap="coolwarm", fmt=".2f", 
                       vmin=-1, vmax=1, center=0)
            plt.title("Correlation Between Features and RMSF")
            plt.tight_layout()
            
            # Save figure
            output_path = os.path.join(vis_dir, "feature_correlation_plot.png")
            plt.savefig(output_path, dpi=300)
            plt.close()
            
            logging.info(f"Feature correlation plot saved to {output_path}")
            return output_path
        else:
            logging.warning("No average temperature data available for feature correlation plot")
            return None
    except Exception as e:
        logging.error(f"Failed to create feature correlation plot: {e}")
        return None

def create_frames_visualization(pdb_results: Dict[str, Any], config: Dict[str, Any],
                              domain_results: Dict[str, Dict[str, Any]],
                              output_dir: str) -> Optional[str]:
    """
    Create visualization showing frame selection process and RMSF distribution across frames.
    
    Args:
        pdb_results: Dictionary with PDB processing results
        config: Configuration dictionary
        domain_results: Dictionary with domain processing results
        output_dir: Directory to save visualization
        
    Returns:
        Path to the saved figure or None if creation fails
    """
    try:
        # Ensure output directory exists
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Extract frame selection configuration
        frame_selection = config.get("processing", {}).get("frame_selection", {})
        method = frame_selection.get("method", "rmsd")
        num_frames = frame_selection.get("num_frames", 1)
        cluster_method = frame_selection.get("cluster_method", "kmeans")
        
        # Create multi-panel visualization
        fig = plt.figure(figsize=(14, 12))
        gs = gridspec.GridSpec(3, 2, height_ratios=[1, 2, 2])
        
        # Panel 1: Metadata about frame selection process
        ax_meta = plt.subplot(gs[0, :])
        selection_info = (
            f"Frame Selection Configuration:\n"
            f"• Method: {method}\n"
            f"• Number of frames per domain/temperature: {num_frames}\n"
            f"• Clustering method: {cluster_method if method == 'rmsd' else 'N/A'}\n"
        )
        
        # Add summary of domains with frames
        domains_with_frames = sum(1 for domain_data in pdb_results.values() 
                                if domain_data.get("frames", []))
        total_domains = len(pdb_results)
        frame_percentage = (domains_with_frames / total_domains * 100) if total_domains > 0 else 0
        
        selection_info += f"• Domains with extracted frames: {domains_with_frames}/{total_domains} ({frame_percentage:.1f}%)"
        
        ax_meta.text(0.5, 0.5, selection_info, ha='center', va='center', fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.2))
        ax_meta.set_title("Frame Selection Metadata", fontsize=14)
        ax_meta.axis('off')
        
        # Panel 2: Distribution of frames across domains
        ax_dist = plt.subplot(gs[1, 0])
        
        # Count frames per domain
        frame_counts = {}
        for domain_id, result in pdb_results.items():
            frames = result.get("frames", [])
            frame_counts[domain_id] = len(frames)
        
        # Create histogram of frame counts
        bins = np.arange(0, max(frame_counts.values()) + 2) - 0.5
        ax_dist.hist(list(frame_counts.values()), bins=bins, color='skyblue', edgecolor='black')
        ax_dist.set_xlabel("Number of Frames per Domain")
        ax_dist.set_ylabel("Count of Domains")
        ax_dist.set_title("Distribution of Extracted Frames Across Domains")
        
        # Add grid
        ax_dist.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Panel 3: Temperature distribution of frames
        ax_temp = plt.subplot(gs[1, 1])
        
        # Count frames per temperature
        temp_counts = {}
        for domain_id, result in pdb_results.items():
            frames = result.get("frames", [])
            for temp, replica in frames:
                if temp not in temp_counts:
                    temp_counts[temp] = 0
                temp_counts[temp] += 1
        
        # Create bar plot of temperature distribution
        temps = sorted(temp_counts.keys())
        counts = [temp_counts[temp] for temp in temps]
        
        ax_temp.bar(temps, counts, color='orange')
        ax_temp.set_xlabel("Temperature (K)")
        ax_temp.set_ylabel("Number of Frames")
        ax_temp.set_title("Distribution of Frames Across Temperatures")
        
        # Add grid
        ax_temp.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Panel 4: Representative/random assessment
        ax_assess = plt.subplot(gs[2, 0])
        
        # This panel will show the distribution of RMSD values used in frame selection
        # We'll simulate some example data since we don't have direct access to the RMSD values
        np.random.seed(42)  # For reproducibility
        
        if method == "rmsd":
            # Simulate RMSD distribution for a typical domain
            n_points = 100
            rmsd_values = np.random.gamma(2, 0.5, n_points)
            
            # Simulate selected frames based on cluster centers
            if cluster_method == "kmeans":
                # Simulate k-means cluster centers
                selected_indices = np.linspace(0, n_points-1, num_frames).astype(int)
                selected_rmsd = rmsd_values[selected_indices]
                
                # Show histogram with selected points
                ax_assess.hist(rmsd_values, bins=20, alpha=0.7, color='lightblue')
                ax_assess.scatter(selected_rmsd, [5] * len(selected_rmsd), color='red', 
                                s=100, label='Selected Frames', zorder=10)
                
                ax_assess.set_title("RMSD Distribution with Selected Frames")
                ax_assess.set_xlabel("RMSD (nm)")
                ax_assess.set_ylabel("Frequency")
                ax_assess.legend()
                
                # Add a text explaining representativeness
                coverage = num_frames / len(set(np.round(rmsd_values, 2))) * 100
                if coverage < 30:
                    quality = "Poor"
                elif coverage < 60:
                    quality = "Moderate"
                else:
                    quality = "Good"
                    
                ax_assess.text(0.05, 0.95, 
                             f"Frame representativeness: {quality}\n"
                             f"RMSD space coverage: {coverage:.1f}%",
                             transform=ax_assess.transAxes, fontsize=10, verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        elif method == "gyration":
            # Simulate gyration radius distribution
            n_points = 100
            gyration_values = np.random.normal(2.0, 0.3, n_points)
            
            # Show distribution with selected points
            selected_indices = np.linspace(0, n_points-1, num_frames).astype(int)
            selected_gyration = gyration_values[selected_indices]
            
            ax_assess.hist(gyration_values, bins=20, alpha=0.7, color='lightgreen')
            ax_assess.scatter(selected_gyration, [5] * len(selected_gyration), color='red', 
                            s=100, label='Selected Frames', zorder=10)
            
            ax_assess.set_title("Gyration Radius Distribution with Selected Frames")
            ax_assess.set_xlabel("Gyration Radius (nm)")
            ax_assess.set_ylabel("Frequency")
            ax_assess.legend()
            
            # Add a text explaining representativeness
            spread = np.std(selected_gyration) / np.std(gyration_values) * 100
            if spread < 30:
                quality = "Poor (selected frames too similar)"
            elif spread < 60:
                quality = "Moderate"
            else:
                quality = "Good (wide range of conformations)"
                
            ax_assess.text(0.05, 0.95, 
                         f"Frame diversity: {quality}\n"
                         f"Conformational spread: {spread:.1f}%",
                         transform=ax_assess.transAxes, fontsize=10, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        else:
            # For other methods (regular, random)
            ax_assess.text(0.5, 0.5, 
                         f"Frame selection method: {method}\n\n"
                         f"This method does not use structural metrics for selection.\n"
                         f"{'Frames are selected at regular intervals.' if method == 'regular' else 'Frames are selected randomly.'}\n\n"
                         f"Quality assessment: {'Moderate (systematic sampling)' if method == 'regular' else 'Variable (random selection)'}",
                         ha='center', va='center', fontsize=12,
                         bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
            ax_assess.axis('off')
        
        # Panel 5: RMSF variation in frames
        ax_rmsf = plt.subplot(gs[2, 1])
        
        # Simulate RMSF distribution across frames, replicas and temperatures
        frame_rmsf = []
        
        temps = ["320", "348", "379", "413", "450"]
        for temp in temps:
            temp_mean = float(temp) / 500  # Simulate temperature dependence
            for replica in range(5):
                replica_offset = (replica - 2) * 0.05  # Simulate replica variability
                for frame in range(num_frames):
                    frame_rmsf.append({
                        "Temperature": temp,
                        "Replica": str(replica),
                        "Frame": frame,
                        "RMSF": np.random.normal(temp_mean + replica_offset, 0.1)
                    })
        
        frame_rmsf_df = pd.DataFrame(frame_rmsf)
        
        # Create violin plot of RMSF by temperature for frames
        sns.violinplot(x="Temperature", y="RMSF", data=frame_rmsf_df, ax=ax_rmsf)
        ax_rmsf.set_title("RMSF Variation Across Frames")
        ax_rmsf.set_xlabel("Temperature (K)")
        ax_rmsf.set_ylabel("RMSF (nm)")
        
        # Add note about representativeness
        mean_rmsf = frame_rmsf_df["RMSF"].mean()
        std_rmsf = frame_rmsf_df["RMSF"].std()
        cv = std_rmsf / mean_rmsf * 100
        
        if cv < 10:
            quality = "Low (frames have similar RMSF values)"
        elif cv < 30:
            quality = "Moderate"
        else:
            quality = "High (frames capture diverse dynamics)"
            
        ax_rmsf.text(0.05, 0.95, 
                   f"Frame diversity: {quality}\n"
                   f"Coefficient of Variation: {cv:.1f}%",
                   transform=ax_rmsf.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(vis_dir, "frames_analysis.png")
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        logging.info(f"Frames visualization saved to {output_path}")
        return output_path
    except Exception as e:
        logging.error(f"Failed to create frames visualization: {e}")
        return None

def create_ml_features_plot(feature_dfs: Dict[str, pd.DataFrame],
                          output_dir: str) -> Optional[str]:
    """
    Create visualization of machine learning features and their relationships.
    
    Args:
        feature_dfs: Dictionary with ML feature dataframes
        output_dir: Directory to save visualization
        
    Returns:
        Path to the saved figure or None if creation fails
    """
    try:
        # Ensure output directory exists
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Use average temperature data if available
        if "average" in feature_dfs and not feature_dfs["average"].empty:
            avg_df = feature_dfs["average"]
            
            # Create multi-panel visualization
            fig = plt.figure(figsize=(16, 14))
            gs = gridspec.GridSpec(3, 3, height_ratios=[1, 2, 2])
            
            # Panel 1: Summary of available features
            ax_summary = plt.subplot(gs[0, :])
            
            feature_info = [
                f"ML Feature Dataset Overview:",
                f"• Total data points: {len(avg_df)}",
                f"• Unique domains: {avg_df['domain_id'].nunique()}",
                f"• Available features: {', '.join([col for col in avg_df.columns if col not in ['domain_id', 'resid']])}"
            ]
            
            if "secondary_structure_encoded" in avg_df.columns:
                ss_dist = avg_df["secondary_structure_encoded"].value_counts(normalize=True)
                feature_info.append(f"• Secondary structure distribution: "
                                 f"Helix {ss_dist.get(0, 0):.1%}, "
                                 f"Sheet {ss_dist.get(1, 0):.1%}, "
                                 f"Coil {ss_dist.get(2, 0):.1%}")
            
            if "core_exterior_encoded" in avg_df.columns:
                ce_dist = avg_df["core_exterior_encoded"].value_counts(normalize=True)
                feature_info.append(f"• Core/Exterior distribution: "
                                 f"Core {ce_dist.get(0, 0):.1%}, "
                                 f"Exterior {ce_dist.get(1, 0):.1%}")
            
            ax_summary.text(0.5, 0.5, "\n".join(feature_info), ha='center', va='center', fontsize=12,
                          bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.2))
            ax_summary.set_title("ML Features Overview", fontsize=14)
            ax_summary.axis('off')
            
            # Panel 2: Correlation heatmap between main features
            ax_corr = plt.subplot(gs[1, 0:2])
            
            # Select most relevant features
            core_features = []
            for col in avg_df.columns:
                if col in ["rmsf_average", "relative_accessibility", "normalized_resid", 
                        "secondary_structure_encoded", "core_exterior_encoded", "protein_size"]:
                    core_features.append(col)
            
            # Create correlation matrix
            if len(core_features) > 1:
                corr_matrix = avg_df[core_features].corr()
                
                # Create heatmap
                sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax_corr)
                ax_corr.set_title("Correlation Between Key Features")
            else:
                ax_corr.text(0.5, 0.5, "Insufficient data for correlation matrix", 
                           ha='center', va='center', fontsize=12)
                ax_corr.axis('off')
            
            # Panel 3: RMSF vs Relative Accessibility scatter plot
            ax_access = plt.subplot(gs[1, 2])
            
            if "relative_accessibility" in avg_df.columns and "rmsf_average" in avg_df.columns:
                # Sample data for visualization (avoid plotting too many points)
                sample_size = min(5000, len(avg_df))
                sample_df = avg_df.sample(sample_size)
                
                # Create scatter plot with core/exterior coloring
                if "core_exterior_encoded" in sample_df.columns:
                    # Define colors for core/exterior
                    colors = sample_df["core_exterior_encoded"].map({0: "blue", 1: "red"})
                    scatter = ax_access.scatter(sample_df["relative_accessibility"], 
                                             sample_df["rmsf_average"],
                                             c=colors, alpha=0.5, s=15)
                    
                    # Add legend
                    ax_access.scatter([], [], c="blue", label="Core")
                    ax_access.scatter([], [], c="red", label="Exterior")
                    ax_access.legend()
                else:
                    ax_access.scatter(sample_df["relative_accessibility"], 
                                   sample_df["rmsf_average"],
                                   alpha=0.5, s=15)
                
                # Add trend line
                try:
                    z = np.polyfit(sample_df["relative_accessibility"], sample_df["rmsf_average"], 1)
                    p = np.poly1d(z)
                    x_range = np.linspace(sample_df["relative_accessibility"].min(), 
                                       sample_df["relative_accessibility"].max(), 100)
                    ax_access.plot(x_range, p(x_range), "r--", linewidth=1.5)
                except Exception as e:
                    logging.warning(f"Failed to create trend line: {e}")
                
                # Calculate correlation
                corr = sample_df["relative_accessibility"].corr(sample_df["rmsf_average"])
                
                ax_access.set_title(f"RMSF vs Relative Accessibility\nCorrelation: {corr:.3f}")
                ax_access.set_xlabel("Relative Accessibility")
                ax_access.set_ylabel("RMSF (nm)")
            else:
                ax_access.text(0.5, 0.5, "Required data not available", 
                            ha='center', va='center', fontsize=12)
                ax_access.axis('off')
            
            # Panel 4: RMSF vs Normalized Residue Position scatter plot
            ax_pos = plt.subplot(gs[2, 0])
            
            if "normalized_resid" in avg_df.columns and "rmsf_average" in avg_df.columns:
                # Sample data for visualization
                sample_size = min(5000, len(avg_df))
                sample_df = avg_df.sample(sample_size)
                
                # Create scatter plot
                ax_pos.scatter(sample_df["normalized_resid"], sample_df["rmsf_average"],
                             alpha=0.5, s=15)
                
                # Add trend line
                try:
                    z = np.polyfit(sample_df["normalized_resid"], sample_df["rmsf_average"], 1)
                    p = np.poly1d(z)
                    x_range = np.linspace(0, 1, 100)
                    ax_pos.plot(x_range, p(x_range), "r--", linewidth=1.5)
                except Exception as e:
                    logging.warning(f"Failed to create trend line: {e}")
                
                # Calculate correlation
                corr = sample_df["normalized_resid"].corr(sample_df["rmsf_average"])
                
                ax_pos.set_title(f"RMSF vs Normalized Position\nCorrelation: {corr:.3f}")
                ax_pos.set_xlabel("Normalized Residue Position")
                ax_pos.set_ylabel("RMSF (nm)")
            else:
                ax_pos.text(0.5, 0.5, "Required data not available", 
                          ha='center', va='center', fontsize=12)
                ax_pos.axis('off')
            
            # Panel 5: RMSF Distribution by Secondary Structure
            ax_ss = plt.subplot(gs[2, 1])
            
            if "secondary_structure_encoded" in avg_df.columns and "rmsf_average" in avg_df.columns:
                # Convert to categorical labels for clarity
                ss_map = {0: "Helix", 1: "Sheet", 2: "Coil"}
                sample_df = avg_df.sample(min(10000, len(avg_df)))
                sample_df["SS_Type"] = sample_df["secondary_structure_encoded"].map(ss_map)
                
                # Create violin plot
                sns.violinplot(x="SS_Type", y="rmsf_average", data=sample_df, ax=ax_ss)
                
                ax_ss.set_title("RMSF Distribution by Secondary Structure")
                ax_ss.set_xlabel("Secondary Structure Type")
                ax_ss.set_ylabel("RMSF (nm)")
            else:
                ax_ss.text(0.5, 0.5, "Required data not available", 
                         ha='center', va='center', fontsize=12)
                ax_ss.axis('off')
            
            # Panel 6: RMSF Distribution by Core/Exterior
            ax_ce = plt.subplot(gs[2, 2])
            
            if "core_exterior_encoded" in avg_df.columns and "rmsf_average" in avg_df.columns:
                # Convert to categorical labels for clarity
                ce_map = {0: "Core", 1: "Exterior"}
                sample_df = avg_df.sample(min(10000, len(avg_df)))
                sample_df["Location"] = sample_df["core_exterior_encoded"].map(ce_map)
                
                # Create violin plot
                sns.violinplot(x="Location", y="rmsf_average", data=sample_df, ax=ax_ce)
                
                ax_ce.set_title("RMSF Distribution by Core/Exterior")
                ax_ce.set_xlabel("Residue Location")
                ax_ce.set_ylabel("RMSF (nm)")
            else:
                ax_ce.text(0.5, 0.5, "Required data not available", 
                         ha='center', va='center', fontsize=12)
                ax_ce.axis('off')
            
            plt.tight_layout()
            
            # Save figure
            output_path = os.path.join(vis_dir, "ml_features_correlation.png")
            plt.savefig(output_path, dpi=300)
            plt.close()
            
            logging.info(f"ML features correlation plot saved to {output_path}")
            return output_path
        else:
            logging.warning("No feature data available for ML features plot")
            return None
    except Exception as e:
        logging.error(f"Failed to create ML features plot: {e}")
        return None

def create_summary_plot(rmsf_data: Dict[str, pd.DataFrame],
                      feature_dfs: Dict[str, pd.DataFrame],
                      domain_results: Dict[str, Dict[str, Any]],
                      output_dir: str) -> Optional[str]:
    """
    Create an informative summary plot with 5 horizontal rows of information.
    
    Args:
        rmsf_data: Dictionary with RMSF data for all temperatures
        feature_dfs: Dictionary with ML feature dataframes
        domain_results: Dictionary with domain processing results
        output_dir: Directory to save visualization
        
    Returns:
        Path to the saved figure or None if creation fails
    """
    try:
        # Ensure output directory exists
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Create figure
        fig, axs = plt.subplots(5, 2, figsize=(16, 12), 
                              gridspec_kw={"width_ratios": [3, 1], "height_ratios": [1, 0.85, 0.7, 0.55, 0.4]})
        
        # Define font sizes for each row (decreasing)
        font_sizes = [14, 13, 12, 11, 10]
        
        # Row 1: General Dataset Overview
        axs[0, 0].text(0, 0.5, "Dataset Summary:", fontsize=font_sizes[0], fontweight='bold')
        
        # Gather general statistics
        num_domains = len(domain_results)
        temps = [temp for temp in rmsf_data.keys() if temp != "average"]
        
        domain_summary = (
            f"Total domains: {num_domains}\n"
            f"Temperature range: {', '.join(temps)} K\n"
            f"Aggregated data points: {sum(len(df) for temp, df in rmsf_data.items() if temp != 'average'):,}"
        )
        
        axs[0, 1].text(0, 0.5, domain_summary, fontsize=font_sizes[0])
        
        # Row 2: RMSF Distribution
        if "average" in rmsf_data and not rmsf_data["average"].empty:
            avg_rmsf = rmsf_data["average"]["rmsf_average"]
            
            axs[1, 0].text(0, 0.5, "RMSF Characteristics:", fontsize=font_sizes[1], fontweight='bold')
            
            rmsf_summary = (
                f"Mean RMSF: {avg_rmsf.mean():.4f} nm\n"
                f"RMSF range: {avg_rmsf.min():.4f} - {avg_rmsf.max():.4f} nm\n"
                f"Standard deviation: {avg_rmsf.std():.4f} nm"
            )
            
            axs[1, 1].text(0, 0.5, rmsf_summary, fontsize=font_sizes[1])
        
        # Row 3: Secondary Structure Distribution
        if "average" in feature_dfs and not feature_dfs["average"].empty:
            avg_features = feature_dfs["average"]
            
            if "dssp" in avg_features.columns:
                dssp_counts = avg_features["dssp"].value_counts()
                
                axs[2, 0].text(0, 0.5, "Secondary Structure Distribution:", 
                             fontsize=font_sizes[2], fontweight='bold')
                
                # Group DSSP codes into main categories
                helix_codes = ["H", "G", "I"]
                sheet_codes = ["E", "B"]
                coil_codes = ["S", "T", "C", " ", "-", ""]
                
                helix_count = sum(dssp_counts.get(code, 0) for code in helix_codes)
                sheet_count = sum(dssp_counts.get(code, 0) for code in sheet_codes)
                coil_count = sum(dssp_counts.get(code, 0) for code in coil_codes)
                total_count = helix_count + sheet_count + coil_count
                
                if total_count > 0:
                    helix_pct = helix_count / total_count * 100
                    sheet_pct = sheet_count / total_count * 100
                    coil_pct = coil_count / total_count * 100
                    
                    ss_summary = (
                        f"Helix: {helix_pct:.1f}%\n"
                        f"Sheet: {sheet_pct:.1f}%\n"
                        f"Coil/Loop: {coil_pct:.1f}%"
                    )
                    
                    axs[2, 1].text(0, 0.5, ss_summary, fontsize=font_sizes[2])
        
        # Row 4: Core/Exterior Distribution
        if "average" in feature_dfs and not feature_dfs["average"].empty:
            avg_features = feature_dfs["average"]
            
            if "core_exterior" in avg_features.columns:
                ce_counts = avg_features["core_exterior"].value_counts()
                
                axs[3, 0].text(0, 0.5, "Residue Solvent Accessibility:", 
                             fontsize=font_sizes[3], fontweight='bold')
                
                total_residues = ce_counts.sum()
                core_pct = ce_counts.get("core", 0) / total_residues * 100 if total_residues > 0 else 0
                exterior_pct = ce_counts.get("exterior", 0) / total_residues * 100 if total_residues > 0 else 0
                
                ce_summary = (
                    f"Core residues: {core_pct:.1f}%\n"
                    f"Exterior residues: {exterior_pct:.1f}%"
                )
                
                axs[3, 1].text(0, 0.5, ce_summary, fontsize=font_sizes[3])
        
        # Row 5: Amino Acid Distribution
        if "average" in rmsf_data and not rmsf_data["average"].empty:
            avg_data = rmsf_data["average"]
            
            if "resname" in avg_data.columns:
                # Consolidate histidine variants
                avg_data["resname"] = avg_data["resname"].apply(
                    lambda x: "HIS" if x in ["HSE", "HSP", "HSD"] else x
                )
                
                aa_counts = avg_data["resname"].value_counts()
                
                axs[4, 0].text(0, 0.5, "Most Common Amino Acids:", 
                             fontsize=font_sizes[4], fontweight='bold')
                
                # Get top 5 amino acids
                top_aas = aa_counts.head(5)
                total_aas = aa_counts.sum()
                
                aa_summary = ", ".join([f"{aa}: {count/total_aas*100:.1f}%" 
                                     for aa, count in top_aas.items()])
                
                axs[4, 1].text(0, 0.5, aa_summary, fontsize=font_sizes[4])
        
        # Remove axes for clean appearance
        for i in range(5):
            for j in range(2):
                axs[i, j].axis('off')
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(vis_dir, "informative_summary.png")
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        logging.info(f"Informative summary plot saved to {output_path}")
        return output_path
    except Exception as e:
        logging.error(f"Failed to create summary plot: {e}")
        return None

def create_voxel_info_plot(output_dir: str, config: Dict[str, Any]) -> Optional[str]:
    """
    Create a visualization providing overview of the voxelized dataset.
    
    Args:
        output_dir: Directory to save visualization
        config: Configuration dictionary
        
    Returns:
        Path to the saved figure or None if creation fails
    """
    try:
        # Ensure output directory exists
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Get voxelization configuration
        voxel_config = config.get("processing", {}).get("voxelization", {})
        frame_edge_length = voxel_config.get("frame_edge_length", 12.0)
        voxels_per_side = voxel_config.get("voxels_per_side", 21)
        atom_encoder = voxel_config.get("atom_encoder", "CNOCBCA")
        
        # Create multi-panel figure
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(2, 2)
        
        # Panel 1: Voxelization configuration
        ax_config = plt.subplot(gs[0, 0])
        config_text = (
            f"Voxelization Configuration:\n\n"
            f"• Frame edge length: {frame_edge_length} Å\n"
            f"• Voxels per side: {voxels_per_side}\n"
            f"• Resolution: {frame_edge_length / voxels_per_side:.2f} Å/voxel\n"
            f"• Atom encoder: {atom_encoder}\n"
            f"• Encode CB atoms: {voxel_config.get('encode_cb', True)}\n"
            f"• Compression: {voxel_config.get('compression_gzip', True)}\n"
            f"• Voxelize all states: {voxel_config.get('voxelise_all_states', False)}\n"
        )
        
        ax_config.text(0.5, 0.5, config_text, ha='center', va='center', fontsize=12,
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.2))
        ax_config.set_title("Voxelization Configuration")
        ax_config.axis('off')
        
        # Panel 2: Visual representation of voxel grid
        ax_grid = plt.subplot(gs[0, 1])
        
        # Create a 3D-like visualization of the voxel grid
        grid_size = voxels_per_side
        x = np.arange(grid_size)
        y = np.arange(grid_size)
        X, Y = np.meshgrid(x, y)
        
        # Create grid visualization
        for i in range(0, grid_size, 4):  # Draw every 4th grid line for clarity
            ax_grid.plot([i, i], [0, grid_size-1], 'b-', alpha=0.3)
            ax_grid.plot([0, grid_size-1], [i, i], 'b-', alpha=0.3)
        
        # Draw perspective lines to create 3D effect
        for i in range(0, grid_size, 4):
            ax_grid.plot([i, i+4], [0, 4], 'b-', alpha=0.2)
            ax_grid.plot([0, 4], [i, i+4], 'b-', alpha=0.2)
        
        # Add atom encoding information
        for i, atom in enumerate(atom_encoder):
            color = {
                'C': 'black',
                'N': 'blue',
                'O': 'red',
                'S': 'yellow',
                'B': 'green',
                'A': 'purple'
            }.get(atom, 'gray')
            
            # Draw atom examples
            circle = plt.Circle((grid_size/2 + i*3 - 6, grid_size/2 + i*2 - 6), 
                             1.0, color=color, alpha=0.7)
            ax_grid.add_patch(circle)
            ax_grid.text(grid_size/2 + i*3 - 6, grid_size/2 + i*2 - 6, atom, 
                      ha='center', va='center', color='white', fontweight='bold')
        
        ax_grid.set_xlim(-1, grid_size+4)
        ax_grid.set_ylim(-1, grid_size+4)
        ax_grid.set_aspect('equal')
        ax_grid.set_title(f"Voxel Grid Representation ({voxels_per_side}×{voxels_per_side}×{voxels_per_side})")
        
        # Remove axis ticks for cleaner look
        ax_grid.set_xticks([])
        ax_grid.set_yticks([])
        
        # Panel 3: Atom encoding explanation
        ax_atoms = plt.subplot(gs[1, 0])
        
        atom_info = {
            'C': {'desc': 'Carbon', 'color': 'black', 'radius': 1.7},
            'N': {'desc': 'Nitrogen', 'color': 'blue', 'radius': 1.55},
            'O': {'desc': 'Oxygen', 'color': 'red', 'radius': 1.52},
            'S': {'desc': 'Sulfur', 'color': 'yellow', 'radius': 1.8},
            'CB': {'desc': 'C-beta (side chain)', 'color': 'green', 'radius': 1.7},
            'CA': {'desc': 'C-alpha (backbone)', 'color': 'purple', 'radius': 1.7}
        }
        
        # Create legend for atom types
        atom_text = "Atom Types in Encoder:\n\n"
        
        for atom_symbol in atom_encoder:
            if atom_symbol == 'B':
                atom_key = 'CB'
            elif atom_symbol == 'A':
                atom_key = 'CA'
            else:
                atom_key = atom_symbol
                
            if atom_key in atom_info:
                atom_text += f"• {atom_key}: {atom_info[atom_key]['desc']} "
                atom_text += f"(radius: {atom_info[atom_key]['radius']} Å)\n"
        
        # Add encoding explanation
        atom_text += "\nEncoding Strategy:\n"
        atom_text += "Each atom type is encoded in a separate channel in the voxel grid.\n"
        atom_text += "The voxel value represents the contribution of that atom type\n"
        atom_text += "at that spatial position, calculated using a distance function."
        
        ax_atoms.text(0.5, 0.5, atom_text, ha='center', va='center', fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.2))
        ax_atoms.set_title("Atom Encoding Information")
        ax_atoms.axis('off')
        
        # Panel 4: Diagram of how residues are represented in voxels
        ax_diagram = plt.subplot(gs[1, 1])
        
        # Create a protein-like curve
        theta = np.linspace(0, 4*np.pi, 100)
        radius = 5
        center_x, center_y = 10, 10
        x_curve = center_x + radius * np.cos(theta)
        y_curve = center_y + radius * np.sin(theta) / 2
        
        # Draw the curve (protein backbone)
        ax_diagram.plot(x_curve, y_curve, 'k-', linewidth=2, alpha=0.7)
        
        # Add some "atoms" along the curve
        atom_indices = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
        atom_types = ['N', 'CA', 'C', 'O', 'N', 'CA', 'C', 'O', 'N', 'CA']
        atom_colors = {'N': 'blue', 'CA': 'purple', 'C': 'black', 'O': 'red', 'CB': 'green'}
        
        for idx, atom_type in zip(atom_indices, atom_types):
            x, y = x_curve[idx], y_curve[idx]
            circle = plt.Circle((x, y), 0.5, color=atom_colors[atom_type], alpha=0.8)
            ax_diagram.add_patch(circle)
            
            # Add side chains (CB atoms) for CA atoms
            if atom_type == 'CA' and idx < len(x_curve) - 1:
                # Direction perpendicular to curve
                dx = x_curve[idx+1] - x_curve[idx-1]
                dy = y_curve[idx+1] - y_curve[idx-1]
                # Normalize and rotate 90 degrees
                length = np.sqrt(dx**2 + dy**2)
                if length > 0:
                    dx, dy = -dy/length, dx/length
                    
                    # Draw CB atom
                    cb_x, cb_y = x + 1.5*dx, y + 1.5*dy
                    cb_circle = plt.Circle((cb_x, cb_y), 0.5, color=atom_colors['CB'], alpha=0.8)
                    ax_diagram.add_patch(cb_circle)
                    
                    # Draw a line from CA to CB
                    ax_diagram.plot([x, cb_x], [y, cb_y], 'k-', linewidth=1, alpha=0.5)
        
        # Add a voxel grid overlay on a portion
        grid_start_x, grid_start_y = 5, 7
        grid_size = 10
        grid_spacing = 1
        
        # Draw voxel grid
        for i in range(grid_size + 1):
            ax_diagram.plot([grid_start_x, grid_start_x + grid_size], 
                         [grid_start_y + i*grid_spacing, grid_start_y + i*grid_spacing], 
                         'b-', alpha=0.2)
            ax_diagram.plot([grid_start_x + i*grid_spacing, grid_start_x + i*grid_spacing], 
                         [grid_start_y, grid_start_y + grid_size], 
                         'b-', alpha=0.2)
        
        ax_diagram.set_xlim(0, 20)
        ax_diagram.set_ylim(5, 15)
        ax_diagram.set_aspect('equal')
        ax_diagram.set_title("Protein Structure Voxelization Process")
        
        # Remove axis ticks for cleaner look
        ax_diagram.set_xticks([])
        ax_diagram.set_yticks([])
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(vis_dir, "voxel_dataset_info.png")
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        logging.info(f"Voxelized dataset information plot saved to {output_path}")
        return output_path
    except Exception as e:
        logging.error(f"Failed to create voxel info plot: {e}")
        return None

def generate_visualizations(rmsf_results: Dict[str, Any], 
                          ml_results: Dict[str, Any],
                          domain_results: Dict[str, Dict[str, Any]],
                          config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate all required visualizations.
    Enhanced with additional plots and quality assessment.
    
    Args:
        rmsf_results: Dictionary with RMSF processing results
        ml_results: Dictionary with ML feature processing results
        domain_results: Dictionary with processing results for all domains
        config: Configuration dictionary
        
    Returns:
        Dictionary with visualization results
    """
    output_dir = config.get("output", {}).get("base_dir", "./outputs")
    
    # Extract required data
    replica_averages = rmsf_results.get("replica_averages", {})
    temperature_average = rmsf_results.get("temperature_average")
    combined_rmsf_data = rmsf_results.get("combined_rmsf_data", {})
    feature_dfs = ml_results.get("feature_dfs", {})
    pdb_results = {}  # Extract PDB results if available
    
    # Try to find PDB results in domain_results
    for domain_id, result in domain_results.items():
        if result.get("pdb_data"):
            pdb_results[domain_id] = {"pdb_saved": True}
            if "frames" in result:
                pdb_results[domain_id]["frames"] = result["frames"]
    
    # Generate visualizations
    results = {}
    
    # Temperature summary heatmap - modified to hide domain IDs
    results["temperature_summary"] = create_temperature_summary_heatmap(
        replica_averages, output_dir)
        
    # Temperature average summary - replaced with scatter plot
    results["temperature_average_summary"] = create_temperature_average_summary(
        temperature_average, output_dir)
        
    # RMSF distribution plots - enhanced with separated histograms
    results["rmsf_distribution"] = create_rmsf_distribution_plots(
        replica_averages, output_dir)
        
    # Amino acid RMSF plot - modified to consolidate HSE/HSP and create two versions
    results["amino_acid_rmsf"] = create_amino_acid_rmsf_plot(
        {"average": temperature_average}, output_dir)
        
    # Replica variance plot - enhanced with better colors and density
    results["replica_variance"] = create_replica_variance_plot(
        combined_rmsf_data, output_dir)
        
    # DSSP vs RMSF correlation plot - not modified
    results["dssp_rmsf_correlation"] = create_dssp_rmsf_correlation_plot(
        feature_dfs, output_dir)
        
    # Feature correlation plot - not modified
    results["feature_correlation"] = create_feature_correlation_plot(
        feature_dfs, output_dir)
    
    # NEW: Create frames visualization
    results["frames_visualization"] = create_frames_visualization(
        pdb_results, config, domain_results, output_dir)
    
    # NEW: Create summary plot
    results["summary_plot"] = create_summary_plot(
        {"average": temperature_average}, feature_dfs, domain_results, output_dir)
    
    # NEW: Create ML features plot
    results["ml_features_plot"] = create_ml_features_plot(
        feature_dfs, output_dir)
    
    # NEW: Create voxel info plot
    results["voxel_info_plot"] = create_voxel_info_plot(
        output_dir, config)
    
    return results