#!/usr/bin/env python3
"""
Minion Cluster Search Pipeline
================================

A flexible pipeline for analyzing MinION sequencing data to identify mutations and clusters
in genomic samples. Supports both signature-based mutation analysis and de novo clustering approaches.

The pipeline consists of three main steps:
1. BAM processing - Processes raw MinION sequencing data
2. Alignment - Aligns processed sequences to a reference
3. Analysis - Either signature-based mutation detection or de novo clustering

Prerequisites:
--------------    
Local Python Package Dependencies:

process_bam_data.py: BAM file processing functions
batch_align_augur.py: Contains run_alignment() function
fasta_mutation_selector.py: Contains run_mutation_selector() function
denovo_clustering_preprocessing.py
denovo_clustering_analysis.py: Core clustering analysis functions


Required Python Packages:

numpy>=1.19.0
pandas>=1.2.0
scipy>=1.6.0
matplotlib>=3.3.0
seaborn>=0.11.0
scikit-learn>=0.24.0
Bio>=1.78
ete3>=3.1.2
networkx>=2.5
psutil>=5.8.0


Input parameters:
-----------------
Required for all analyses:
barcodes_file   : Path to CSV file with "sample" and "barcode" columns
output_directory: Directory where results will be saved (created if doesn't exist)
reference_fasta : Path to reference FASTA file for alignment
bam_directory   : Directory containing BAM files (named as sample_name.bam)
split-by-signature
split-denovo

Required for specific analyses:
mutations_file  : Mutations to search for (required for --split-by-signature only)

Optional parameters:
mutation_purity : Mutation threshold [0-1], default=1.0 (only with --split-by-signature)
method         : Clustering method choice [spectral/hierphylo/quasiseq/compare_methods],
default='spectral' (only with --split-denovo)

    
Output:
-------
Results will be saved in the following directory structure:
- 01.BAM_Processing/
- 02.Alignment/
- 03.Split_by_signature/ (when using --split-by-signature)
  or
- 03.Split_denovo/ (when using --split-denovo)


Usage:
------
From command line:
# For signature-based splitting
python minion_cluster_search.py --barcodes_file <barcodes.csv> 
--mutations_file <mutations.txt> 
--output_directory <output_dir> 
--reference_fasta <ref.fasta> 
--bam_directory <bam_dir> 
--split-by-signature 
[--mutation_purity <float>]


# For de novo clustering
python minion_cluster_search.py --barcodes_file <barcodes.csv> \
                              --output_directory <output_dir> \
                              --reference_fasta <ref.fasta> \
                              --bam_directory <bam_dir> \
                              --split-denovo \
                              --fasta <sequences.fasta> \
                              [--method <clustering_method>]
    python find_mutations_in_samples.py input_dir --split-denovo [other options]

From Python:
from minion_cluster_search import run_pipeline

# For signature-based splitting
run_pipeline(
    barcodes_file="path/to/barcodes.csv",
    mutations_file="path/to/mutations.txt",
    output_directory="path/to/output",
    reference_fasta="path/to/reference.fasta",
    bam_directory="path/to/bam/files",
    analysis_type="signature",
    mutation_purity=0.8  # optional
)

# For de novo clustering
run_pipeline(
    barcodes_file="path/to/barcodes.csv",
    output_directory="path/to/output",
    reference_fasta="path/to/reference.fasta",
    bam_directory="path/to/bam/files",
    analysis_type="denovo",
    fasta="path/to/sequences.fasta",
    method="spectral"  # optional, default="spectral"
)    
 

Author: Noa Bar-Liss
Version: 1.0.0
Date: February 2025

"""

import os
import sys
import shutil
import pandas as pd
from pathlib import Path
import logging
import argparse
# local_functions_path = ".../minion_cluster_search/scripts"

# Add the folder containing the specific functions to the Python path
# sys.path.append(local_functions_path)
# from batch_align_augur import run_alignment
# import process_bam_data
# from fasta_mutation_selector import run_mutation_selector 
# from denovo_clustering_pipeline import ClusteringPipeline

from utils.batch_align_augur import run_alignment
from pipelines import process_bam_data
from pipelines import fasta_mutation_selector
from pipelines.denovo_clustering_pipeline import ClusteringPipeline


def parse_arguments():
    """Parse command line arguments with detailed help messages"""
    parser = argparse.ArgumentParser(
        description="""
Process BAM files through a pipeline that includes:
1. BAM file processing
2. Sequence alignment
3. Mutation search (by footprint or denovo)

This script takes a CSV file containing sample information and processes each sample
through multiple stages, creating organized output directories for each step.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Common required arguments
    parser.add_argument(
        '--barcodes_file',
        required=True,
        help='Path to CSV file containing columns "sample" and "barcode"'
    )
    
    parser.add_argument(
        '--output_directory',
        required=True,
        help='Path to the top-level output directory where sample folders will be created'
    )
    
    parser.add_argument(
        '--reference_fasta',
        required=True,
        help='Path to the reference FASTA file'
    )
    
    parser.add_argument(
        '--bam_directory',
        required=True,
        help='Path to directory containing BAM files (each named as sample_name.bam)'
    )
    
    # Analysis type selection (mutually exclusive)
    analysis_group = parser.add_mutually_exclusive_group(required=True)
    analysis_group.add_argument(
        "-sbs", "--split-by-signature",
        action="store_true",
        help="Perform signature-based splitting analysis"
    )
    analysis_group.add_argument(
        "-sd", "--split-denovo",
        action="store_true",
        help="Perform de novo clustering analysis"
    )

    # Create argument groups for method-specific arguments
    signature_group = parser.add_argument_group('Signature-based analysis arguments')
    signature_group.add_argument(
        '--mutations_file',
        help='Path to file containing mutations to search for (required for --split-by-signature)'
    )
    signature_group.add_argument(
        '--mutation_purity',
        type=float,
        help='Mutation purity threshold for mutation search (optional, only for --split-by-signature)'
    )

    denovo_group = parser.add_argument_group('De novo clustering arguments')
    denovo_group.add_argument(
        "--method",
        default='spectral',
        choices=['spectral', 'hierphylo', 'quasiseq', 'compare_methods'],
        help='Clustering method to use for denovo analysis (required for --split_denovo) (default: spectral)'
    )

    args = parser.parse_args()

    # Validate argument combinations
    if args.split_by_signature:
        if not args.mutations_file:
            parser.error("--mutations_file is required when using --split-by-signature")
    elif args.split_denovo:
        if args.mutation_purity is not None:
            parser.error("--mutation_purity can only be used with --split-by-signature")
        if args.mutations_file:
            parser.error("--mutations_file can only be used with --split-by-signature")

    return args


def setup_logging(output_directory):
    """Configure logging to both file and console with enhanced formatting"""
    log_file = Path(output_directory) / 'pipeline_execution.log'
    
    # Remove existing log file if it exists
    if log_file.exists():
        log_file.unlink()
        
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Log initial message
    logging.info(f"Pipeline started - Log file created at: {log_file}")
   

def check_input_files(args):
    """Validate that all required input files and directories exist"""
    missing_files = []
    
    # Check common required files
    if not Path(args.barcodes_file).is_file():
        missing_files.append(f"Barcodes file: {args.barcodes_file}")
    if not Path(args.reference_fasta).is_file():
        missing_files.append(f"Reference fasta: {args.reference_fasta}")
    if not Path(args.bam_directory).is_dir():
        missing_files.append(f"BAM directory: {args.bam_directory}")
    
    # Check mode-specific files
    if args.split_by_signature:
        if not Path(args.mutations_file).is_file():
            missing_files.append(f"Mutations file: {args.mutations_file}")
        
    if missing_files:
        error_msg = "The following required files/directories are missing:\n" + "\n".join(missing_files)
        logging.error(error_msg)
        sys.exit(1)


def run_denovo_clustering(input_dir: str, output_dir: str, fasta_file: str, method: str = 'spectral') -> None:
    """
    Run the de novo clustering analysis
    
    Args:
        input_dir: Input directory path
        output_dir: Output directory path
        fasta_file: Path to input FASTA file
        method: Clustering method to use (default: 'spectral')
                Options: 'spectral', 'hierphylo', 'quasiseq', 'compare_methods'
    """
    # denovo_output_dir = os.path.join(output_dir, "03.Split_denovo")
    # os.makedirs(denovo_output_dir, exist_ok=True)
    
    # Create denovo clustering pipeline instance
    pipeline = ClusteringPipeline(
        fasta_file=fasta_file,
        output_dir=output_dir,
        method=method
    )
    
    # Run the pipeline
    pipeline.run()

def process_sample(sample_name, args, current_sample, total_samples):
    """Process a single sample through the pipeline"""
    logging.info(f"Processing sample {current_sample}/{total_samples}: {sample_name}")
    
    # Create main sample output directory
    sample_output_directory = Path(args.output_directory) / sample_name
    sample_output_directory.mkdir(parents=True, exist_ok=True)
    logging.info(f"Created sample output directory: {sample_output_directory}")

    try:
        # Check if BAM file exists
        bam_file = Path(args.bam_directory) / f"{sample_name}.bam"
        if not bam_file.is_file():
            logging.error(f"BAM file not found for sample {sample_name}: {bam_file}")
            return False

        # 1. BAM Games stage
        logging.info(f"[Stage 1/3] Starting BAM processing for {sample_name}")
        bam_output_directory = sample_output_directory / "01.BAM_games"
        bam_output_directory.mkdir(parents=True, exist_ok=True)

        # Get and log the full absolute path of the BAM file being used
        bam_absolute_path = bam_file.resolve()
        logging.info(f"Using BAM file: {bam_absolute_path}")

        # Create a directory for temporary BAM storage
        temp_bam_dir = bam_output_directory / "temp_bam"
        temp_bam_dir.mkdir(parents=True, exist_ok=True)

        # Copy BAM file to local directory
        local_bam_path = temp_bam_dir / f"{sample_name}.bam"
        logging.info(f"Copying BAM file to local directory: {local_bam_path}")
        
        try:
            shutil.copy2(bam_absolute_path, local_bam_path)
            logging.info("BAM file copied successfully")
        
            # Process BAM data 
            bam_process_args = {
                'output_directory': bam_output_directory,
                'BAM_directory': temp_bam_dir,
                'reference_type': "full",
                'mapping_quality_threshold': None,
                'min_seq_length': None,
                'max_seq_length': None,
                'save_plots': True
            }
            cmd_str = "process_bam_data(\n" + \
                      "\n".join(f"{key}={value}," for key, value in bam_process_args.items()) + \
                      "\n)"
            logging.info(f"Executing BAM processing command:\n{cmd_str}")
            
            process_bam_data.process_bam_data(**bam_process_args)
            logging.info("BAM processing completed successfully")
        
        finally:
            # Clean up: remove the temporary BAM file and directory
            if local_bam_path.exists():
                logging.info(f"Cleaning up: removing temporary BAM file {local_bam_path}")
                local_bam_path.unlink()
            
            if temp_bam_dir.exists():
                logging.info(f"Cleaning up: removing temporary directory {temp_bam_dir}")
                temp_bam_dir.rmdir()  # This will only work if directory is empty


        # 2. Alignment stage
        logging.info(f"[Stage 2/3] Starting alignment for {sample_name}")
        alignment_output_directory = sample_output_directory / "02.Alignment"
        alignment_output_directory.mkdir(parents=True, exist_ok=True)
        
        # Extract reference ID from reference fasta
        try:
            with open(args.reference_fasta, 'r') as f:
                first_line = f.readline().strip()
                if first_line.startswith('>'):
                    reference_id = first_line[1:].split()[0]  # Take first part of header after '>'
                    logging.info(f"Extracted reference ID: {reference_id}")
                else:
                    logging.error("Reference fasta file does not start with '>'")
                    return False
        except Exception as e:
            logging.error(f"Error reading reference fasta file: {str(e)}")
            return False
        
        # Take the name of the reference to align from the actual input reference fasta
        input_fasta_specific = Path(bam_output_directory) / "fastq" / "by_Reference_Label" / f"{reference_id}_full.fasta"
        if not input_fasta_specific.is_file():
            logging.warning(f"Input fasta file not found: {input_fasta_specific}")
            
            # Look for any file ending with '_full.fasta' in the directory
            fasta_files = list(input_fasta_specific.parent.glob("*_full.fasta"))
            
            if not fasta_files:
                logging.error(f"No suitable fasta file found in {input_fasta_specific.parent}")
                return False
            
            # Select the first file found
            input_fasta_specific = fasta_files[0]
            logging.info(f"Using alternative fasta file: {input_fasta_specific}")
            
        
        alignment_args = {
            'input_fasta': input_fasta_specific,
            'reference_fasta': args.reference_fasta,
            'output_dir': alignment_output_directory
        }

        cmd_str = "run_alignment(\n" + \
                  "\n".join(f"{key}={value}," for key, value in alignment_args.items()) + \
                  "\n)"
        logging.info(f"Executing alignment command:\n{cmd_str}")
        
        run_alignment(**alignment_args)
        logging.info("Alignment completed successfully")
        
        # 3. Analysis stage (either mutation search or denovo clustering)
        logging.info(f"[Stage 3/3] Starting analysis for {sample_name}")
        
        all_aligned_fasta = alignment_output_directory / "all_aligned.fasta"
        if not all_aligned_fasta.is_file():
            logging.error(f"Aligned fasta file not found: {all_aligned_fasta}")
            return False
        
        if args.split_by_signature:
            # Signature-based analysis
            mutations_output_directory = sample_output_directory / "03.Split_by_signature"
            mutations_output_directory.mkdir(parents=True, exist_ok=True)
            
     
            mutation_args = {
                'mutations_file': args.mutations_file,
                'input_fasta': all_aligned_fasta,
                'output_dir': mutations_output_directory
            }
            
            if args.mutation_purity is not None:
                mutation_args['mutation_purity'] = args.mutation_purity
            else:
                logging.info("mutation_purity not specified, using default value (1)")
                
            cmd_str = "run_mutation_selector(\n" + \
                      "\n".join(f"{key}={value}," for key, value in mutation_args.items()) + \
                      "\n)"
            logging.info(f"Executing mutation search command:\n{cmd_str}")
            
            run_mutation_selector(**mutation_args)
            logging.info("Mutation search completed successfully")
        
        elif args.split_denovo:
            # De novo clustering analysis
            denovo_output_directory = sample_output_directory / "03.Split_denovo"
            denovo_output_directory.mkdir(parents=True, exist_ok=True)
    
            logging.info(f"Running denovo clustering with method: {args.method}")
            run_denovo_clustering(
                input_dir=str(alignment_output_directory),
                output_dir=str(denovo_output_directory),
                fasta_file=str(all_aligned_fasta),
                method=args.method
            )
            logging.info("De novo clustering completed successfully")
            
        else:
            logging.error("No analysis type specified. Must use either --split-by-signature or --split-denovo")
            return False

        logging.info(f"Successfully completed all processing stages for sample: {sample_name}")
        return True

    except Exception as e:
        logging.error(f"Error processing sample {sample_name}: {str(e)}")
        logging.error("Stack trace:", exc_info=True)
        return False


def format_command(script_name, args):
    """Format command-line arguments into a readable command string"""
    cmd_parts = [script_name]
    for arg, value in vars(args).items():
        cmd_parts.append(f'--{arg} "{value}"')
    return ' '.join(cmd_parts)

def main():
    """Main function to process all samples"""
    # Parse command line arguments
    args = parse_arguments()

    # Create output directory if it doesn't exist
    Path(args.output_directory).mkdir(parents=True, exist_ok=True)

    # Setup logging with output directory
    setup_logging(args.output_directory)
    
    # Log the full command that was used to execute the script
    script_name = os.path.basename(sys.argv[0])
    full_command = format_command(script_name, args)
    logging.info(f"Executed command:\n{full_command}")
    
    # Log script start with all arguments
    logging.info("Pipeline execution started with parameters:")
    for arg, value in vars(args).items():
        logging.info(f"  {arg}: {value}")

    # Check input files
    try:
        check_input_files(args)
    except Exception as e:
        logging.error(f"Input file validation failed: {str(e)}")
        sys.exit(1)

    try:
        # Read barcodes file
        df = pd.read_csv(args.barcodes_file)
        if 'sample' not in df.columns:
            logging.error("Barcodes file must contain a 'sample' column")
            sys.exit(1)

        # Process each sample
        total_samples = len(df)
        successful_samples = 0
        failed_samples = []
        
        logging.info(f"Starting processing of {total_samples} samples")
        
        for idx, row in df.iterrows():
            sample_name = str(row['sample'])
            current_sample = idx + 1
            
            if process_sample(sample_name, args, current_sample, total_samples):
                successful_samples += 1
            else:
                failed_samples.append(sample_name)

        # Log final statistics
        logging.info("\nPipeline Execution Summary:")
        logging.info(f"Total samples processed: {total_samples}")
        logging.info(f"Successfully completed: {successful_samples}")
        logging.info(f"Failed: {len(failed_samples)}")
        
        if failed_samples:
            logging.info("Failed samples:")
            for sample in failed_samples:
                logging.info(f"  - {sample}")

    except Exception as e:
        logging.error(f"Critical error in main execution: {str(e)}")
        logging.error("Stack trace:", exc_info=True)
        sys.exit(1)

    finally:
        logging.info("Pipeline execution completed")


if __name__ == "__main__":
    main()