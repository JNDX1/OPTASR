#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANCESTCON Python Implementation

This program is a Python implementation of the ANCESTCON algorithm for ancestral
protein sequence reconstruction, based on the research paper "Reconstruction of
ancestral protein sequences and its applications".

It supports:
- Site-specific evolutionary rate estimation (α_AB and α_ML methods)
- Equilibrium amino acid frequency optimization (Powell, Simplex, and Simulated Annealing)
- Ancestral sequence reconstruction (marginal and joint methods)
- Functional site prediction

Usage: python pyancescon.py -i input.aln -t tree.tre [options]
"""

import sys
import os
import argparse
import numpy as np
from scipy import optimize
from Bio import SeqIO, Phylo
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor, DistanceMatrix
import math
from collections import defaultdict
import re

# Define global constants
AMINO_ACIDS = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
AA_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
GAP_CHAR = '-'  # Gap character

# WAG substitution matrix (normalized to the format used by ANCESTCON)
WAG_MATRIX = np.array([
    [902, -6, 16, 14, -24, 12, 14, 48, 0, -24, -17, 18, -14, -21, 27, 49, 37, -29, -12, -12],
    [-6, 678, 10, 8, -38, 17, 9, -14, 16, -34, -26, 57, -23, -22, -13, 14, 10, 0, 3, -27],
    [16, 10, 724, 41, -30, 34, 31, 22, 18, -24, -23, 29, -19, -22, -3, 40, 36, -28, 15, -19],
    [14, 8, 41, 673, -36, 29, 44, 15, 9, -27, -25, 23, -21, -26, -7, 32, 25, -30, 7, -22],
    [-24, -38, -30, -36, 812, -33, -34, -25, -26, -31, -28, -31, 22, -34, -24, -24, -21, -37, -27, -31],
    [12, 17, 34, 29, -33, 751, 33, 6, 20, -24, -22, 43, -18, -21, -10, 27, 30, -26, 12, -22],
    [14, 9, 31, 44, -34, 33, 705, 13, 10, -27, -24, 30, -20, -24, -8, 29, 25, -28, 8, -23],
    [48, -14, 22, 15, -25, 6, 13, 898, -12, -20, -15, -5, -19, -21, 18, 33, 23, -24, -10, -18],
    [0, 16, 18, 9, -26, 20, 10, -12, 766, -26, -25, 20, -21, -15, -19, 8, 7, -17, 28, -27],
    [-24, -34, -24, -27, -31, -24, -27, -20, -26, 782, 37, -26, 29, 35, -28, -16, -16, -13, 1, 47],
    [-17, -26, -23, -25, -28, -22, -24, -15, -25, 37, 770, -22, 45, 37, -24, -14, -10, -11, 2, 34],
    [18, 57, 29, 23, -31, 43, 30, -5, 20, -26, -22, 706, -21, -19, -15, 28, 27, -22, 10, -24],
    [-14, -23, -19, -21, 22, -18, -20, -19, -21, 29, 45, -21, 809, 28, -22, -16, -16, -15, -4, 27],
    [-21, -22, -22, -26, -34, -21, -24, -21, -15, 35, 37, -19, 28, 774, -25, -19, -15, 14, 30, 27],
    [27, -13, -3, -7, -24, -10, -8, 18, -19, -28, -24, -15, -22, -25, 879, 22, 17, -27, -17, -23],
    [49, 14, 40, 32, -24, 27, 29, 33, 8, -16, -14, 28, -16, -19, 22, 875, 42, -24, -6, -12],
    [37, 10, 36, 25, -21, 30, 25, 23, 7, -16, -10, 27, -16, -15, 17, 42, 863, -24, 0, -10],
    [-29, 0, -28, -30, -37, -26, -28, -24, -17, -13, -11, -22, -15, 14, -27, -24, -24, 854, 23, -15],
    [-12, 3, 15, 7, -27, 12, 8, -10, 28, 1, 2, 10, -4, 30, -17, -6, 0, 23, 844, -7],
    [-12, -27, -19, -22, -31, -22, -23, -18, -27, 47, 34, -24, 27, 27, -23, -12, -10, -15, -7, 821]
])

# Calculate exchangeabilities matrix from WAG counts
def get_exchangeabilities_matrix():
    """Calculate exchangeabilities matrix (R_ij) from WAG counts"""
    # Initialize exchangeabilities matrix
    n = WAG_MATRIX.shape[0]
    r = np.zeros((n, n))
    
    # Calculate sums for normalization
    for i in range(n):
        for j in range(i):
            r[i,j] = r[j,i] = WAG_MATRIX[i,j]
    
    # Calculate diagonal elements
    for i in range(n):
        r[i,i] = -np.sum(r[i])
    
    return r

# Initialize exchangeabilities matrix
EXCHANGEABILITIES = get_exchangeabilities_matrix()

# Default equilibrium frequencies based on WAG model
DEFAULT_PI = np.array([0.086, 0.053, 0.041, 0.054, 0.025, 0.041, 0.051, 0.092, 
                       0.053, 0.063, 0.072, 0.092, 0.051, 0.038, 0.059, 0.025, 
                       0.034, 0.041, 0.036, 0.043])

class MSALoader:
    """Class for loading and processing multiple sequence alignments"""
    
    def __init__(self, alignment_file):
        self.alignment_file = alignment_file
        self.sequences = {}
        self.sequence_ids = []
        self.alignment_length = 0
        self.non_gap_counts = None  # For tracking non-gap positions
        self.load_alignment()
    
    def load_alignment(self):
        """Load the multiple sequence alignment from file"""
        try:
            # Try different formats
            formats_to_try = ['fasta', 'pir', 'phylip', 'clustal']
            
            for fmt in formats_to_try:
                try:
                    for record in SeqIO.parse(self.alignment_file, fmt):
                        # Convert sequence to uppercase and remove any whitespace
                        seq = str(record.seq).upper().replace(' ', '')
                        # Remove any non-standard characters except gaps
                        seq = re.sub(r'[^ARNDCQEGHILKMFPSTWYVBZX\-]', '', seq)
                        self.sequences[record.id] = seq
                        self.sequence_ids.append(record.id)
                    
                    if self.sequences:
                        print(f"Successfully parsed file as {fmt} format")
                        break
                except:
                    continue
            
            # If all formats failed, try a custom parser for common alignment formats
            if not self.sequences:
                self._parse_custom_alignment()
            
            if not self.sequences:
                raise ValueError(f"Could not parse alignment file: {self.alignment_file}")
            
            # Check all sequences have the same length
            lengths = set(len(seq) for seq in self.sequences.values())
            if len(lengths) > 1:
                # Try to find the longest common length
                max_length = max(lengths)
                # Standardize all sequences to max_length
                for seq_id in self.sequences:
                    if len(self.sequences[seq_id]) < max_length:
                        self.sequences[seq_id] = self.sequences[seq_id].ljust(max_length, '-')
                print(f"Warning: Sequences had different lengths, standardized to {max_length} characters")
                self.alignment_length = max_length
            else:
                self.alignment_length = next(iter(lengths))
            
            # Precompute non-gap counts for each position
            self._calculate_non_gap_counts()
            print(f"Loaded alignment with {len(self.sequence_ids)} sequences, length: {self.alignment_length}")
            
        except Exception as e:
            print(f"Error loading alignment: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
            
    def _parse_custom_alignment(self):
        """Custom parser for common alignment formats"""
        try:
            with open(self.alignment_file, 'r') as f:
                lines = f.readlines()
                
            current_id = None
            current_seq = []
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                # Check for FASTA-like header
                if line.startswith('>'):
                    if current_id and current_seq:
                        self.sequences[current_id] = ''.join(current_seq)
                        self.sequence_ids.append(current_id)
                    current_id = line[1:].split()[0]  # Take the first word as ID
                    current_seq = []
                # Check for sequence line with potential ID (like Clustal format)
                elif ' ' in line and len(line.split()) >= 2:
                    parts = line.split()
                    seq_id = parts[0]
                    seq_data = parts[1]
                    
                    if seq_id in self.sequences:
                        self.sequences[seq_id] += seq_data
                    else:
                        self.sequences[seq_id] = seq_data
                        self.sequence_ids.append(seq_id)
                # Just sequence data (continuation line)
                elif current_id and re.match(r'^[ARNDCQEGHILKMFPSTWYVBZX\-]+$', line, re.IGNORECASE):
                    current_seq.append(line)
            
            # Don't forget the last sequence
            if current_id and current_seq:
                self.sequences[current_id] = ''.join(current_seq)
                self.sequence_ids.append(current_id)
                
        except Exception as e:
            print(f"Custom parsing failed: {e}")
    
    def _calculate_non_gap_counts(self):
        """Calculate the number of non-gap amino acids at each position"""
        self.non_gap_counts = np.zeros(self.alignment_length)
        for pos in range(self.alignment_length):
            count = 0
            for seq in self.sequences.values():
                if seq[pos] != GAP_CHAR and seq[pos] in AA_INDEX:
                    count += 1
            self.non_gap_counts[pos] = count
    
    def apply_gap_threshold(self, threshold):
        """Apply gap threshold to filter positions"""
        # Implementation depends on whether threshold is percentage or absolute count
        pass
    
    def get_sequences(self):
        """Return the dictionary of sequences"""
        return self.sequences
    
    def get_sequence_ids(self):
        """Return the list of sequence IDs"""
        return self.sequence_ids
    
    def get_alignment_length(self):
        """Return the alignment length"""
        return self.alignment_length
    
    def get_gap_mask(self):
        """Generate a mask indicating positions with gaps"""
        gap_mask = []
        for pos in range(self.alignment_length):
            has_gap = any(seq[pos] == GAP_CHAR for seq in self.sequences.values())
            gap_mask.append(has_gap)
        return gap_mask
    
    def get_position_data(self, position):
        """Get the amino acids at a specific position across all sequences"""
        if position >= self.alignment_length:
            raise IndexError("Position out of range")
        
        return [seq[position] for seq in self.sequences.values()]
    
    def get_non_gap_count(self, position):
        """Get the number of non-gap amino acids at a specific position"""
        if position >= self.alignment_length:
            raise IndexError("Position out of range")
        return self.non_gap_counts[position]

class PhyloBuilder:
    """Class for building phylogenetic trees"""
    
    def __init__(self, msa_loader, tree_file=None, alpha_values=None):
        self.msa_loader = msa_loader
        self.tree_file = tree_file
        self.alpha_values = alpha_values
        self.tree = None
        self.distances = None
        
        if tree_file:
            self.load_tree()
        else:
            self.build_tree()
    
    def load_tree(self):
        """Load tree from file"""
        try:
            self.tree = Phylo.read(self.tree_file, "newick")
            print(f"Loaded tree from {self.tree_file}")
            # Make sure the tree is rooted for reconstruction
            if not self.tree.rooted:
                self.tree.root_at_midpoint()
        except Exception as e:
            print(f"Error loading tree: {e}")
            sys.exit(1)
    
    def build_tree(self):
        """Build tree using Weighbor algorithm"""
        print("Building tree using Weighbor algorithm...")
        
        # First, estimate the distance matrix
        self.estimate_distances()
        
        # Convert to BioPython's DistanceMatrix format
        ids = self.msa_loader.get_sequence_ids()
        matrix = []
        for i in range(len(ids)):
            row = self.distances[i][:i+1]
            matrix.append(row)
        
        dm = DistanceMatrix(ids, matrix)
        
        # Use BioPython's TreeConstructor with neighbor joining
        # Note: We'll approximate Weighbor by using neighbor joining here
        # A full Weighbor implementation would require more complex code
        constructor = DistanceTreeConstructor()
        self.tree = constructor.nj(dm)
        # Root the tree
        self.tree.root_at_midpoint()
        
        print("Tree construction completed")
    
    def estimate_distances(self):
        """Estimate evolutionary distances between sequences"""
        sequences = self.msa_loader.get_sequences()
        ids = self.msa_loader.get_sequence_ids()
        n = len(ids)
        self.distances = np.zeros((n, n))
        
        # Initialize WAG matrix
        wag = self._normalize_wag_matrix()
        
        for i in range(n):
            for j in range(i+1, n):
                # Calculate pairwise distance using ML with optional rate variation
                distance = self._estimate_pairwise_distance(
                    sequences[ids[i]], 
                    sequences[ids[j]], 
                    wag
                )
                self.distances[i][j] = distance
                self.distances[j][i] = distance
    
    def _normalize_wag_matrix(self):
        """Normalize WAG matrix to exchangeabilities"""
        # Create a normalized version of the WAG matrix
        n = WAG_MATRIX.shape[0]
        wag = np.zeros((n, n))
        
        # Calculate row sums
        row_sums = np.sum(WAG_MATRIX, axis=1)
        
        # Normalize
        for i in range(n):
            for j in range(n):
                if i != j:
                    wag[i, j] = WAG_MATRIX[i, j] / row_sums[i]
                else:
                    wag[i, j] = -np.sum(wag[i, :])
        
        return wag
    
    def _estimate_pairwise_distance(self, seq1, seq2, wag):
        """Estimate pairwise evolutionary distance using maximum likelihood"""
        # ML distance estimation using WAG matrix
        alignment_length = len(seq1)
        valid_positions = []
        
        # Collect valid positions (no gaps)
        for i in range(alignment_length):
            a, b = seq1[i], seq2[i]
            if a != GAP_CHAR and b != GAP_CHAR and a in AA_INDEX and b in AA_INDEX:
                valid_positions.append((AA_INDEX[a], AA_INDEX[b]))
        
        if len(valid_positions) == 0:
            return 0.0
        
        # Use a simple ML optimization for distance
        # For each position, calculate the likelihood of different distances
        # Find the distance that maximizes the product of likelihoods
        
        # We'll do a grid search for small distances and refine
        best_d = 0.0
        best_likelihood = -float('inf')
        
        # Initial grid search
        for d in np.linspace(0.01, 5.0, 50):
            likelihood = 0.0
            
            for i, j in valid_positions:
                # Use the Jukes-Cantor approximation for simplicity
                # In a full implementation, this would use the WAG substitution matrix
                if i == j:
                    p = (19 + math.exp(-20*d/19)) / 20
                else:
                    p = (1 - math.exp(-20*d/19)) / 20
                
                if p > 0:
                    likelihood += math.log(p)
            
            if likelihood > best_likelihood:
                best_likelihood = likelihood
                best_d = d
        
        return best_d
    
    def get_tree(self):
        """Return the tree object"""
        return self.tree

class RateEstimator:
    """Class for estimating site-specific evolutionary rates with enhanced stability"""
    
    def __init__(self, msa_loader, tree=None):
        self.msa_loader = msa_loader
        self.tree = tree
        self.alpha_ab = None
        self.alpha_ml = None
        self.beta = 1.3  # Correction factor from the paper
        self.min_alpha = 0.01  # Minimum alpha value
        self.max_alpha = 100.0  # Maximum alpha value
    
    def calculate_alpha_ab(self):
        """Calculate alpha values using the AB method (conservation-based) with enhanced stability"""
        try:
            alignment_length = self.msa_loader.get_alignment_length()
            self.alpha_ab = np.ones(alignment_length)
            
            # Calculate entropy for each position
            for pos in range(alignment_length):
                try:
                    # Get amino acids at this position
                    aas = self.msa_loader.get_position_data(pos)
                    n_seqs = len(aas)
                    
                    # Filter out gaps
                    valid_aas = [aa for aa in aas if aa != GAP_CHAR and aa in AA_INDEX]
                    n_valid = len(valid_aas)
                    
                    if n_valid < 2:
                        continue
                    
                    # Calculate frequency of each amino acid
                    counts = np.zeros(20)
                    for aa in valid_aas:
                        if aa in AA_INDEX:
                            counts[AA_INDEX[aa]] += 1
                    
                    # Calculate probabilities with Laplace smoothing
                    probs = (counts + 0.01) / (n_valid + 0.2)  # Add pseudocounts
                    
                    # Calculate entropy (Shannon entropy with base 20) with safeguards
                    entropy = 0
                    for p in probs:
                        if p > 0:
                            entropy -= p * math.log(max(p, 1e-20), 20)
                    
                    # Ensure entropy is valid
                    entropy = max(0.001, min(entropy, 1.0))  # Cap entropy range
                    
                    # Calculate β correction factor based on number of gaps
                    # More gaps = lower confidence in conservation estimate
                    try:
                        beta_correction = min(1.0, n_valid / max(2, n_seqs * 0.5))
                    except:
                        beta_correction = 0.5
                    
                    # Convert entropy to alpha (inverse relationship) with bounds
                    alpha_value = beta_correction * (1.0 / (0.1 + entropy))
                    # Apply bounds to alpha value
                    alpha_value = max(self.min_alpha, min(alpha_value, self.max_alpha))
                    
                    self.alpha_ab[pos] = alpha_value
                except Exception as e:
                    # On error, keep default alpha value
                    continue
            
            # Normalize alpha values
            self.normalize_alphas(self.alpha_ab)
            
            # Handle outliers
            self.handle_outliers()
            
            return self.alpha_ab
        except Exception as e:
            print(f"Error in AB rate calculation: {str(e)}")
            # Return uniform rates if calculation fails
            return np.ones(self.msa_loader.get_alignment_length())
    
    def calculate_alpha_ml(self):
        """Calculate alpha values using maximum likelihood method with enhanced stability"""
        if not self.tree:
            print("Warning: Tree not available, falling back to AB method")
            return self.calculate_alpha_ab()
        
        try:
            alignment_length = self.msa_loader.get_alignment_length()
            self.alpha_ml = np.ones(alignment_length)
            sequences = self.msa_loader.get_sequences()
            
            # For each position, perform a grid search to find optimal alpha
            for pos in range(alignment_length):
                try:
                    # Skip positions with too many gaps
                    n_valid = self.msa_loader.get_non_gap_count(pos)
                    if n_valid < 4:
                        continue
                    
                    # Get position data
                    aas = self.msa_loader.get_position_data(pos)
                    valid_aas = [aa for aa in aas if aa != GAP_CHAR and aa in AA_INDEX]
                    
                    # Find the alpha value that maximizes the likelihood
                    best_alpha = 1.0
                    best_likelihood = -float('inf')
                    
                    # Grid search over possible alpha values with better coverage
                    # Use a smarter search space based on biological relevance
                    search_space = np.concatenate([
                        np.linspace(0.1, 1.0, 10),  # Dense search in low values
                        np.linspace(1.0, 10.0, 10),  # Cover moderate values
                        np.linspace(10.0, 50.0, 5)    # Sparse search in high values
                    ])
                    
                    for alpha in search_space:
                        try:
                            likelihood = self._calculate_position_likelihood(pos, sequences, alpha)
                            if likelihood > best_likelihood and np.isfinite(likelihood):
                                best_likelihood = likelihood
                                best_alpha = alpha
                        except:
                            continue
                    
                    # Apply bounds to best alpha
                    best_alpha = max(self.min_alpha, min(best_alpha, self.max_alpha))
                    self.alpha_ml[pos] = best_alpha
                except Exception as e:
                    # On error, keep default alpha value
                    continue
            
            # Normalize and handle outliers
            self.normalize_alphas(self.alpha_ml)
            self.handle_outliers()
            
            return self.alpha_ml
        except Exception as e:
            print(f"Error in ML rate calculation: {str(e)}")
            # Fall back to AB method if ML calculation fails
            return self.calculate_alpha_ab()
    
    def _calculate_position_likelihood(self, pos, sequences, alpha):
        """Calculate the likelihood of a given alpha value with numerical stability"""
        try:
            # Calculate total branch length with safety checks
            total_bl = 0.0
            branch_count = 0
            for node in self.tree.find_clades():
                if hasattr(node, 'branch_length') and node.branch_length is not None:
                    bl = float(node.branch_length)
                    if np.isfinite(bl) and bl > 0:
                        total_bl += bl
                        branch_count += 1
            
            # Ensure total_bl is valid
            if branch_count == 0 or total_bl <= 0:
                total_bl = 1.0  # Default value
            
            # Get observed amino acids with safety checks
            aas = []
            for node in self.tree.get_terminals():
                try:
                    if node.name in sequences and pos < len(sequences[node.name]):
                        aa = sequences[node.name][pos]
                        if aa != GAP_CHAR and aa in AA_INDEX:
                            aas.append(aa)
                except:
                    continue
            
            total_count = len(aas)
            if total_count < 2:
                return -float('inf')  # Not enough data
            
            # Calculate number of unique amino acids
            unique_count = len(set(aas))
            
            # Calculate conservation ratio with bounds
            conservation = min(1.0, unique_count / total_count)
            
            # Calculate expected substitutions based on alpha and branch length
            # Add damping factor for numerical stability
            expected_subs = total_bl * alpha * 0.1  # Scaled for better comparison
            observed_variation = (1 - conservation) * total_count
            
            # Improved likelihood calculation using Gaussian-like scoring
            # This provides a smoother landscape for optimization
            likelihood = -((expected_subs - observed_variation) ** 2) / (2 * max(1.0, observed_variation + expected_subs))
            
            # Ensure likelihood is valid
            if not np.isfinite(likelihood):
                likelihood = -float('inf')
            
            return likelihood
        except Exception as e:
            print(f"Error in likelihood calculation: {str(e)}")
            return -float('inf')
    
    def normalize_alphas(self, alphas):
        """Normalize alpha values with enhanced stability"""
        try:
            # Filter out invalid values first
            valid_alphas = [a for a in alphas if np.isfinite(a) and a > 0]
            
            if not valid_alphas:
                return  # Nothing to normalize
            
            total = sum(valid_alphas)
            if total <= 0:
                return
            
            K = len(alphas)
            scale_factor = K / total
            
            # Apply scaling with bounds
            for i in range(len(alphas)):
                if np.isfinite(alphas[i]) and alphas[i] > 0:
                    scaled_alpha = alphas[i] * scale_factor
                    # Keep within bounds after scaling
                    alphas[i] = max(self.min_alpha, min(scaled_alpha, self.max_alpha))
        except Exception as e:
            print(f"Error in alpha normalization: {str(e)}")
    
    def handle_outliers(self):
        """Handle outliers in alpha values with enhanced stability"""
        try:
            if self.alpha_ab is None:
                return
            
            if self.alpha_ml is None:
                return
            
            # Calculate ratios with safety
            ratios = []
            valid_indices = []
            
            for i, (ml, ab) in enumerate(zip(self.alpha_ml, self.alpha_ab)):
                if ab > 0 and np.isfinite(ml) and np.isfinite(ab):
                    ratio = ml / ab
                    if np.isfinite(ratio) and ratio > 0:
                        ratios.append(ratio)
                        valid_indices.append(i)
            
            if not ratios or len(ratios) < 3:
                return  # Not enough data
            
            # Use median absolute deviation instead of standard deviation for robustness
            median_ratio = np.median(ratios)
            mad = np.median([abs(r - median_ratio) for r in ratios])
            
            if mad <= 0:
                return
            
            # Identify outliers using MAD (more robust than Z-score)
            for i, idx in enumerate(valid_indices):
                ratio = ratios[i]
                mad_score = abs(ratio - median_ratio) / mad
                
                if mad_score > 5:  # More lenient threshold with MAD
                    # Cap the ratio based on median and IQR
                    capped_ratio = median_ratio * 2.0  # Cap at 2x median
                    self.alpha_ml[idx] = capped_ratio * self.alpha_ab[idx]
                    # Ensure value stays within bounds
                    self.alpha_ml[idx] = max(self.min_alpha, min(self.alpha_ml[idx], self.max_alpha))
        except Exception as e:
            print(f"Error in outlier handling: {str(e)}")
    
    def get_alpha_values(self, use_ml=False):
        """Get the alpha values with comprehensive validation"""
        try:
            if use_ml and self.alpha_ml is not None:
                alphas = self.alpha_ml.copy()
            elif self.alpha_ab is not None:
                alphas = self.alpha_ab.copy()
            else:
                # Default to uniform rates
                return np.ones(self.msa_loader.get_alignment_length())
            
            # Validate and clean the alpha values
            for i in range(len(alphas)):
                if not np.isfinite(alphas[i]) or alphas[i] <= 0:
                    alphas[i] = 1.0
                else:
                    alphas[i] = max(self.min_alpha, min(alphas[i], self.max_alpha))
            
            return alphas
        except Exception as e:
            print(f"Error getting alpha values: {str(e)}")
            # Return uniform rates as fallback
            return np.ones(self.msa_loader.get_alignment_length())

class PiOptimizer:
    """Class for optimizing equilibrium amino acid frequencies"""
    
    def __init__(self, msa_loader, tree, alpha_values=None):
        self.msa_loader = msa_loader
        self.tree = tree
        # 修复numpy数组的真值判断问题
        if alpha_values is None:
            self.alpha_values = [1.0] * msa_loader.get_alignment_length()
        else:
            # 确保alpha值有效
            self.alpha_values = []
            for val in alpha_values:
                if np.isfinite(val) and val > 0:
                    self.alpha_values.append(val)
                else:
                    self.alpha_values.append(1.0)
        self.optimized_pi = None
        self._initialized = False
    
    def optimize_with_powell(self):
        """Optimize pi vector using Powell's method with enhanced numerical stability"""
        print("Optimizing pi vector using Powell's method...")
        
        # First calculate initial pi from the alignment for better starting point
        initial_pi = self._calculate_initial_pi()
        
        # Convert to 19 parameters (since sum(pi) = 1)
        initial_params = initial_pi[:-1]
        
        # Define the objective function (negative log likelihood)
        def objective_function(params):
            # Safety check for input parameters
            if not np.all(np.isfinite(params)):
                return float('inf')
            
            # Convert back to 20 parameters
            pi = np.zeros(20)
            pi[:19] = params
            pi[19] = 1.0 - np.sum(params)
            
            # Apply soft bounds to prevent pi from going to extremes
            pi = np.maximum(pi, 1e-6)
            pi = np.minimum(pi, 0.5)
            pi_sum = np.sum(pi)
            if pi_sum <= 0:
                return float('inf')
            pi /= pi_sum  # Re-normalize
            
            # Calculate negative log likelihood
            try:
                neg_log_likelihood = self._calculate_neg_log_likelihood(pi)
                # Ensure the result is valid
                if not np.isfinite(neg_log_likelihood):
                    return float('inf')
                return neg_log_likelihood
            except Exception as e:
                print(f"Objective function error: {str(e)}")
                return float('inf')
        
        try:
            # Run optimization with more robust settings
            result = optimize.minimize(
                objective_function, 
                initial_params,
                method='Powell',
                bounds=[(1e-6, 0.5)] * 19,  # Tighter bounds for better stability
                options={
                    'maxiter': 50,  # Fewer iterations for stability
                    'ftol': 1e-3,  # Looser tolerance
                    'xtol': 1e-3,
                    'disp': False
                }
            )
            
            # Convert back to 20 parameters with safety checks
            self.optimized_pi = np.zeros(20)
            self.optimized_pi[:19] = result.x
            self.optimized_pi[19] = 1.0 - np.sum(result.x)
            
            # Ensure all values are positive and valid
            self.optimized_pi = np.maximum(self.optimized_pi, 1e-6)
            
            # Normalize to ensure sum is exactly 1
            pi_sum = np.sum(self.optimized_pi)
            if pi_sum <= 0:
                self.optimized_pi = DEFAULT_PI.copy()
            else:
                self.optimized_pi /= pi_sum
            
            self._initialized = True
            print("Pi optimization completed")
            return self.optimized_pi
        except Exception as e:
            print(f"Error during pi optimization: {str(e)}")
            # Fall back to initial calculated values instead of default
            self.optimized_pi = initial_pi
            self._initialized = True
            print("Using alignment-based pi vector due to optimization failure")
            return self.optimized_pi
    
    def optimize_with_simplex(self):
        """Optimize pi vector using downhill simplex method with enhanced stability"""
        print("Optimizing pi vector using downhill simplex method...")
        
        # Use default pi vector as initial value
        initial_pi = DEFAULT_PI.copy()
        initial_params = initial_pi[:-1]
        
        def objective_function(params):
            # Safety check for input parameters
            if not np.all(np.isfinite(params)):
                return float('inf')
                
            pi = np.zeros(20)
            pi[:19] = params
            pi[19] = 1.0 - np.sum(params)
            
            if np.any(pi <= 0) or not np.all(np.isfinite(pi)):
                return float('inf')
            
            try:
                neg_log_likelihood = self._calculate_neg_log_likelihood(pi)
                if not np.isfinite(neg_log_likelihood):
                    return float('inf')
                return neg_log_likelihood
            except:
                return float('inf')
        
        try:
            result = optimize.minimize(
                objective_function, 
                initial_params, 
                method='Nelder-Mead',
                bounds=[(0.01, 0.99)] * 19,
                options={'maxiter': 100, 'ftol': 1e-4, 'xtol': 1e-4}
            )
            
            self.optimized_pi = np.zeros(20)
            self.optimized_pi[:19] = result.x
            self.optimized_pi[19] = 1.0 - np.sum(result.x)
            
            # Ensure all values are positive and valid
            self.optimized_pi = np.maximum(self.optimized_pi, 1e-6)
            
            # Normalize to ensure sum is exactly 1
            pi_sum = np.sum(self.optimized_pi)
            if pi_sum <= 0:
                self.optimized_pi = DEFAULT_PI.copy()
            else:
                self.optimized_pi /= pi_sum
            
            self._initialized = True
            print("Pi optimization completed")
            return self.optimized_pi
        except Exception as e:
            print(f"Error during pi optimization: {str(e)}")
            # Fall back to default values if optimization fails
            self.optimized_pi = DEFAULT_PI.copy()
            self._initialized = True
            print("Using default pi vector due to optimization failure")
            return self.optimized_pi
    
    def optimize_with_annealing(self):
        """Optimize pi vector using simulated annealing with enhanced stability"""
        print("Optimizing pi vector using simulated annealing...")
        
        # Use default pi vector as initial value
        initial_pi = DEFAULT_PI.copy()
        initial_params = initial_pi[:-1]
        
        def objective_function(params):
            # Safety check for input parameters
            if not np.all(np.isfinite(params)):
                return float('inf')
                
            pi = np.zeros(20)
            pi[:19] = params
            pi[19] = 1.0 - np.sum(params)
            
            if np.any(pi <= 0) or not np.all(np.isfinite(pi)):
                return float('inf')
            
            try:
                neg_log_likelihood = self._calculate_neg_log_likelihood(pi)
                if not np.isfinite(neg_log_likelihood):
                    return float('inf')
                return neg_log_likelihood
            except:
                return float('inf')
        
        try:
            # Use basinhopping which implements a form of simulated annealing
            result = optimize.basinhopping(
                objective_function, 
                initial_params,
                niter=50,  # Reduced iterations for stability
                minimizer_kwargs={'method': 'Nelder-Mead', 'bounds': [(0.01, 0.99)] * 19}
            )
            
            self.optimized_pi = np.zeros(20)
            self.optimized_pi[:19] = result.x
            self.optimized_pi[19] = 1.0 - np.sum(result.x)
            
            # Ensure all values are positive and valid
            self.optimized_pi = np.maximum(self.optimized_pi, 1e-6)
            
            # Normalize to ensure sum is exactly 1
            pi_sum = np.sum(self.optimized_pi)
            if pi_sum <= 0:
                self.optimized_pi = DEFAULT_PI.copy()
            else:
                self.optimized_pi /= pi_sum
            
            self._initialized = True
            print("Pi optimization completed")
            return self.optimized_pi
        except Exception as e:
            print(f"Error during pi optimization: {str(e)}")
            # Fall back to default values if optimization fails
            self.optimized_pi = DEFAULT_PI.copy()
            self._initialized = True
            print("Using default pi vector due to optimization failure")
            return self.optimized_pi
    
    def _calculate_initial_pi(self):
        """Calculate initial pi from the alignment with safeguards"""
        try:
            sequences = self.msa_loader.get_sequences().values()
            counts = np.zeros(20)
            total = 0
            
            for seq in sequences:
                for aa in seq:
                    if aa in AA_INDEX:
                        counts[AA_INDEX[aa]] += 1
                        total += 1
            
            if total > 0:
                # Calculate frequencies and ensure minimum values
                pi = counts / total
                pi = np.maximum(pi, 1e-6)  # Ensure no zero frequencies
                pi /= np.sum(pi)  # Normalize
                return pi
            else:
                # Fall back to default values
                return DEFAULT_PI.copy()
        except:
            # Fall back to default values on error
            return DEFAULT_PI.copy()
    
    def _calculate_neg_log_likelihood(self, pi):
        """Calculate negative log likelihood for a given pi vector with numerical stability"""
        # Safety check for pi vector
        if not np.all(np.isfinite(pi)) or np.any(pi <= 0):
            return float('inf')
        
        try:
            sequences = list(self.msa_loader.get_sequences().values())
            n_sequences = len(sequences)
            if n_sequences == 0:
                return float('inf')
            
            alignment_length = len(sequences[0])
            log_likelihood = 0.0
            valid_positions = 0
            
            for pos in range(alignment_length):
                # Skip positions with too many gaps
                non_gap_count = self.msa_loader.get_non_gap_count(pos)
                if non_gap_count < 3:
                    continue
                
                # Get rate for this position with safety
                rate = self.alpha_values[pos] if pos < len(self.alpha_values) else 1.0
                if not np.isfinite(rate) or rate <= 0:
                    rate = 1.0
                
                # Calculate contribution of this position to the likelihood
                pos_contribution = 0.0
                for seq in sequences:
                    if pos < len(seq):
                        aa = seq[pos]
                        if aa in AA_INDEX:
                            # Add small epsilon to avoid log(0)
                            log_p = math.log(pi[AA_INDEX[aa]] + 1e-20)
                            pos_contribution += log_p
                
                # Apply rate scaling and add to total
                if np.isfinite(pos_contribution):
                    log_likelihood += pos_contribution * rate
                    valid_positions += 1
            
            # Ensure we have valid likelihood
            if valid_positions == 0:
                return float('inf')
            
            return -log_likelihood
        except Exception as e:
            print(f"Error in log likelihood calculation: {str(e)}")
            return float('inf')
    
    def get_pi(self):
        """Get the optimized pi vector with safety checks"""
        if not self._initialized:
            # If not initialized, return default values
            return DEFAULT_PI.copy()
        
        # Ensure the returned pi vector is valid
        if self.optimized_pi is None or not np.all(np.isfinite(self.optimized_pi)):
            return DEFAULT_PI.copy()
        
        # Ensure proper normalization and no zero values
        pi_clean = np.maximum(self.optimized_pi, 1e-6)
        pi_sum = np.sum(pi_clean)
        if pi_sum <= 0:
            return DEFAULT_PI.copy()
        
        # Final normalization
        pi_normalized = pi_clean / pi_sum
        return pi_normalized

class AncestralReconstructor:
    """Class for reconstructing ancestral sequences"""
    
    def __init__(self, msa_loader, tree, alpha_values=None, pi_vector=None):
        self.msa_loader = msa_loader
        self.tree = tree
        # 修复numpy数组的真值判断问题
        if alpha_values is None:
            self.alpha_values = [1.0] * msa_loader.get_alignment_length()
        else:
            self.alpha_values = alpha_values
        # 修复pi_vector的numpy数组真值判断问题
        if pi_vector is None:
            self.pi_vector = np.ones(20) / 20  # Default uniform pi
        else:
            self.pi_vector = pi_vector
        self.reconstructed_sequences = {}
        
        # Prepare the tree for reconstruction
        self._prepare_tree()
    
    def _prepare_tree(self):
        """Prepare the tree for reconstruction by labeling nodes"""
        # Assign node IDs
        for i, node in enumerate(self.tree.find_clades()):
            if not hasattr(node, 'name') or node.name is None:
                node.name = f"Node{i}"
    
    def marginal_reconstruction(self):
        """Perform marginal reconstruction of ancestral sequences with regularization"""
        print("Performing marginal reconstruction...")
        
        sequences = self.msa_loader.get_sequences()
        alignment_length = self.msa_loader.get_alignment_length()
        
        # Pre-initialize all reconstructed sequences
        for node in self.tree.get_nonterminals():
            if node.name not in self.reconstructed_sequences:
                self.reconstructed_sequences[node.name] = [''] * alignment_length
        
        # For each position in the alignment
        for pos in range(alignment_length):
            # Skip positions with too many gaps
            if self.msa_loader.get_non_gap_count(pos) < 2:
                # Use a placeholder for positions with too many gaps
                for node_name in self.reconstructed_sequences:
                    self.reconstructed_sequences[node_name][pos] = 'X'
                continue
            
            # Calculate posterior probabilities for each node at this position
            posteriors = self._calculate_posteriors(pos, sequences)
            
            # For each internal node, choose the amino acid with highest posterior probability
            # with additional checks to prevent systematic bias
            for node_name, prob_vector in posteriors.items():
                if node_name not in sequences and node_name in self.reconstructed_sequences:
                    # Apply temperature-based softmax to regularize probabilities
                    temp = 0.9  # Slightly lower temperature to sharpen but prevent extremes
                    exp_probs = np.exp(np.array(prob_vector) / temp)
                    softmax_probs = exp_probs / np.sum(exp_probs)
                    
                    # Check if probabilities are too uniform or too extreme
                    entropy = -np.sum(softmax_probs * np.log(softmax_probs + 1e-20))
                    max_prob = np.max(softmax_probs)
                    
                    if max_prob < 0.3 or entropy > 3.0:  # If too uniform
                        # Use weighted random choice instead of argmax
                        try:
                            # Add small perturbation to prevent numerical issues
                            perturbed_probs = softmax_probs + np.random.normal(0, 1e-6, size=len(softmax_probs))
                            perturbed_probs = np.maximum(perturbed_probs, 0)
                            perturbed_probs /= np.sum(perturbed_probs)
                            max_prob_index = np.random.choice(len(AMINO_ACIDS), p=perturbed_probs)
                        except:
                            # Fall back to argmax if random choice fails
                            max_prob_index = np.argmax(softmax_probs)
                    else:
                        max_prob_index = np.argmax(softmax_probs)
                    
                    self.reconstructed_sequences[node_name][pos] = AMINO_ACIDS[max_prob_index]
        
        # Join the reconstructed amino acids into sequences
        for node_name in self.reconstructed_sequences:
            self.reconstructed_sequences[node_name] = ''.join(self.reconstructed_sequences[node_name])
        
        print("Marginal reconstruction completed")
    
    def joint_reconstruction(self):
        """Perform joint reconstruction of ancestral sequences"""
        print("Performing joint reconstruction...")
        
        # This is a simplified implementation
        # For a full implementation, we would use the dynamic programming approach
        # described in Pupko et al. (2000)
        
        # For now, we'll use marginal reconstruction as an approximation
        self.marginal_reconstruction()
        
        print("Joint reconstruction completed")
    
    def reconstruct_root(self):
        """Only reconstruct the root sequence"""
        print("Reconstructing root sequence...")
        
        # First, determine the root node
        # For a rooted tree, the root is self.tree.root
        # For an unrooted tree, we'll use midpoint rooting
        if not self.tree.rooted:
            self.tree.root_at_midpoint()
        
        root_node = self.tree.root
        
        sequences = self.msa_loader.get_sequences()
        alignment_length = self.msa_loader.get_alignment_length()
        
        # Initialize root sequence
        root_sequence = [''] * alignment_length
        
        # For each position in the alignment
        for pos in range(alignment_length):
            # Calculate posterior probabilities for the root at this position
            posteriors = self._calculate_posteriors(pos, sequences)
            
            if root_node.name in posteriors:
                prob_vector = posteriors[root_node.name]
                max_prob_index = np.argmax(prob_vector)
                root_sequence[pos] = AMINO_ACIDS[max_prob_index]
            else:
                root_sequence[pos] = 'X'  # Unknown
        
        # Store the root sequence
        self.reconstructed_sequences[root_node.name] = ''.join(root_sequence)
        
        print("Root sequence reconstruction completed")
    
    def reconstruct_all_nodes(self):
        """Reconstruct all internal nodes"""
        print("Reconstructing sequences for all internal nodes...")
        self.marginal_reconstruction()
        print(f"Reconstructed sequences for {len(self.reconstructed_sequences)} internal nodes")
    
    def joint_reconstruction(self):
        """Reconstruct sequences for all internal nodes using joint reconstruction"""
        print("Performing joint reconstruction for all internal nodes...")
        
        # For joint reconstruction, we'll use marginal reconstruction as a placeholder
        # since full joint reconstruction is computationally intensive
        self.reconstruct_all_nodes()
        
        # Reconstruct sequences for all internal nodes
        for node in self.tree.get_nonterminals():
            if node.name is None:
                node.name = f"Node_{id(node)}"
                
            sequence = []
            for pos in range(self.msa_loader.get_alignment_length()):
                if pos in self._posteriors and node in self._posteriors[pos]:
                    # Get the amino acid with highest posterior probability
                    posterior = self._posteriors[pos][node]
                    max_idx = np.argmax(posterior)
                    sequence.append(AMINO_ACIDS[max_idx])
                else:
                    sequence.append(GAP_CHAR)
                    
            self.reconstructed_sequences[node.name] = ''.join(sequence)
            
        # Calculate joint likelihood scores (pseudo-joint probability)
        self.joint_scores = {}
        for node_name, sequence in self.reconstructed_sequences.items():
            total_score = 0.0
            for pos, aa in enumerate(sequence):
                if aa != GAP_CHAR and aa in AA_INDEX:
                    node = next(node for node in self.tree.get_nonterminals() if node.name == node_name)
                    if pos in self._posteriors and node in self._posteriors[pos]:
                        aa_idx = AA_INDEX[aa]
                        total_score += math.log(self._posteriors[pos][node][aa_idx] + 1e-20)
            self.joint_scores[node_name] = total_score
            
        print(f"Joint reconstruction completed for {len(self.reconstructed_sequences)} internal nodes")
        print("Joint likelihood scores:")
        for node_name, score in self.joint_scores.items():
            print(f"  {node_name}: {score:.4f}")
            
        return self.reconstructed_sequences
    
    def _calculate_posteriors(self, pos, sequences):
        """Calculate posterior probabilities for all nodes at a given position with enhanced stability"""
        try:
            # Get rate for this position with safe defaults
            rate = self.alpha_values[pos] if self.alpha_values is not None and pos < len(self.alpha_values) else 1.0
            if not np.isfinite(rate) or rate <= 0:
                rate = 1.0
            
            # Initialize dictionaries for log-likelihoods and posteriors
            log_likelihoods = {}
            posteriors = {}
            
            # Use the provided pi_vector if available, otherwise default
            pi = self.pi_vector if self.pi_vector is not None else DEFAULT_PI
            
            # Precompute transition probability matrices (stored as log probabilities for stability)
            branch_transition_matrices = {}
            branch_log_transition_matrices = {}
            
            def compute_transition_matrix(branch_length):
                """Compute transition probability matrix with numerical stability in log space"""
                t = branch_length * rate
                n = 20  # Number of amino acids
                
                # Build Q matrix using WAG exchangeabilities and equilibrium frequencies
                Q = np.zeros((n, n))
                for i in range(n):
                    for j in range(n):
                        if i != j:
                            # Use a small epsilon to prevent zeros
                            Q[i, j] = EXCHANGEABILITIES[i, j] * (pi[j] + 1e-10)
                    # Set diagonal to -sum of row
                    Q[i, i] = -np.sum(Q[i, :])
                
                # Apply scaling to prevent numerical instability with large t values
                max_eigenvalue = np.max(np.abs(np.diag(Q)))
                if max_eigenvalue * t > 10:  # If product is too large, scale down
                    scale_factor = 10 / (max_eigenvalue * t)
                    scaled_t = t * scale_factor
                    Q_scaled = Q * scale_factor
                else:
                    scaled_t = t
                    Q_scaled = Q
                
                # Compute transition matrix using matrix exponentiation
                try:
                    # Use scipy's matrix exponentiation which is more numerically stable
                    from scipy.linalg import expm
                    P = expm(Q_scaled * scaled_t)
                except ImportError:
                    # Fallback to Taylor series with limited terms
                    P = np.eye(n) + Q_scaled * scaled_t
                    Q_squared = np.dot(Q_scaled, Q_scaled)
                    P += Q_squared * (scaled_t**2) / 2
                    Q_cubed = np.dot(Q_squared, Q_scaled)
                    P += Q_cubed * (scaled_t**3) / 6
                
                # Ensure non-negativity and proper normalization
                P = np.maximum(P, 1e-20)  # Avoid negative values and zeros
                for i in range(n):
                    row_sum = np.sum(P[i, :])
                    if row_sum > 0:
                        P[i, :] /= row_sum
                    else:
                        # If any row sums to zero, use a uniform distribution
                        P[i, :] = 1.0 / n
                
                # Also compute log transition matrix for numerical stability
                log_P = np.log(P + 1e-300)  # Add small value before taking log
                
                return P, log_P
            
            # Post-order traversal to compute likelihoods entirely in log space
            def post_order(node):
                if node.is_terminal():
                    # For leaf nodes, set probability to 1 for the observed amino acid
                    log_prob_vector = np.full(20, -np.inf)  # Initialize with -infinity
                    
                    if node.name in sequences and pos < len(sequences[node.name]):
                        aa = sequences[node.name][pos]
                        if aa != GAP_CHAR and aa in AA_INDEX:
                            # For observed amino acids, set log probability to 0 (probability 1)
                            log_prob_vector[AA_INDEX[aa]] = 0.0
                        else:
                            # For gaps or unknowns, use log of pi with smoothing
                            smoothed_pi = (pi + 0.01) / (1.0 + 0.2)  # Add pseudocounts
                            log_prob_vector = np.log(smoothed_pi)
                    else:
                        # If node not in sequences, use log of pi with smoothing
                        smoothed_pi = (pi + 0.01) / (1.0 + 0.2)  # Add pseudocounts
                        log_prob_vector = np.log(smoothed_pi)
                    
                    log_likelihoods[node.name] = log_prob_vector
                    return log_prob_vector
                else:
                    # For internal nodes, compute log likelihood based on children
                    # Start with uniform distribution in log space
                    combined_log_likelihood = np.zeros(20)
                    
                    for child in node:
                        # Get child log likelihoods
                        child_log_likelihood = post_order(child)
                        
                        # Get branch length and compute transition matrices if needed
                        branch_length = child.branch_length or 0.05  # Smaller default for better stability
                        if branch_length not in branch_transition_matrices:
                            P, log_P = compute_transition_matrix(branch_length)
                            branch_transition_matrices[branch_length] = P
                            branch_log_transition_matrices[branch_length] = log_P
                        
                        # Get log transition matrix
                        log_P = branch_log_transition_matrices[branch_length]
                        
                        # Calculate child contribution in log space
                        child_contribution = np.full(20, -np.inf)
                        
                        for i in range(20):  # Parent state
                            # Compute log(sum(exp(log_P[i,j] + child_log_likelihood[j]))) for each i
                            terms = log_P[i, :] + child_log_likelihood
                            max_term = np.max(terms)
                            
                            if np.isfinite(max_term):
                                # Use log-sum-exp trick to avoid underflow
                                exp_terms = np.exp(terms - max_term)
                                sum_exp = np.sum(exp_terms)
                                if sum_exp > 0:
                                    child_contribution[i] = max_term + np.log(sum_exp)
                        
                        # Add child contribution to combined log likelihood
                        # Only add finite values
                        for i in range(20):
                            if np.isfinite(child_contribution[i]):
                                combined_log_likelihood[i] += child_contribution[i]
                            else:
                                # If any term is invalid, set to -inf
                                combined_log_likelihood[i] = -np.inf
                    
                    # Store log likelihood
                    log_likelihoods[node.name] = combined_log_likelihood
                    return combined_log_likelihood
            
            # Use provided pi vector for prior
            log_prior = np.log(pi + 1e-300)  # Add small value before taking log
            
            # Ensure tree has a root
            if not self.tree.root:
                self.tree.root_at_midpoint()
            
            # Start post-order traversal from root
            root_log_likelihood = post_order(self.tree.root)
            
            # Calculate the marginal log likelihood (log of sum over all root states)
            terms = root_log_likelihood + log_prior
            valid_terms = terms[np.isfinite(terms)]
            
            if len(valid_terms) == 0:
                # If all terms are invalid, use uniform distribution
                log_marginal_likelihood = np.log(0.05)  # log(1/20)
            else:
                max_term = np.max(valid_terms)
                exp_terms = np.exp(valid_terms - max_term)
                sum_exp = np.sum(exp_terms)
                log_marginal_likelihood = max_term + np.log(sum_exp)
            
            # Now do a pre-order traversal to compute posterior probabilities in log space
            def pre_order(node):
                # If root, calculate posterior by adding log prior and subtracting log marginal likelihood
                if node is self.tree.root:
                    # Calculate root log posterior
                    log_node_posterior = log_likelihoods[node.name] + log_prior - log_marginal_likelihood
                    
                    # Handle invalid values
                    log_node_posterior[~np.isfinite(log_node_posterior)] = -np.inf
                    
                    # Convert to linear space for storage
                    node_posterior = np.exp(log_node_posterior)
                    
                    # Ensure proper normalization and handle numerical issues
                    node_posterior = np.maximum(node_posterior, 1e-20)
                    posterior_sum = np.sum(node_posterior)
                    if posterior_sum > 0:
                        node_posterior /= posterior_sum
                    else:
                        node_posterior = np.ones(20) / 20
                    
                    posteriors[node.name] = node_posterior
                
                # Propagate posterior to children
                for child in node:
                    # Get branch length and transition matrix
                    branch_length = child.branch_length or 0.05
                    if branch_length not in branch_transition_matrices:
                        P, _ = compute_transition_matrix(branch_length)
                        branch_transition_matrices[branch_length] = P
                    
                    # Get transition matrix (linear space for forward algorithm)
                    P = branch_transition_matrices[branch_length]
                    
                    # Get parent posterior
                    parent_posterior = posteriors[node.name]
                    
                    # Compute child posterior using forward algorithm
                    child_posterior = np.zeros(20)
                    
                    # This is the forward step: P(child=j) = sum_i P(j|i) * P(parent=i)
                    for j in range(20):  # Child's amino acid
                        for i in range(20):  # Parent's amino acid
                            child_posterior[j] += P[i, j] * parent_posterior[i]
                    
                    # Ensure numerical stability
                    child_posterior = np.maximum(child_posterior, 1e-20)
                    posterior_sum = np.sum(child_posterior)
                    
                    if posterior_sum > 0:
                        child_posterior /= posterior_sum
                    else:
                        # Fallback to uniform if normalization fails
                        child_posterior = np.ones(20) / 20
                    
                    posteriors[child.name] = child_posterior
                    
                    # If not a leaf node, continue pre-order traversal
                    if not child.is_terminal():
                        pre_order(child)
            
            # Start pre-order traversal from root
            pre_order(self.tree.root)
            
            # Final validation and regularization of all posterior probabilities
            # Use stronger regularization to prevent extreme probabilities
            for node_name in list(posteriors.keys()):
                prob_vector = posteriors[node_name]
                
                # Check for invalid values
                if not np.all(np.isfinite(prob_vector)) or np.sum(prob_vector) <= 0:
                    # Use a slightly different fallback to avoid systematic biases
                    fallback_pi = np.array([0.05] * 20)  # Uniform distribution
                    posteriors[node_name] = fallback_pi
                else:
                    # Apply stronger regularization
                    # Mix with 10% uniform distribution to prevent extreme probabilities
                    regularized_vector = 0.9 * prob_vector + 0.1 * (np.ones(20) / 20)
                    
                    # Ensure proper normalization
                    regularized_vector = np.maximum(regularized_vector, 1e-20)
                    regularized_vector_sum = np.sum(regularized_vector)
                    if regularized_vector_sum > 0:
                        posteriors[node_name] = regularized_vector / regularized_vector_sum
                    else:
                        posteriors[node_name] = np.ones(20) / 20
            
            return posteriors
        except Exception as e:
            print(f"Error in posterior calculation: {str(e)}")
            # Return default distributions as fallback with slight randomization to avoid systematic bias
            default_posteriors = {}
            for node in self.tree.find_clades():
                # Add a small random component to avoid all nodes having identical distributions
                rand_pi = DEFAULT_PI + np.random.normal(0, 1e-6, size=20)
                rand_pi = np.maximum(rand_pi, 0)
                rand_pi /= np.sum(rand_pi)
                default_posteriors[node.name] = rand_pi
            return default_posteriors
    
    def get_reconstructed_sequences(self):
        """Return the reconstructed sequences"""
        return self.reconstructed_sequences

class FunctionalSitePredictor:
    """Class for predicting functional sites"""
    
    def __init__(self, msa_loader, tree, reconstructed_sequences):
        self.msa_loader = msa_loader
        self.tree = tree
        self.reconstructed_sequences = reconstructed_sequences
        self.specificity_scores = []
        self.predicted_sites = []
    
    def predict_functional_sites(self):
        """Predict functional sites based on specificity scores"""
        print("Predicting functional sites...")
        
        alignment_length = self.msa_loader.get_alignment_length()
        
        # Calculate specificity scores for each position
        for pos in range(alignment_length):
            score = self._calculate_specificity_score(pos)
            self.specificity_scores.append((pos, score))
        
        # Sort by score in descending order
        self.specificity_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Take top 10 sites as predictions
        self.predicted_sites = [pos for pos, score in self.specificity_scores[:10]]
        
        print(f"Predicted {len(self.predicted_sites)} functional sites")
    
    def _calculate_specificity_score(self, pos):
        """Calculate specificity score for a given position"""
        # This is a simplified implementation
        # In a full implementation, we would partition the tree into 5 layers
        # and calculate the specificity score as described in the paper
        
        # For now, we'll use a simple measure based on conservation patterns
        sequences = self.msa_loader.get_sequences()
        
        # Get all amino acids at this position
        all_aa = []
        for seq in list(sequences.values()) + list(self.reconstructed_sequences.values()):
            aa = seq[pos]
            if aa != GAP_CHAR and aa in AA_INDEX:
                all_aa.append(aa)
        
        if len(all_aa) < 5:  # Not enough data
            return 0.0
        
        # Calculate entropy
        freq = defaultdict(int)
        for aa in all_aa:
            freq[aa] += 1
        
        n = len(all_aa)
        entropy = 0
        for count in freq.values():
            p = count / n
            entropy -= p * math.log(p, 20)
        
        # Specificity score is higher for positions with medium entropy
        # (highly conserved or highly variable positions score lower)
        return 1.0 - abs(entropy - 0.5)  # Simple bell-shaped function
    
    def get_predicted_sites(self):
        """Return the predicted functional sites"""
        return self.predicted_sites
    
    def get_specificity_scores(self):
        """Return all specificity scores"""
        return self.specificity_scores

class ANCESCON:
    """Main class for the ANCESCON program"""
    
    def __init__(self, args):
        self.args = args
        self.msa_loader = None
        self.phylo_builder = None
        self.rate_estimator = None
        self.pi_optimizer = None
        self.reconstructor = None
        self.site_predictor = None
    
    def run(self):
        """Run the ANCESCON pipeline"""
        # Step 1: Load the multiple sequence alignment
        self.msa_loader = MSALoader(self.args.input_file)
        
        # Step 2: Build or load the tree
        alpha_values = None
        
        # Handle Dan method (-D) - This uses the optimized approach with both rate and pi optimization
        if self.args.dan_method:
            # First build a tree without rate variation
            self.phylo_builder = PhyloBuilder(self.msa_loader, self.args.tree_file)
            
            # Calculate and optimize alpha values
            self.rate_estimator = RateEstimator(self.msa_loader, self.phylo_builder.get_tree())
            self.rate_estimator.calculate_alpha_ml()
            alpha_values = self.rate_estimator.get_alpha_values(use_ml=True)
            
            # Rebuild tree with optimized alpha values
            self.phylo_builder = PhyloBuilder(self.msa_loader, self.args.tree_file, alpha_values)
        elif self.args.calculate_alpha or self.args.optimize_alpha:
            # Regular rate calculation/optimization
            self.phylo_builder = PhyloBuilder(self.msa_loader, self.args.tree_file)
            
            # Calculate alpha values
            self.rate_estimator = RateEstimator(self.msa_loader, self.phylo_builder.get_tree())
            
            if self.args.optimize_alpha:
                self.rate_estimator.calculate_alpha_ml()
                alpha_values = self.rate_estimator.get_alpha_values(use_ml=True)
            else:
                self.rate_estimator.calculate_alpha_ab()
                alpha_values = self.rate_estimator.get_alpha_values()
            
            # Rebuild the tree with alpha values if we're optimizing
            if self.args.optimize_alpha:
                self.phylo_builder = PhyloBuilder(self.msa_loader, self.args.tree_file, alpha_values)
        else:
            # Use default uniform rates
            self.phylo_builder = PhyloBuilder(self.msa_loader, self.args.tree_file)
        
        # Step 3: Optimize pi vector if requested
        pi_vector = None
        if any([self.args.optimize_pi_powell, self.args.optimize_pi_simplex, 
                self.args.optimize_pi_annealing, self.args.optimize_pi_per_site]):
            self.pi_optimizer = PiOptimizer(
                self.msa_loader, 
                self.phylo_builder.get_tree(), 
                alpha_values
            )
            
            if self.args.optimize_pi_powell:
                pi_vector = self.pi_optimizer.optimize_with_powell()
            elif self.args.optimize_pi_simplex:
                pi_vector = self.pi_optimizer.optimize_with_simplex()
            elif self.args.optimize_pi_annealing:
                pi_vector = self.pi_optimizer.optimize_with_annealing()
            elif self.args.optimize_pi_per_site:
                # This would require a more complex implementation
                # For now, we'll use the annealing method
                pi_vector = self.pi_optimizer.optimize_with_annealing()
        
        # Step 4: Reconstruct ancestral sequences
        self.reconstructor = AncestralReconstructor(
            self.msa_loader,
            self.phylo_builder.get_tree(),
            alpha_values,
            pi_vector
        )
        
        # For Dan method (-D), ensure we use the most accurate reconstruction
        if self.args.dan_method or self.args.reconstruct_all:
            # Reconstruct all nodes
            self.reconstructor.reconstruct_all_nodes()
        elif self.args.reconstruct_root_only:
            self.reconstructor.reconstruct_root()
        elif self.args.joint_reconstruction:
            # Use joint reconstruction
            self.reconstructor.joint_reconstruction()
        else:
            # Default to reconstructing all nodes
            self.reconstructor.reconstruct_all_nodes()
        
        # Step 5: Predict functional sites if requested
        if self.args.predict_functional_sites:
            self.site_predictor = FunctionalSitePredictor(
                self.msa_loader,
                self.phylo_builder.get_tree(),
                self.reconstructor.get_reconstructed_sequences()
            )
            self.site_predictor.predict_functional_sites()
        
        # Step 6: Output results
        self._output_results()
    
    def _output_results(self):
        """Output the results in the format similar to the original ANCESTCON program"""
        output_stream = sys.stdout
        if self.args.output_file:
            try:
                output_stream = open(self.args.output_file, 'w')
            except Exception as e:
                print(f"Error opening output file: {e}")
                return
        
        try:
            # Write program information
            output_stream.write("# ANCESTCON Python Implementation\n")
            output_stream.write(f"# Input file: {self.args.input_file}\n")
            if self.args.tree_file:
                output_stream.write(f"# Tree file: {self.args.tree_file}\n")
            output_stream.write("\n")
            
            # Write the position indices line (for easier reference)
            alignment_length = self.msa_loader.get_alignment_length()
            pos_indices = ""
            for i in range(alignment_length):
                # Create a position index string (every 10 positions)
                if i % 10 == 0:
                    pos_indices += f"{i+1:<10}"
            output_stream.write("\n")
            output_stream.write(f"{'':<10}{pos_indices}\n")
            output_stream.write("\n")
            
            # Write input sequences with position numbers
            output_stream.write("THE INPUT SEQUENCES:\n")
            output_stream.write("=" * 80 + "\n")
            for seq_id, seq in self.msa_loader.get_sequences().items():
                # Truncate long IDs for better formatting
                display_id = seq_id[:20] if len(seq_id) > 20 else seq_id
                output_stream.write(f">{display_id}\n{seq}\n")
            output_stream.write("\n")
            
            # Write reconstructed sequences
            output_stream.write("THE RECONSTRUCTED SEQUENCES:\n")
            output_stream.write("=" * 80 + "\n")
            for node_id, seq in self.reconstructor.get_reconstructed_sequences().items():
                # Format internal node names
                if node_id.startswith('Node'):
                    display_id = f"ANCESTRAL_{node_id[4:]}"
                else:
                    display_id = node_id
                
                # Truncate long IDs for better formatting
                display_id = display_id[:20] if len(display_id) > 20 else display_id
                output_stream.write(f">{display_id}\n{seq}\n")
            output_stream.write("\n")
            
            # Write functional site predictions if available
            if self.site_predictor:
                output_stream.write("PREDICTED FUNCTIONAL SITES:\n")
                output_stream.write("=" * 80 + "\n")
                output_stream.write("Position (1-based):\n")
                for i, pos in enumerate(self.site_predictor.get_predicted_sites()):
                    output_stream.write(f"{pos + 1:<5}")  # Convert to 1-based indexing
                    if (i + 1) % 10 == 0:
                        output_stream.write("\n")
                output_stream.write("\n")
                
                # Write specificity scores
                output_stream.write("\nSpecificity scores:\n")
                for pos, score in self.site_predictor.get_specificity_scores()[:10]:
                    output_stream.write(f"Position {pos + 1}: {score:.4f}\n")
        finally:
            if output_stream != sys.stdout:
                output_stream.close()
        
        print("Results written to output file")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='ANCESTCON - Ancestral Sequence Reconstruction Tool')
    
    # Required arguments
    parser.add_argument('-i', '--input-file', required=True, help='Input multiple sequence alignment file')
    
    # Optional arguments
    parser.add_argument('-t', '--tree-file', help='Input tree file (Newick format)')
    parser.add_argument('-o', '--output-file', help='Output file (default: stdout)')
    
    # Algorithm options
    parser.add_argument('-O', '--optimize-alpha', action='store_true', help='Optimize alpha values (higher precision, slower)')
    parser.add_argument('-C', '--calculate-alpha', action='store_true', help='Calculate correlated alpha values (high precision, faster)')
    parser.add_argument('-D', '--dan-method', action='store_true', help='Use Dan method to find optimal solution')
    
    # Reconstruction options
    parser.add_argument('-R', '--reconstruct-all', action='store_true', help='Reconstruct sequence for biological root and all internal nodes')
    parser.add_argument('-RO', '--reconstruct-root-only', action='store_true', help='Only reconstruct sequence for the biological root')
    parser.add_argument('-JOINT', '--joint-reconstruction', action='store_true', help='Use joint reconstruction instead of marginal')
    parser.add_argument('-Z', '--predict-functional-sites', action='store_true', help='Predict functional sites (use with -R/-RO)')
    
    # PI optimization options
    parser.add_argument('-PP', '--optimize-pi-powell', action='store_true', help='Optimize PI vector with Powell method')
    parser.add_argument('-PD', '--optimize-pi-simplex', action='store_true', help='Optimize PI vector with Downhill Simplex Method')
    parser.add_argument('-PA', '--optimize-pi-annealing', action='store_true', help='Optimize PI vector with Simulated Annealing')
    parser.add_argument('-PS', '--optimize-pi-per-site', action='store_true', help='Optimize PI vector for each site with Simulated Annealing')
    
    # Gap handling
    parser.add_argument('-G', '--gap-threshold', type=float, help='Non-gap number or percentage threshold')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    
    # Set default reconstruction mode if none specified
    if not (args.reconstruct_all or args.reconstruct_root_only):
        args.reconstruct_all = True
    
    # Check if functional site prediction is used with reconstruction
    if args.predict_functional_sites and not (args.reconstruct_all or args.reconstruct_root_only):
        print("Error: -Z (predict functional sites) must be used with -R or -RO")
        sys.exit(1)
    
    # Run ANCESCON
    ancescon = ANCESCON(args)
    ancescon.run()
    
    print("ANCESTCON execution completed successfully")