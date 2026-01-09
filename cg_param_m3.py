#!/usr/bin/env python

import os
import numpy as np
import itertools
import requests
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem import rdchem
from rdkit.Chem import rdMolDescriptors
from rdkit import RDConfig
import sys
#ssys.setrecursionlimit(999999999)
import re
import math
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import collections
import random
import time
import pandas as pd
import copy

# --- Global Data ---
delta_Gs = {
    0:{
        'standard':{
            'T': [-14.8,-15.2,-12.1,-9.8,-8.8,-7.2,-6.1,-4.9,-2.9,-3.1,0.3,2.3,3.6,4.5,6.4,6.7,7.8,12.0],
            'S': [-12.0,-11.8,-9.8,-7.7,-6.9,-5.2,-4.2,-3.6,-0.9,-1.8,2.1,3.6,5.3,6.3,8.4,9.2,9.9,14.2],
            'R': [-9.2,-9.1,-7.4,-5.1,-3.8,-2.0,-1.1,0.0,2.2,1.8,5.6,8.1,10.1,11.2,13.4,13.8,14.8,18.9]
        }
    },
    4:{
        'standard':{
            'T': [-5.23,-5.77,-3.77,-0.35,0.44,2.18,2.90,4.08,6.03,5.41,8.92,10.64,11.84,12.56,14.35,14.74,15.74,19.10],
            'S': [-3.89,-4.15,-1.84,-0.08,0.78,2.39,3.31,4.20,6.66,5.84,9.78,11.56,12.84,13.99,16.20,16.56,17.32,20.91],
            'R': [-4.27,-4.01,-1.64,0.26,1.66,3.55,4.53,5.43,7.93,7.43,11.49,13.79,15.63,16.77,18.90,19.61,20.59,24.01]
        }
    },
    3:{
        'standard':{
            'T': [-7.73,-8.25,-6.19,-2.63,-1.85,0.12,0.80,2.12,4.12,3.66,7.13,8.93,10.11,10.85,12.71,12.92,13.99,17.66],
            'S': [-5.45,-5.57,-3.22,-1.59,-0.61,0.99,1.87,2.82,5.28,4.46,8.36,10.01,11.38,12.53,14.61,15.07,15.78,19.58],
            'R': [-4.68,-4.33,-2.10,-0.27,1.13,3.05,4.04,4.93,7.32,6.90,10.69,12.92,14.77,15.83,18.11,18.61,19.73,23.19]
        }
    },
    2:{
        'standard':{
            'T': [-10.22,-10.74,-8.61,-4.98,-4.09,-2.10,-1.33,0.09,1.97,1.62,5.16,6.99,8.26,9.04,10.98,11.23,16.19],
            'S': [-7.64,-7.69,-5.34,-3.62,-2.57,-0.92,-0.10,0.90,3.47,2.51,6.57,8.29,9.61,10.85,12.95,13.49,14.21,18.13],
            'R': [-6.43,-6.10,-3.79,-1.94,-0.50,1.42,2.40,3.30,5.83,5.27,9.22,11.67,13.39,14.43,16.75,17.37,18.41,22.03]
        }}}

m3_beads = {
    'standard': ['P6','P5','P4','P3','P2','P1','N6','N5','N4','N3','N2','N1','C6','C5','C4','C3','C2','C1']
    }

preset_beads = {
    'CC':'TC2',
    'CCC':'SC2'
    }

# --- Helper Functions ---

def read_DG_data(DGfile):
    DG_data = {}
    with open(DGfile) as f:
        for line in f:
            (smi,DG,src) = line.rstrip().split()
            DG_data[smi] = {'DG':float(DG),'src':src}
    return DG_data

def include_weights(A,w):
    A_weighted = np.copy(A)
    for i,weight in enumerate(w):
        A_weighted[i,i] = weight
    return A_weighted

def get_weights(groups,w_init,path_matrix):
    w = []
    for node in groups:
        avgmass = get_avgmass(node,w_init)
        wi = avgmass * (get_size(node,path_matrix))
        w.append(wi)
    return w

def rank_nodes(A):
    vals,vecs = np.linalg.eig(A)
    maxval = np.argmax(np.real(vals))
    scores = np.absolute(vecs[:,maxval])
    ranked = np.argsort(scores)

    ties = []
    sublist = []

    score_prev = scores[ranked[0]]
    for i in ranked:
        score_i = scores[i]
        if np.isclose(score_i,score_prev):
            sublist.append(i)
        else:
            ties.append(sublist)
            sublist = [i]
        score_prev = score_i
    ties.append(sublist)

    return scores,ties

def check_connectivity(group, A_init):
    """
    Checks if a list of atoms (group) forms a connected component based on 
    the atomistic adjacency matrix A_init.
    """
    if len(group) <= 1:
        return True
    
    group_set = set(group)
    start_node = group[0]
    
    # BFS to check connectivity
    q = [start_node]
    visited = {start_node}
    count = 0
    
    while q:
        curr = q.pop(0)
        count += 1
        neighbors = np.nonzero(A_init[curr])[0]
        for neighbor in neighbors:
            if neighbor in group_set and neighbor not in visited:
                visited.add(neighbor)
                q.append(neighbor)
                
    return count == len(group)

def lone_atom(ties, A, A_init, scores, ring_beads, matched_maps, comp, exclusion_list):
    """
    Improved lone_atom function. Returns groups of ATOMS.
    Used during path_contraction (post-processing).
    """
    groups = []
    temp_exclusions = []
    MAX_ATOMS_PER_BEAD = 6 

    for rank in ties:
        for node in rank:
            if len(comp[node]) == 1:
                current_atom = comp[node][0]
                
                # Check if atom already processed
                if any(current_atom in g for g in groups) or current_atom in exclusion_list:
                    continue

                test_group = [current_atom]
                temp_exclusions.append(current_atom)

                # Neighbors in CG graph
                connects = A[node]
                # Note: matched_maps here should be ATOM lists, we check if node (bead) is in them
                # But typically lone_atom is used when beads=atoms largely.
                # Simplified check:
                bonded = [i for i in np.nonzero(connects)[0]]
                
                bonded_scores = np.asarray([scores[bonded[k]] for k in range(len(bonded))])
                bonded_sorted = np.argsort(bonded_scores)
                
                merged = False
                
                for j in bonded_sorted:
                    nbor_idx = bonded[j]
                    neighbor_atoms = comp[nbor_idx]
                    
                    if any(x in exclusion_list for x in neighbor_atoms):
                        continue
                    
                    if len(test_group) + len(neighbor_atoms) > MAX_ATOMS_PER_BEAD:
                        continue
                        
                    # Check if neighbor is part of a ring
                    if any(np.size(np.intersect1d(neighbor_atoms, ring)) != 0 for ring in ring_beads):
                         pass

                    combined_atoms = test_group + neighbor_atoms
                    if not check_connectivity(combined_atoms, A_init):
                        continue
                    
                    test_group.extend(neighbor_atoms)
                    merged = True
                    break 

                groups.append(test_group)

    exclusion_list.extend(temp_exclusions)
    
    temp_groups = groups[:]
    final_groups = []
    
    processed_atoms = set()
    for g in temp_groups:
        is_new = True
        for existing in final_groups:
            if not set(g).isdisjoint(set(existing)):
                is_new = False 
                break
        if is_new:
            final_groups.append(g)
            for a in g: processed_atoms.add(a)

    for bead in comp:
        # Add remaining beads (as atom lists)
        remaining = [a for a in bead if a not in processed_atoms]
        if remaining:
             # This assumes bead integrity was kept mostly, or we split it
             # If bead was partially merged, we keep the rest.
             # Ideally check connectivity of remaining part too, but usually it's fine.
             final_groups.append(bead)

    # Re-evaluate ring_beads (return as indices of the new groups)
    new_ring_beads_indices = []
    for ring in ring_beads:
        for i,group in enumerate(final_groups): 
            if any(atom in ring for atom in group) and i not in new_ring_beads_indices:
                new_ring_beads_indices.append(i)

    return final_groups, new_ring_beads_indices, exclusion_list

def spectral_grouping(ties, A, scores, local_ring_indices, comp, path_matrix, max_size, local_matched_indices, A_init):
    """
    Performs spectral clustering on the current CG graph.
    Input ring/matched indices are INDICES into 'comp' (current beads).
    Returns groups of INDICES.
    """
    groups = []
    MAX_ATOMS = 6 
    
    processed_nodes = set()

    for rank in ties:
        for node in rank:
            if node in processed_nodes:
                continue
            
            # Prevent merging if node is part of a ring or matched structure
            if any(node in a for a in groups) or \
               any(node in a for a in local_ring_indices) or \
               any(node in m for m in local_matched_indices):
                continue 
            
            test_group = [node]
            connects = A[node]
            
            # Find neighbors
            bonded = [i for i in np.nonzero(connects)[0] if not (any(i in a for a in groups))]
            # Filter neighbors that are in rings/matches
            bonded = [i for i in bonded if not (any(i in a for a in local_matched_indices))]

            bonded_scores = np.asarray([scores[bonded[k]] for k in range(len(bonded))])
            bonded_sorted = np.argsort(bonded_scores)
            
            for j in bonded_sorted:
                neighbor_idx = bonded[j]
                if neighbor_idx in processed_nodes: continue

                current_atoms = []
                for g_idx in test_group: current_atoms.extend(comp[g_idx])
                neighbor_atoms = comp[neighbor_idx]
                proposed_atoms = current_atoms + neighbor_atoms

                is_connected = check_connectivity(proposed_atoms, A_init)
                is_small_enough = len(proposed_atoms) <= MAX_ATOMS
                is_path_ok = get_size(proposed_atoms, path_matrix) <= max_size

                if test_group == [node]:
                    if any(neighbor_idx in a for a in local_ring_indices):
                        continue
                    if is_path_ok and is_small_enough and is_connected:
                        test_group.append(neighbor_idx)
                        processed_nodes.add(neighbor_idx)
                    else:
                        break 
                
                elif np.isclose(scores[node], scores[neighbor_idx]):
                     if is_small_enough and is_connected:
                        test_group.append(neighbor_idx)
                        processed_nodes.add(neighbor_idx)
                else:
                    break
            
            processed_nodes.add(node)
            groups.append(test_group)

    # Flatten and resolve overlaps (standard clean up)
    final_cg_groups = [] 
    for group in groups:
        merged = False
        for i in range(len(final_cg_groups)):
            if np.size(np.intersect1d(group, final_cg_groups[i])) != 0:
                combined = np.unique(np.concatenate((group, final_cg_groups[i]), axis=None)).tolist()
                
                combined_atoms = []
                for idx in combined: combined_atoms.extend(comp[idx])
                
                if len(combined_atoms) <= MAX_ATOMS and check_connectivity(combined_atoms, A_init):
                    final_cg_groups[i] = combined
                    merged = True
                break
        if not merged:
            final_cg_groups.append(group)

    # Revert if too big
    cleaned_groups = []
    for k in final_cg_groups:
        compk = []
        for cg_idx in k:
            compk.extend(comp[cg_idx])

        if get_size(compk, path_matrix) > max_size or len(compk) > MAX_ATOMS:
            for x in k:
                cleaned_groups.append([x])
        else:
            cleaned_groups.append(k)
            
    # Process rings using the INDICES
    groups, _, _ = process_rings(local_ring_indices, local_matched_indices, cleaned_groups)
    
    return groups

def process_rings(ring_beads, matched_maps, groups):
    """
    Merges groups that share ring/matched components.
    All inputs and outputs are lists of INDICES (bead indices).
    """
    # If ring-bead not already in a bead, add as its own bead
    for bead in ring_beads:
        if not any(any(a in group for a in bead) for group in groups):
            groups.append(bead)
    
    for match in matched_maps:
        # Check if match is already fully contained
        if not any(set(match).issubset(set(group)) for group in groups):
             groups.append(match)
        
    # If bead includes part of a ring bead, add rest of ring bead
    for i in range(len(groups)):
        for bead in ring_beads:
            if np.size(np.intersect1d(bead, groups[i])) != 0:
                groups[i] = np.unique(np.concatenate((groups[i], bead), axis=None)).tolist()

    # Combine beads which share a ring bead
    new_groups = []
    for l in range(len(groups)):
        if any(any(atom in bead for bead in new_groups) for atom in groups[l]):
            continue
        new_group = groups[l][:]
        for m in range(len(groups)):
            if np.size(np.intersect1d(new_group, groups[m])) != 0:
                new_group = np.unique(np.concatenate((new_group, groups[m]), axis=None)).tolist()
        new_groups.append(new_group)
    
    # Return flattened groups
    groups = new_groups 
    
    return groups, [], [] 

def new_connectivity(groups_of_atoms, A_init):
    """
    Calculates connectivity between beads based on ATOM groups.
    """
    num_beads = len(groups_of_atoms)
    newA = np.zeros((num_beads, num_beads), dtype=int)
    
    for i in range(num_beads):
        for j in range(i + 1, num_beads):
            connected = False
            for atom_k in groups_of_atoms[i]:
                for atom_l in groups_of_atoms[j]:
                    # Bounds check
                    if atom_k < A_init.shape[0] and atom_l < A_init.shape[1]:
                        if A_init[atom_k, atom_l] == 1:
                            newA[i, j] = 1
                            newA[j, i] = 1
                            connected = True
                            break
                if connected:
                    break
    return newA

def iteration(results, itr, A_init, w_init, ring_beads_atoms, path_matrix, matched_maps_atoms):
    """
    Orchestrates one iteration of spectral clustering.
    ring_beads_atoms and matched_maps_atoms are lists of ATOM indices.
    """
    results_dict = dict.fromkeys(['A','comp'])

    if itr == 0:
        oldA = np.copy(A_init)
        comp = [[i] for i in range(len(w_init))] # Current beads are atoms
        w = w_init[:]
    else:
        oldA = results[itr-1]['A']
        comp = results[itr-1]['comp'] # Current beads are lists of atoms
        w = get_weights(comp,w_init,path_matrix)
        
    A_weighted = include_weights(oldA,w)
    scores, ties = rank_nodes(A_weighted)
    
    # --- Convert Atoms to Current Bead Indices for spectral_grouping ---
    # Create map: atom_index -> bead_index
    atom_to_bead = {}
    for bead_idx, atoms in enumerate(comp):
        for atom in atoms:
            atom_to_bead[atom] = bead_idx
            
    def map_atoms_to_indices(atom_groups):
        index_groups = []
        for group in atom_groups:
            bead_indices = set()
            valid = True
            for atom in group:
                if atom in atom_to_bead:
                    bead_indices.add(atom_to_bead[atom])
                else:
                    valid = False
            if valid and bead_indices:
                index_groups.append(list(bead_indices))
        return index_groups

    local_ring_indices = map_atoms_to_indices(ring_beads_atoms)
    local_matched_indices = map_atoms_to_indices(matched_maps_atoms)
    
    # --- Perform Grouping (returns groups of bead indices) ---
    groups_indices = spectral_grouping(ties, oldA, scores, local_ring_indices, comp, path_matrix, 3, local_matched_indices, A_init)
    
    # --- Convert Bead Indices back to Atom Lists ---
    comp_new = []
    for gj in groups_indices:
        # gj is a list of indices into 'comp'
        bead_atoms = []
        for bead_idx in gj:
             bead_atoms.extend(comp[bead_idx])
        comp_new.append(bead_atoms)
    
    # --- Calculate New Connectivity using Atom Lists ---
    results_dict['A'] = new_connectivity(comp_new, A_init)
    results_dict['comp'] = comp_new[:]

    return results_dict, ring_beads_atoms, matched_maps_atoms

def group_rings(A, ring_atoms, matched_maps, moli):
    """
    Initial grouping of ring fragments.
    """
    new_groups = []
    edge_frags = collections.OrderedDict()
    edge_frags["[R1][R1][R1][R1][R1][R1]"] =  [[0,1],[2,3],[4,5]]
    edge_frags["[R1][R1][R1][R1][R1]"] = [[0,1,2],[2,3]]
    edge_frags["[R1][R1][R1][R1]"] = [[0,1],[2,3]]
    edge_frags["[R1][R1][R1]"] =  [[0,1,2]]
    edge_frags["[R1][R1]"] = [[0,1]]
    edge_frags["[R2][R1][R2]"] = [[0,1,2]]

    for substruct in edge_frags:
        matches = moli.GetSubstructMatches(Chem.MolFromSmarts(substruct)) 
        for match in matches:
            all_shared = False
            for system in ring_atoms:
                if all(m in system for m in match):
                    all_shared = True
                    break
            if not all_shared:
                continue
            if substruct == "[R2][R1][R2]":
                overlap = False
                for matchj in matches:
                    if match != matchj:
                        if list(set(match).intersection(matchj)):
                            overlap = True
                            break
                if overlap:
                    continue
            for bead in edge_frags[substruct]:
                test_bead = [match[x] for x in bead]
                if not any(any(y in ngroup for ngroup in new_groups) for y in test_bead):
                    new_groups.append(test_bead)

    unmapped = []
    for ring in ring_atoms:
        for a in ring:
            if not any(a in group for group in new_groups):
                unmapped.append(a)
    
    if unmapped:
        unm_smi = Chem.rdmolfiles.MolFragmentToSmiles(moli,unmapped)
        unm_smi = unm_smi.upper()
        unm_mol = Chem.MolFromSmiles(unm_smi)
        unmapped_frags = Chem.GetMolFrags(unm_mol)
        for frag in unmapped_frags:
            indices = [unmapped[k] for k in frag]
            frag_smi = Chem.rdmolfiles.MolFragmentToSmiles(moli,unmapped).split(".")[0]
            frag_smi = frag_smi.upper()
            frag_mol = Chem.MolFromSmiles(frag_smi) 
            A_frag = np.asarray(Chem.GetAdjacencyMatrix(frag_mol))

            assign_atom_maps(frag_mol)
            core_map=re.findall(r"\:([^\]]*)\]",frag_smi)
            if len(core_map) != len(indices):
                core_map.insert(0,0)

            frag_ring_atoms = get_ring_atoms(frag_mol)
            if frag_ring_atoms:
                # Recurse for fragment rings
                # Note: group_rings returns ring_beads as ATOM lists in context of fragment
                frag_ring_beads_local, _, _ = group_rings(A_frag,frag_ring_atoms,matched_maps,frag_mol)
            else:
                frag_ring_beads_local = []

            # Map fragment local indices (0..N)
            comp = [[i] for i in range(frag_mol.GetNumAtoms())]
            
            # Map global matched_maps to local indices
            local_matched_maps = []
            g2l = {gid: i for i, gid in enumerate(indices)}
            for match in matched_maps:
                if set(match).issubset(set(indices)):
                    local_match = [g2l[m] for m in match]
                    local_matched_maps.append(local_match)

            if sum([len(b) for b in frag_ring_beads_local]) < len(frag): 
                path_frag = floyd_warshall(csgraph=A_frag,directed=False)
                w_frag = [1.0 for atom in frag_mol.GetAtoms()] 
                A_fragw = include_weights(A_frag,w_frag)
                scores,ties = rank_nodes(A_fragw)
                
                # Spectral grouping on fragment. Returns INDICES.
                new_indices = spectral_grouping(ties, A_frag, scores, frag_ring_beads_local, comp, path_frag, 2, local_matched_maps, A_frag)
                
                # Convert INDICES to ATOM lists (global IDs)
                for bead_indices in new_indices:
                    # check if matched
                    is_match = False
                    # Check against local matched maps
                    for m in local_matched_maps:
                         if sorted(m) == sorted(bead_indices):
                             is_match = True
                    
                    global_atom_indices = []
                    # bead_indices are local 0..N. Map to global core_map/indices
                    # The original code logic mapped via core_map
                    bead_global = [int(core_map[x]) for x in bead_indices]
                    
                    if not is_match:
                         new_groups.append(bead_global)
            else:
                 # Just use the ring beads found
                 for bead_indices in frag_ring_beads_local:
                      new_groups.append([int(core_map[x]) for x in bead_indices])

    # Construct final ring_beads list (Atom Lists) from new_groups
    ring_beads_out = []
    # Identify which of new_groups are rings
    for group in new_groups:
        # If this group overlaps with any original ring_atoms, keep track
        if any(any(a in ring for ring in ring_atoms) for a in group):
            ring_beads_out.append(group)

    # Add matched maps to new_groups if not present
    for m in matched_maps:
        if not any(set(m).issubset(set(g)) for g in new_groups):
             new_groups.append(m)

    # Add remaining single atoms
    # Use sets for speed
    grouped_atoms = set(itertools.chain.from_iterable(new_groups))
    for i in range(A.shape[0]):
        if i not in grouped_atoms:
            new_groups.append([i])

    return ring_beads_out, new_groups, A

def postprocessing(results,ring_atoms,n_iter,A_init,w_init,path_matrix,matched_maps):
    last_iter = results[n_iter -1]
    exclusion_list = []
    postprocess = 1
    
    loop_count = 0
    max_loops = 5

    while postprocess and loop_count < max_loops:
        min_size = 1000 
        avg_size = 0
        count = 0.0
        for i,bead in enumerate(last_iter['comp']):
            size = len(bead)
            if size < min_size:
                min_size = size
            if i not in ring_atoms: # Ring atoms here is confusing, usually ring_beads index.
                # But standard postprocessing logic checks size.
                avg_size += size
                count += 1.0
        
        if count > 0:
            avg_size = avg_size / count
        else:
            avg_size = 0
    
        if min_size == 1:
            postprocess = 1
        else:
            postprocess = 0

        if postprocess:
            results_dict, _, exclusion_list = path_contraction(last_iter, postprocess, A_init, w_init, ring_atoms, matched_maps, path_matrix, exclusion_list)
            last_iter = results_dict.copy()
            loop_count += 1
        else:
            results_dict = last_iter.copy()
    
    return results_dict

def path_contraction(last_iter, postprocess, A_init, w_init, ring_beads, matched_maps, path_matrix, exclusion_list):
    """
    Merging single atom beads into neighbors.
    ring_beads here refers to lists of ATOMS.
    """
    results_dict = dict.fromkeys(['A','comp'])

    oldA = last_iter['A']
    comp = last_iter['comp']
    w = get_weights(comp,w_init,path_matrix)

    A_weighted = include_weights(oldA,w)
    scores, ties = rank_nodes(A_weighted)
    
    # lone_atom returns groups of ATOMS
    groups_atoms, new_ring_indices, exclusion_list = lone_atom(ties, oldA, A_init, scores, ring_beads, matched_maps, comp, exclusion_list)
    
    results_dict['A'] = new_connectivity(groups_atoms, A_init)
    results_dict['comp'] = groups_atoms[:]
    
    # We pass back ring_beads (atoms) as is, or we could update if merged.
    # Usually rings are preserved.
    return results_dict, ring_beads, exclusion_list

def get_size(comp,path_matrix):
    longpath = 0
    for i in comp:
        for j in comp:
            path = path_matrix[i,j]
            if path > longpath:
                    longpath = path
    return longpath

def get_avgmass(comp,masses):
    avgmass = sum([masses[i] for i in comp])/len(comp)
    return avgmass
    
def assign_atom_maps(mol_dict):
    for atom in mol_dict.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol_dict

def mapping(mol,ring_atoms,matched_maps,n_iter,mol_dict):
    A_atom = np.asarray(Chem.GetAdjacencyMatrix(mol))
    path_matrix = floyd_warshall(csgraph=A_atom,directed=False)
    w_init = [atom.GetMass() for atom in mol.GetAtoms()]
    
    assign_atom_maps(mol_dict) 
    # group_rings returns ring_beads (Atoms) and comp (Atoms)
    ring_beads, comp, A_init = group_rings(A_atom, ring_atoms, matched_maps, mol_dict)

    results = []
    # itr=0 state
    results.append({'A': new_connectivity(comp, A_atom), 'comp': comp})
    
    current_ring_beads = ring_beads # Atoms
    current_matched_maps = matched_maps # Atoms

    for itr in range(n_iter):
        results_dict, current_ring_beads, current_matched_maps = iteration(results, itr+1, A_atom, w_init, current_ring_beads, path_matrix, current_matched_maps)
        results.append(results_dict)
 
    results_dict_final = postprocessing(results, ring_atoms, n_iter, A_atom, w_init, path_matrix, matched_maps)
    
    # Reconstruct final ring beads (indices) for output
    final_ring_beads_indices = []
    for ring in ring_atoms:
        cgring = []
        for atom in ring:
            for i,bead in enumerate(results_dict_final['comp']):
                if (atom in bead) and (i not in cgring):
                    cgring.append(i)
        final_ring_beads_indices.append(cgring)

    return results_dict_final['A'], results_dict_final['comp'], final_ring_beads_indices, path_matrix

def get_ring_atoms(mol):
    rings = mol.GetRingInfo().AtomRings()
    ring_systems = []
    for ring in rings:
        ring_atoms = set(ring)
        new_systems = []
        for system in ring_systems:
            shared = len(ring_atoms.intersection(system))
            if shared:
                ring_atoms = ring_atoms.union(system)
            else:
                new_systems.append(system)
        new_systems.append(ring_atoms)
        ring_systems = new_systems
    return [list(ring) for ring in ring_systems]
        
def get_hbonding(mol,beads):
    fdefName = os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    feats = factory.GetFeaturesForMol(mol)
 
    h_donor = []
    h_acceptor = []
    for feat in feats:
        if feat.GetFamily() == "Donor":
            for i in feat.GetAtomIds():
                for b,bead in enumerate(beads):
                    if i in bead:
                       if b not in h_donor:
                           h_donor.append(b)
                       break
        if feat.GetFamily() == "Acceptor":
            for i in feat.GetAtomIds():
                for b,bead in enumerate(beads):
                    if i in bead:
                       if b not in h_acceptor:
                           h_acceptor.append(b)
                       break
    return h_donor,h_acceptor

def get_smi(bead,mol):
    bead_smi = Chem.rdmolfiles.MolFragmentToSmiles(mol,bead)
    lc = re.compile('[cn([nH\\])os]+')
    string_lst = ['c','\\[nH\\]','(?<!\\[)n','o']
    lowerlist = re.findall(r"(?=("+'|'.join(string_lst)+r"))",bead_smi)
    
    ring_size = 0
    frag_size = 0
    if lowerlist:
        frag_size = len(lowerlist)
        if frag_size == 2:
            subs = bead_smi.split(''.join(lowerlist))
            for i in range(len(subs)):
                if subs[i] != '':
                    subs[i] = '({})'.format(subs[i])
            try:
                bead_smi = 'c1c{}{}{}{}cc1'.format(lowerlist[0],subs[0],lowerlist[1],subs[1])
            except:
                bead_smi = Chem.rdmolfiles.MolFragmentToSmiles(mol,bead,kekuleSmiles=True)
            ring_size = 6
            if not Chem.MolFromSmiles(bead_smi): 
                bead_smi = 'c1c{}{}{}{}c1'.format(lowerlist[0],subs[0],lowerlist[1],subs[1])
                ring_size = 5
        elif len(lowerlist) == 3:
            split1 = bead_smi.split(''.join(lowerlist[:2]))
            split2 = split1[1].split(lowerlist[2])
            subs = [split1[0],split2[0],split2[1]]
            for i in range(len(subs)):
                if subs[i] != '' and subs[i][0] != '(':
                    subs[i] = '({})'.format(subs[i])
            try:
                bead_smi = 'c1c{}{}{}{}{}{}c1'.format(lowerlist[0],subs[0],lowerlist[1],subs[1],lowerlist[2],subs[2])       
            except:
                bead_smi = Chem.rdmolfiles.MolFragmentToSmiles(mol,bead,kekuleSmiles=True)

            ring_size = 6
            if not Chem.MolFromSmiles(bead_smi):
                bead_smi = 'c1{}{}{}{}{}{}c1'.format(lowerlist[0],subs[0],lowerlist[1],subs[1],lowerlist[2],subs[2])
                ring_size = 5

    if not Chem.MolFromSmiles(bead_smi):
        bead_smi = Chem.rdmolfiles.MolFragmentToSmiles(mol,bead,kekuleSmiles=True)
        bead_smi=bead_smi.replace(":","") 
        ring_size = 0
        frag_size = 0

    bead_smi = Chem.rdmolfiles.MolToSmiles(Chem.MolFromSmiles(bead_smi))
    return bead_smi,ring_size,frag_size
    
def get_types(beads,mol,ring_beads):
    script_path = os.path.dirname(os.path.realpath(__file__))
    try:
        DG_data = read_DG_data('{}/fragments-exp.dat'.format(script_path))
    except:
        DG_data = {}

    bead_types = []
    charges = []
    all_smi = []
    h_donor,h_acceptor = get_hbonding(mol,beads)
    for i,bead in enumerate(beads):
        qbead = sum([mol.GetAtomWithIdx(int(j)).GetFormalCharge() for j in bead])
        charges.append(qbead)
        bead_smi,ring_size, frag_size = get_smi(bead,mol)
        all_smi.append(bead_smi)
        bead_types.append(param_bead(bead,bead_smi,ring_size,frag_size,any(i in ring for ring in ring_beads),qbead,i in h_donor,i in h_acceptor,DG_data))

    if tuning:
        bead_types = tune_model(beads,bead_types,all_smi)

    return bead_types,charges,all_smi,DG_data

def tune_model(beads,bead_types,all_smi):
    scores,ties = rank_nodes(A_cg)

    def is_tunable(nbor):
        if any(nbor in ring for ring in ring_beads):
            return False
        elif not any(element in all_smi[nbor] for element in ['O','N','S','F','Cl','Br','I']):
            return False
        else:
            return True

    tuned = []
    fixed = []

    for rank in ties:
        for bead in rank:
            if not any(bead in ring for ring in ring_beads):
                bonded = [j for j in np.nonzero(A_cg[bead])[0]]
                for nbor in bonded:
                    if scores[nbor] >= scores[bead] and is_tunable(nbor):
                        tuned.append(nbor)
                        fixed.append(bead)
                if (bead not in fixed) and is_tunable(bead) and len(bonded) >= 1:
                    tuned.append(bead)    
                    fixed.append(bonded[0])
    
    for t,f in zip(tuned,fixed):
        bead_types[t] = tune_bead(beads[t],bead_types[t],beads[f],bead_types[f])
    
    return bead_types

def get_diffs(alogps,ring_size,frag_size,category,size):
    diffs = np.abs(np.array(delta_Gs[ring_size-frag_size][category][size]) - alogps)
    return diffs

def param_bead(bead,bead_smi,ring_size,frag_size,ring,qbead,don,acc,DG_data):
    btype = ''
    for m,match in enumerate(matched_maps):
        if sorted(match) == sorted(bead):
            btype = matched_beads[m]

    if bead_smi in preset_beads:
        btype = preset_beads[bead_smi]

    category = 'standard'
    suffix = ''
    types = m3_beads[category]
    path_length = get_size(bead,path_matrix)
        
    if path_length == 1:
        size = 'T'
        prefix = 'T'
    elif path_length == 2:
        size = 'S'
        prefix = 'S'
    else:
        size = 'R'
        prefix = ''

    if btype == '':
        if qbead != 0:
            btype = 'Qx' 
        else:
            try:
                alogps = DG_data[bead_smi]['DG']
            except:
                print('{} not on list'.format(bead_smi))
                alogps = get_alogps(bead_smi)

            diffs = get_diffs(alogps,ring_size,frag_size,category,size)
            sort_diffs = np.argsort(diffs)
            btype = types[sort_diffs[0]]

        btype = prefix + btype + suffix

    return btype                        

def get_alogps(bead_smi):
    try:
        alogps = requests.get('http://vcclab.org/web/alogps/calc?SMILES=' + bead_smi).text
    except:
        logK = rdMolDescriptors.CalcCrippenDescriptors(Chem.MolFromSmiles(bead_smi))[0]
        print(bead_smi,'Data from Wildmann-Crippen')
        return logK*5.74
    if 'error' not in alogps:
        try:
            logK = float(alogps.split()[4])
        except:
             logK = rdMolDescriptors.CalcCrippenDescriptors(Chem.MolFromSmiles(bead_smi))[0]
    else:
        logK = rdMolDescriptors.CalcCrippenDescriptors(Chem.MolFromSmiles(bead_smi))[0]
        print(bead_smi,'Data from Wildmann-Crippen')
    return logK*5.74

def bead_coords(bead,conf):
    coords = np.array([0.0,0.0,0.0])
    total = 0.0
    for atom in bead:
        mass = mol.GetAtomWithIdx(atom).GetMass() 
        coords += conf.GetAtomPosition(atom)*mass
        total += mass
    coords /= (total)
    return coords

def write_gro(mol_name,bead_types,coords0,gro_name):
    with open(gro_name,'w') as gro:
        gro.write('single molecule of {}\n'.format(mol_name))
        gro.write('{}\n'.format(len(bead_types)))
        i = 1
        for bead,xyz in zip(bead_types,coords0):
            gro.write('{:5d}{:5}{:>5}{:5d}{:8.3f}{:8.3f}{:8.3f}\n'.format(1,mol_name,bead,i,xyz[0]/10.0,xyz[1]/10.0,xyz[2]/10.0))
            i += 1
        gro.write('5.0 5.0 5.0')

def get_virtual_sites(ring,coords,A_cg):
    coords_r = np.empty((len(ring),3))
    for i,a in enumerate(ring):
        coords_r[i] = coords[a]

    com = np.sum(coords_r,axis=0)/coords_r.shape[0]
    coords_c = np.subtract(coords_r,com)

    I_xx = sum([(c[1]**2 + c[2]**2) for c in coords_c])
    I_yy = sum([(c[0]**2 + c[2]**2) for c in coords_c])
    I_zz = sum([(c[0]**2 + c[1]**2) for c in coords_c])
    I_xy = -sum([(c[0]*c[1]) for c in coords_c])
    I_xz = -sum([(c[0]*c[2]) for c in coords_c])
    I_yz = -sum([(c[1]*c[2]) for c in coords_c])
    I = np.array([[I_xx,I_xy,I_xz],[I_xy,I_yy,I_yz],[I_xz,I_yz,I_zz]])

    Ivals,Ivecs = np.linalg.eig(I)
    Isort = np.argsort(Ivals)
    plane_x = Ivecs[:,Isort[0]]
    plane_y = Ivecs[:,Isort[1]]
    
    coords_p = np.empty((coords_c.shape[0],2))
    for i,coord in enumerate(coords_c):
        coords_p[i][0] = np.dot(plane_x,coord)
        coords_p[i][1] = np.dot(plane_y,coord)

    if len(ring) <= 3:
        real_sites = [r for r in ring]
        virtual_sites = []
    else:
        hull = ConvexHull(coords_p)
        verts = hull.vertices
        real_sites = [ring[j] for j in verts]
        virtual_sites = [site for site in ring if site not in real_sites]

    for vs in list(virtual_sites):
        bonded = [j for j in np.nonzero(A_cg[vs])[0]]
        rvs = coords[vs]
        for b in bonded:
            if b not in ring:
                virtual_sites.remove(vs)
                min_v = 100000
                closest = 0
                for e in range(len(real_sites)):
                    ra = coords[real_sites[e]]
                    rb = coords[real_sites[(e+1)%(len(real_sites))]]
                    rab = np.subtract(rb,ra)
                    rav = np.subtract(rvs,ra)
                    rproj = np.add(ra,(np.dot(rab,rav)/np.dot(rab,rab))*rab)
                    dist = np.linalg.norm(np.subtract(rvs,rproj))
                    if dist < min_v:
                        closest = e
                        min_v = dist
                real_sites.insert((closest+1)%len(real_sites),vs)
                break

    vs_weights = {}
    for vs in virtual_sites:
        vs_weights[vs] = (construct_vs(ring.index(vs),verts,coords_p,ring))

    return real_sites,vs_weights

def construct_vs(vs,real_sites,coords_p,ring):
    dists = [np.linalg.norm(coords_p[vs]-coords_p[rs]) for rs in real_sites]
    weights = {}
    vx,vy = coords_p[vs]

    if len(real_sites) >= 4:
        closest = np.argsort(dists)[:4]
        vertices = [real_sites[r] for r in range(len(real_sites)) if r in closest]
        r1x,r1y = coords_p[vertices[0]]
        r2x,r2y = coords_p[vertices[3]]
        r3x,r3y = coords_p[vertices[1]]
        r4x,r4y = coords_p[vertices[2]]
        tx = r4x + r1x -r3x - r2x
        ty = r4y + r1y - r3y - r2y
        c = ((r1y-vy)*(r3x-r1x) - (r1x-vx)*(r3y-r1y))
        b = (r2y-r1y)*(r3x-r1x) + (r1y-vy)*tx - (r2x-r1x)*(r3y-r1y) - (r1x-vx)*ty
        a = (r2y-r1y)*tx - (r2x-r1x)*ty
        roots = np.roots([a,b,c])

        # --- 修复点：安全地获取 f1 ---
        f1 = 0.5 # 默认值
        for f in roots:
            if np.isreal(f):
                f_real = np.real(f)
                if (f_real >= 0.0 and f_real <= 1.0) or np.isclose(f_real,1.0) or np.isclose(f_real,0.0):
                    f1 = f_real
                    break
        
        # 避免分母为0
        denom = ( (r3x-r1x) + f1*tx)
        if abs(denom) < 1e-6:
             f2 = 0.5
        else:
             f2 = -( (r1x-vx) + f1*(r2x-r1x)) / denom

        weights = {}
        weights[ring[vertices[0]]] = (1-f1)*(1-f2)
        weights[ring[vertices[3]]] = f1*(1-f2)
        weights[ring[vertices[1]]] = (1-f1)*f2
        weights[ring[vertices[2]]] = f1*f2

    elif len(real_sites) == 3:
        vertices = real_sites[:]
        r1x,r1y = coords_p[vertices[0]]
        r2x,r2y = coords_p[vertices[1]]
        r3x,r3y = coords_p[vertices[2]]

        M = np.array([[(r2x-r1x),(r3x-r1x)],[(r2y-r1y),(r3y-r1y)]])
        B = np.array([(vx-r1x),(vy-r1y)])
        try:
            P = np.linalg.solve(M,B)
            weights[ring[vertices[1]]] = P[0]
            weights[ring[vertices[2]]] = P[1]
            weights[ring[vertices[0]]] = 1.0 - P[0] - P[1]
        except:
            # 奇异矩阵 fallback
            weights[ring[vertices[0]]] = 0.33
            weights[ring[vertices[1]]] = 0.33
            weights[ring[vertices[2]]] = 0.34

    return weights

def ring_bonding(real,virtual,A_cg,dihedrals):
    for vs in list(virtual.keys()):
        for i in range(A_cg.shape[0]):
            A_cg[vs,i] = 0
            A_cg[i,vs] = 0

    A_cg[real[0],real[-1]] = 1
    A_cg[real[-1],real[0]] = 1
    for r in range(len(real)-1):
        A_cg[real[r],real[r+1]] = 1
        A_cg[real[r+1],real[r]] = 1
    
    n_struts = len(real)-3
    j = len(real)-1
    k = 1
    struts = 0
    for s in range(int(math.ceil(n_struts/2.0))):
        A_cg[real[j],real[k]] = 1
        A_cg[real[k],real[j]] = 1
        struts += 1
        i = (j+1)%len(real) 
        l = k+1
        dihedrals.append([real[i],real[j],real[k],real[l]])
        k += 1
        if struts == n_struts:
            break
        A_cg[real[j],real[k]] = 1
        A_cg[real[k],real[j]] = 1
        struts += 1
        i = k-1
        l = j-1
        dihedrals.append([real[i],real[j],real[k],real[l]])
        j -= 1

    return A_cg,dihedrals
        
def get_masses(all_smi,A_cg,virtual):
    m_H = 1.00727645209
    masses = []
    for b,smi in enumerate(all_smi):
        aa_frag = Chem.MolFromSmiles(smi)
        frag_mass = rdMolDescriptors.CalcExactMolWt(aa_frag)
        excess_mass = np.sum(A_cg[b])*m_H
        masses.append(frag_mass-excess_mass)

    for vsite,refs in virtual.items():
        vmass = masses[vsite]
        masses[vsite] = 0.0
        for rsite,weight in refs.items():
            masses[rsite] += weight*vmass

    return masses
            
def write_itp(mol_name,bead_types,coords0,charges,all_smi,A_cg,itp_name):
    with open(itp_name,'w') as itp:
        itp.write('[moleculetype]\n')
        itp.write('MOL    2\n')
        virtual,real = write_atoms(itp,A_cg,mol_name,bead_types,charges,all_smi,coords0,ring_beads)
        bonds,constraints,dihedrals = write_bonds(itp,A_cg,ring_beads,real,virtual)
        angles = write_angles(itp,bonds,constraints)
        if dihedrals:
            write_dihedrals(itp,dihedrals,coords0)
        if virtual:
            write_virtual_sites(itp,virtual)

def write_atoms(itp,A_cg,mol_name,bead_types,charges,all_smi,coords,ring_beads):
    real = []
    virtual = {}
    for ring in ring_beads:
        rs,vs = get_virtual_sites(ring,coords,A_cg)
        virtual.update(vs)
        real.append(rs)

    masses = get_masses(all_smi,A_cg,virtual)
    itp.write('\n[atoms]\n')
    for b in range(len(bead_types)):
        itp.write('{:5d}{:>5}{:5d}{:>5}{:>5}{:5d}{:>10.3f}{:>10.3f};{}\n'.format(b+1,bead_types[b],1,mol_name,'CG'+str(b+1),b+1,charges[b],masses[b],all_smi[b]))
    return virtual,real
    
def write_bonds(itp,A_cg,ring_atoms,real,virtual):
    dihedrals = []
    for r,ring in enumerate(ring_atoms):
        A_cg,dihedrals = ring_bonding(real[r],virtual,A_cg,dihedrals)

    itp.write('\n[bonds]\n')
    bonds = [list(pair) for pair in np.argwhere(A_cg) if pair[1] > pair[0]]
    constraints = []
    k = 1250.0

    rs = np.zeros(len(bonds))
    coords = np.zeros((len(beads),3))
    for conf in mol.GetConformers():
        for i,bead in enumerate(beads):
            coords[i] = bead_coords(bead,conf)
        for b,bond in enumerate(bonds):
            rs[b] += np.linalg.norm(np.subtract(coords[bond[0]],coords[bond[1]]))/nconfs 
            
    rs = rs / 10.0 

    con_rs = []
    for bond,r in zip(bonds,rs):
        share_ring = False
        for ring in ring_atoms:
            if bond[0] in ring and bond[1] in ring:
                share_ring = True
                constraints.append(bond)
                con_rs.append(r)
                break
        if not share_ring:
            itp.write('{:5d}{:3d}{:5d}{:10.3f}{:10.1f}\n'.format(bond[0]+1,bond[1]+1,1,r,k))

    if len(constraints) > 0:
        itp.write('\n#ifdef min\n')
        k = 5000000.0
        for con,r in zip(constraints,con_rs):
            itp.write('{:5d}{:3d}{:5d}{:10.3f}{:10.1f}\n'.format(con[0]+1,con[1]+1,1,r,k))

        itp.write('\n#else\n')
        itp.write('[constraints]\n')
        for con,r in zip(constraints,con_rs):
            itp.write('{:5d}{:3d}{:5d}{:10.3f}\n'.format(con[0]+1,con[1]+1,1,r))
        itp.write('#endif\n')

    return bonds,constraints,dihedrals

def write_angles(itp,bonds,constraints):
    k = 25.0
    angles = []
    for bi in range(len(bonds)-1):
        for bj in range(bi+1,len(bonds)):
            shared = np.intersect1d(bonds[bi],bonds[bj])
            if np.size(shared) == 1:
                if bonds[bi] not in constraints or bonds[bj] not in constraints:
                    x = [i for i in bonds[bi] if i != shared][0]
                    z = [i for i in bonds[bj] if i != shared][0]
                    angles.append([x,int(shared),z])
    if angles:
        itp.write('\n[angles]\n')
        coords = np.zeros((len(beads),3))
        thetas = np.zeros(len(angles))
        for conf in mol.GetConformers():
            for i,bead in enumerate(beads):
                coords[i] = bead_coords(bead,conf)
            for a,angle in enumerate(angles):
                vec1 = np.subtract(coords[angle[0]],coords[angle[1]])
                vec1 = vec1/np.linalg.norm(vec1)
                vec2 = np.subtract(coords[angle[2]],coords[angle[1]])
                vec2 = vec2/np.linalg.norm(vec2)
                val = np.dot(vec1,vec2)
                val = max(min(val, 1.0), -1.0)
                theta = np.arccos(val)
                thetas[a] += theta

        thetas = thetas*180.0/(np.pi*nconfs)
        for a,t in zip(angles,thetas):
            itp.write('{:5d}{:3d}{:3d}{:5d}{:10.3f}{:10.1f}\n'.format(a[0]+1,a[1]+1,a[2]+1,2,t,k))

def write_dihedrals(itp,dihedrals,coords0):
    itp.write('\n[dihedrals]\n')
    k = 500.0
    for dih in dihedrals:
        vec1 = np.subtract(coords0[dih[1]],coords0[dih[0]])
        vec2 = np.subtract(coords0[dih[2]],coords0[dih[1]])
        vec3 = np.subtract(coords0[dih[3]],coords0[dih[2]])
        vec1 = vec1/np.linalg.norm(vec1)
        vec2 = vec2/np.linalg.norm(vec2)
        vec3 = vec3/np.linalg.norm(vec3)
        cross1 = np.cross(vec1,vec2)
        cross1 = cross1/np.linalg.norm(cross1)
        cross2 = np.cross(vec2,vec3)
        cross2 = cross2/np.linalg.norm(cross2)
        val = np.dot(cross1,cross2)
        val = max(min(val, 1.0), -1.0)
        angle = np.arccos(val)*180.0/np.pi
        itp.write('{:5d}{:3d}{:3d}{:3d}{:5d}{:10.3f}{:10.1f}\n'.format(dih[0]+1,dih[1]+1,dih[2]+1,dih[3]+1,2,angle,k))

def write_virtual_sites(itp,virtual_sites):
    itp.write('\n[virtual_sitesn]\n')
    vs_iter = sorted(virtual_sites.keys())
    for vs in vs_iter:
        cs = sorted(virtual_sites[vs].items())
        if len(cs) == 4:
            itp.write('{:5d}{:3d}{:5d}{:7.3f}{:5d}{:7.3f}{:5d}{:7.3f}{:5d}{:7.3f}\n'.format(vs+1,3,cs[0][0]+1,cs[0][1],cs[1][0]+1,cs[1][1],cs[2][0]+1,cs[2][1],cs[3][0]+1,cs[3][1]))
        elif len(cs) == 3:
            itp.write('{:5d}{:3d}{:5d}{:7.3f}{:5d}{:7.3f}{:5d}{:7.3f}\n'.format(vs+1,3,cs[0][0]+1,cs[0][1],cs[1][0]+1,cs[1][1],cs[2][0]+1,cs[2][1]))
    
    itp.write('\n[exclusions]\n')
    done = []
    for vs in vs_iter:
        excl = str(vs+1)
        for i in range(len(beads)):
            if i != vs and i not in done:
                excl += ' '+str(i+1)
        done.append(vs)
        itp.write('{}\n'.format(excl))

def get_coords(mol,beads):
    mol_Hs = Chem.AddHs(mol)
    conf = mol_Hs.GetConformer(0)
    cg_coords = []
    for bead in beads:
        coord = np.array([0.0,0.0,0.0])
        total = 0.0
        for atom in bead:
            mass = mol.GetAtomWithIdx(atom).GetMass()
            coord += conf.GetAtomPosition(atom)*mass
            total += mass
        coord /= (total)
        cg_coords.append(coord)
    cg_coords_a = np.array(cg_coords)
    return cg_coords_a

def get_smarts_matches(mol):
    smarts_strings = {
    'S([O-])(=O)(=O)O'  :    'Q2',
    '[S;!$(*OC)]([O-])(=O)(=O)'   :    'SQ4p',
    'CC[N+](C)(C)[O-]' : 'P6',
    'CC(=O)[O-]' : 'SQ5n',
    'CC[N+D1]' : 'SQ4p',
    'C[N+]C' : 'SQ3p',
    'C[N+](C)(C)C' : 'Q2',
    'CC(C)[N+]C' : 'Q2p'
    }
    matched_maps = []
    matched_beads = []
    for smarts in smarts_strings:
        matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
        for match in matches:
            matched_maps.append(list(match))
            matched_beads.append(smarts_strings[smarts])
    return matched_maps,matched_beads

def tune_bead(var_bead,var_type,fix_bead,fix_type):
    print(var_bead,var_type,fix_bead,fix_type)
    dimer_smi = Chem.rdmolfiles.MolFragmentToSmiles(mol,fix_bead+var_bead)
    dimer_DG = get_alogps(dimer_smi)

    var_size = var_type[0] if (var_type[0] in ['T','S']) else 'R'
    fix_size = fix_type[0] if (fix_type[0] in ['T','S']) else 'R'

    var_cat = 'standard'
    fix_cat = 'standard'
    fix_base = fix_type[1:] if (len(fix_type) == 3) else fix_type

    fix_DG = delta_Gs[0][fix_cat][fix_size][m3_beads[fix_cat].index(fix_base)]
    dimer_sum = np.asarray(delta_Gs[0][var_cat][var_size]) + fix_DG
    dimer_diff = np.abs(dimer_sum - dimer_DG)
    var_base = m3_beads[var_cat][np.argmin(dimer_diff)]

    var_type = var_base if (var_size == 'R') else (var_size+var_base)
    return var_type


# --- New Function: PEG Bead Identification ---
def get_peg_beads_groups(mol):
    """
    Identifies PEG chains (continuous CCO segments >= 2 units) and maps them into 4-atom beads.
    """
    # 1. Identify atoms participating in C-C-O patterns
    peg_pattern = Chem.MolFromSmarts("CCO")
    matches = mol.GetSubstructMatches(peg_pattern)
    peg_atoms = set()
    for m in matches:
        peg_atoms.update(m)
    
    if not peg_atoms:
        return []

    # 2. Cluster specific PEG atoms based on connectivity
    peg_atom_list = list(peg_atoms)
    
    # Simple connected components finding for subset of atoms
    components = []
    visited = set()
    mol_adj = Chem.GetAdjacencyMatrix(mol)
    
    for atom in peg_atom_list:
        if atom not in visited:
            # BFS to find component
            component = []
            queue = [atom]
            visited.add(atom)
            while queue:
                curr = queue.pop(0)
                component.append(curr)
                # Check neighbors
                neighbors = np.nonzero(mol_adj[curr])[0]
                for n in neighbors:
                    if n in peg_atoms and n not in visited:
                        visited.add(n)
                        queue.append(n)
            components.append(component)
    
    peg_groups = []
    
    for component in components:
        # Enforce >= 2 units rule (PEG unit -CH2CH2O- is 3 atoms, but user said continuous CCO >= 2)
        # CCOCCO is 6 atoms. So we check if component size >= 6.
        if len(component) < 6: 
            continue 
        
        # 3. Order atoms linearly to slice them correctly
        # Find end point (degree 1 within component graph)
        start_node = -1
        
        # Build local adjacency count
        local_adj = {a: [] for a in component}
        for a in component:
            neighbors = np.nonzero(mol_adj[a])[0]
            for n in neighbors:
                if n in component:
                    local_adj[a].append(n)
        
        # Find leaves (degree 1)
        leaves = [a for a, nbors in local_adj.items() if len(nbors) == 1]
        
        if not leaves:
            # Cycle or isolated ring? Pick arbitrary start if strictly PEG cyclic
            start_node = component[0]
        else:
            start_node = leaves[0]
            
        # Traverse
        ordered_atoms = []
        stack = [start_node]
        traversed_visited = set()
        
        while stack:
            curr = stack.pop()
            if curr not in traversed_visited:
                traversed_visited.add(curr)
                ordered_atoms.append(curr)
                # Add neighbors to stack
                for n in local_adj[curr]:
                    if n not in traversed_visited:
                        stack.append(n)
                        
        # 4. Slice into chunks of 4
        for i in range(0, len(ordered_atoms), 4):
             peg_groups.append(ordered_atoms[i:i+4])
    
    # ------------------ 核心修复点 ------------------
    # 强制将所有原子索引从 numpy.int64 转换为 Python int
    # 否则 RDKit 的 MolFragmentToSmiles 会报错
    final_groups = []
    for group in peg_groups:
        final_groups.append([int(x) for x in group])
    
    return final_groups


# --- Main Execution ---

smi = sys.argv[1]
mol_name = 'MOL'
mol = Chem.MolFromSmiles(smi)
mol_dict = Chem.MolFromSmiles(smi) 

# Coarse-grained mapping
matched_maps,matched_beads = get_smarts_matches(mol)
ring_atoms = get_ring_atoms(mol)

# --- Handle PEG input if present ---
# Check if is_peg argument is provided (8th argument)
is_peg = 0
if len(sys.argv) > 8:
    try:
        is_peg = int(sys.argv[8])
    except:
        is_peg = 0

if is_peg == 1:
    print("Detecting PEG chains...")
    peg_groups = get_peg_beads_groups(mol)
    if peg_groups:
        print(f"Found {len(peg_groups)} PEG beads from continuous segments.")
        # Add to matched_maps so they are treated as fixed blocks
        matched_maps.extend(peg_groups)
        # Extend matched_beads with empty strings so param_bead calculates types automatically
        matched_beads.extend([''] * len(peg_groups))

# Execute Mapping
A_cg, beads, ring_beads, path_matrix = mapping(mol, ring_atoms, matched_maps, 3, mol_dict)

non_ring = [b for b in range(len(beads)) if not any(b in ring for ring in ring_beads)]

# Parametrise beads
tuning = bool(int(sys.argv[4]))
bead_types,charges,all_smi,DG_data = get_types(beads,mol,ring_beads)

is_opt = sys.argv[5]
compoents_id = sys.argv[6]
split_id = sys.argv[7]

# Save Atom to Bead Mapping
atom_to_bead_type = {}
for bead_idx, (bead_atoms, bead_type) in enumerate(zip(beads, bead_types)):
    for atom in bead_atoms:
        atom_to_bead_type[atom] = (bead_idx, bead_type)

with open(f"{compoents_id}atom_to_bead_mapping{split_id}.txt", "w") as f:
    for atom in sorted(atom_to_bead_type.keys()):
        bead_idx, btype = atom_to_bead_type[atom]
        f.write(f"{atom:^8} | {bead_idx:^8} | {btype}\n")

# Geometry Generation and Output
if is_opt == '1':
    nconfs = 1
    Chem.AddHs(mol)
    AllChem.EmbedMultipleConfs(mol,numConfs=nconfs,randomSeed=random.randint(1,1000),useRandomCoords=True)
    AllChem.UFFOptimizeMoleculeConfs(mol)
    Chem.MolToPDBFile(mol, f'{compoents_id}poly{split_id}.pdb')
    coords0 = get_coords(mol,beads)
    pd.DataFrame(coords0).to_csv(f'{compoents_id}xyz{split_id}.csv',index=False)
    write_gro(mol_name,bead_types,coords0,sys.argv[2])
    write_itp(mol_name,bead_types,coords0,charges,all_smi,A_cg,sys.argv[3])
elif is_opt == '0' :
    print('计算构象')
    nconfs = 1
    Chem.AddHs(mol)
    AllChem.EmbedMultipleConfs(mol,numConfs=nconfs,randomSeed=random.randint(1,1000),useRandomCoords=True)
    Chem.MolToPDBFile(mol, f'{compoents_id}poly{split_id}.pdb')
    coords0 = get_coords(mol,beads)
    pd.DataFrame(coords0).to_csv(f'{compoents_id}xyz{split_id}.csv',index=False)
    write_gro(mol_name,bead_types,coords0,sys.argv[2])
    write_itp(mol_name,bead_types,coords0,charges,all_smi,A_cg,sys.argv[3])