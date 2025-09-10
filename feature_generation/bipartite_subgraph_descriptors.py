#!/usr/bin/env python

"""
Introduction:
    Bipartite Subgraph Descriptors

Author:
    Masud Rana (mrana10@kennesaw.edu)

Date last modified:
    Sep 9, 2025

"""

import numpy as np
import pandas as pd
from os import listdir
from scipy.spatial.distance import cdist
from itertools import product
from biopandas.pdb import PandasPdb
from biopandas.mol2 import PandasMol2


class KernelFunction:

    def __init__(self, kernel_type='exponential_kernel',
                 kappa=2.0, tau=1.0):
        self.kernel_type = kernel_type
        self.kappa = kappa
        self.tau = tau

        self.kernel_function = self.build_kernel_function(kernel_type)

    def build_kernel_function(self, kernel_type):
        if kernel_type[0] in ['E', 'e']:
            return self.exponential_kernel
        elif kernel_type[0] in ['L', 'l']:
            return self.lorentz_kernel

    def exponential_kernel(self, d, vdw_radii):
        eta = self.tau*vdw_radii

        return np.exp(-(d/eta)**self.kappa)

    def lorentz_kernel(self, d, vdw_radii):
        eta = self.tau*vdw_radii

        return 1/(1+(d/eta)**self.kappa)


class BipartiteSubgraphDescriptors:

    protein_atom_types_df = pd.read_csv(
        '../utils/protein_atom_types.csv')

    ligand_atom_types_df = pd.read_csv(
        '../utils/ligand_SYBYL_atom_types_revised.csv')

    protein_atom_types = protein_atom_types_df['AtomType'].tolist()
    protein_atom_radii = protein_atom_types_df['Radius'].tolist()

    ligand_atom_types = ligand_atom_types_df['AtomType'].tolist()
    ligand_atom_radii = ligand_atom_types_df['Radius'].tolist()

    protein_ligand_atom_types = [
        i[0]+"-"+i[1] for i in product(protein_atom_types, ligand_atom_types)]

    def __init__(self, Kernel, cutoff):

        self.Kernel = Kernel
        self.cutoff = cutoff

        self.pairwise_atom_type_radii = self.get_pairwise_atom_type_radii()

    def get_pairwise_atom_type_radii(self):

        protein_atom_radii_dict = {a: r for (a, r) in zip(
            self.protein_atom_types, self.protein_atom_radii)}

        ligand_atom_radii_dict = {a: r for (a, r) in zip(
            self.ligand_atom_types, self.ligand_atom_radii)}

        pairwise_atom_type_radii = {i[0]+"-"+i[1]: protein_atom_radii_dict[i[0]] +
                                    ligand_atom_radii_dict[i[1]] for i in product(self.protein_atom_types, self.ligand_atom_types)}

        return pairwise_atom_type_radii

    def mol2_to_df(self, mol2_file):
        df_mol2_all = PandasMol2().read_mol2(mol2_file).df
        df_mol2 = df_mol2_all[df_mol2_all['atom_type'].isin(self.ligand_atom_types)]
        df = pd.DataFrame(data={'ATOM_INDEX': df_mol2['atom_id'],
                                'ATOM_ELEMENT': df_mol2['atom_type'],
                                'X': df_mol2['x'],
                                'Y': df_mol2['y'],
                                'Z': df_mol2['z']})

        if len(set(df["ATOM_ELEMENT"]) - set(self.ligand_atom_types)) > 0:
            print(
                "WARNING: Ligand contains unsupported atom types. Only supported atom-type pairs are counted.")
        return(df)

    def pdb_to_df(self, pdb_file):
        ppdb = PandasPdb()
        ppdb.read_pdb(pdb_file)
        ppdb_all_df = ppdb.df['ATOM']
        ppdb_df = ppdb_all_df[ppdb_all_df['atom_name'].isin(
            self.protein_atom_types)]
        atom_index = ppdb_df['atom_number']
        atom_element = ppdb_df['atom_name']
        x, y, z = ppdb_df['x_coord'], ppdb_df['y_coord'], ppdb_df['z_coord']
        df = pd.DataFrame.from_dict({'ATOM_INDEX': atom_index, 'ATOM_ELEMENT': atom_element,
                                     'X': x, 'Y': y, 'Z': z})

        return df

    def get_mwcg_rigidity(self, protein_file, ligand_file):
       
        protein = self.pdb_to_df(protein_file)
        ligand = self.mol2_to_df(ligand_file)

        # select protein atoms in a cubic with a size of cutoff from ligand
        for i in ["X", "Y", "Z"]:
            protein = protein[protein[i] < float(ligand[i].max())+self.cutoff]
            protein = protein[protein[i] > float(ligand[i].min())-self.cutoff]

        atom_pairs = list(
            product(protein["ATOM_ELEMENT"], ligand["ATOM_ELEMENT"]))
        atom_pairs = [x[0]+"-"+x[1] for x in atom_pairs]
        pairwise_radii = [self.pairwise_atom_type_radii[x]
                          for x in atom_pairs]
        pairwise_radii = np.asarray(pairwise_radii)

        pairwise_mwcg = pd.DataFrame(atom_pairs, columns=["ATOM_PAIR"])
        distances = cdist(protein[["X", "Y", "Z"]],
                          ligand[["X", "Y", "Z"]], metric="euclidean")
        pairwise_radii = pairwise_radii.reshape(
            distances.shape[0], distances.shape[1])
        mwcg_distances = self.Kernel.kernel_function(distances, pairwise_radii)

        distances = distances.ravel()
        mwcg_distances = mwcg_distances.ravel()
        mwcg_distances = pd.DataFrame(
            data={"DISTANCE": distances, "MWCG_DISTANCE": mwcg_distances})
        pairwise_mwcg = pd.concat([pairwise_mwcg, mwcg_distances], axis=1)
        pairwise_mwcg = pairwise_mwcg[pairwise_mwcg["DISTANCE"] <= self.cutoff].reset_index(
            drop=True)

        return pairwise_mwcg

    def get_subgraph_features(self, protein_file, ligand_file):
        features = ['COUNTS','SUM', 'MEAN', 'MEDIAN', 'VAR', 'STD', 'MIN', 'MAX']
        pairwise_mwcg = self.get_mwcg_rigidity(protein_file, ligand_file)
        mwcg_temp_grouped = pairwise_mwcg.groupby('ATOM_PAIR')
        mwcg_temp_grouped.agg(['sum', 'mean', 'median','var','std','min','max'])
        mwcg_temp = mwcg_temp_grouped.size().to_frame(name='COUNTS')
        mwcg_temp = (mwcg_temp
                     .join(mwcg_temp_grouped.agg({'MWCG_DISTANCE': 'sum'}).rename(columns={'MWCG_DISTANCE': 'SUM'}))
                     .join(mwcg_temp_grouped.agg({'MWCG_DISTANCE': 'mean'}).rename(columns={'MWCG_DISTANCE': 'MEAN'}))
                     .join(mwcg_temp_grouped.agg({'MWCG_DISTANCE': 'median'}).rename(columns={'MWCG_DISTANCE': 'MEDIAN'}))
                     .join(mwcg_temp_grouped.agg({'MWCG_DISTANCE': 'var'}).rename(columns={'MWCG_DISTANCE': 'VAR'}))
                     .join(mwcg_temp_grouped.agg({'MWCG_DISTANCE': 'std'}).rename(columns={'MWCG_DISTANCE': 'STD'}))
                     .join(mwcg_temp_grouped.agg({'MWCG_DISTANCE': 'min'}).rename(columns={'MWCG_DISTANCE': 'MIN'}))
                     .join(mwcg_temp_grouped.agg({'MWCG_DISTANCE': 'max'}).rename(columns={'MWCG_DISTANCE': 'MAX'}))
                     )
        mwcg_columns = {'ATOM_PAIR': self.protein_ligand_atom_types}
        for _f in features:
            mwcg_columns[_f] = np.zeros(len(self.protein_ligand_atom_types))
        subgraph_features = pd.DataFrame(data=mwcg_columns)
        subgraph_features = subgraph_features.set_index('ATOM_PAIR').add(
            mwcg_temp, fill_value=0).reindex(self.protein_ligand_atom_types).reset_index()

        return subgraph_features