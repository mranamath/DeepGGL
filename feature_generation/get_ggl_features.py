#!/usr/bin/env python

"""
Introduction:
    Get Bipartite Subgraph Descriptors

Author:
    Masud Rana (mrana10@kennesaw.edu)

Date last modified:
    Sep 9, 2025

"""

import sys
import pandas as pd
from itertools import product
import ntpath
import argparse
import time

from bipartite_subgraph_descriptors import *


class GeometricGraphLearningFeatures:

    df_kernels = pd.read_csv('../utils/kernels.csv')

    def __init__(self, args):
        """
        Parameters
        ----------
        kernel_index: int
            row index in utils/kernels.csv
        cutoff: float
            distance cutoff to define binding site
        path_to_csv: str
            full path to csv data consisting of PDBID and Binding Affinity (pK)


        Output
        -------
        feature csv file

        """
        self.kernel_index = args.kernel_index
        self.cutoff = args.cutoff
        self.path_to_csv = args.path_to_csv

        self.data_folder = args.data_folder
        self.feature_folder = args.feature_folder


    def get_features(self, parameters):

        df_pdbids = pd.read_csv(self.path_to_csv)
        pdbids = df_pdbids['PDBID'].tolist()
        pks = df_pdbids['pK'].tolist()

        Kernel = KernelFunction(kernel_type=parameters['type'],
                                kappa=parameters['power'], tau=parameters['tau'])

        bsd = BipartiteSubgraphDescriptors(Kernel=Kernel, cutoff=parameters['cutoff'])

        for index, _pdbid in enumerate(pdbids):
            print('PDBID: ', _pdbid)
            lig_file = f'{self.data_folder}/{_pdbid}/{_pdbid}_ligand.mol2'

            pro_file = f'{self.data_folder}/{_pdbid}/{_pdbid}_protein.pdb'

            ggl_score = bsd.get_subgraph_features(pro_file, lig_file)

            atom_pairs = ggl_score['ATOM_PAIR'].tolist()
            features = ggl_score.columns[1:].tolist()

            pairwise_features = [i[0]+'_'+i[1]
                                 for i in product(atom_pairs, features)]
            feature_values = ggl_score.drop(
                ['ATOM_PAIR'], axis=1).values.flatten()
            if index == 0:
                df_features = pd.DataFrame(columns=[pairwise_features])
            df_features.loc[index] = feature_values

        df_features.insert(0, 'PDBID', pdbids)
        df_features.insert(1, 'pK', pks)

        return df_features

    def main(self):

        parameters = {
            'type': self.df_kernels.loc[self.kernel_index, 'type'],
            'power': self.df_kernels.loc[self.kernel_index, 'power'],
            'tau': self.df_kernels.loc[self.kernel_index, 'tau'],
            'cutoff': self.cutoff
        }

        df_features = self.get_features(parameters)

        csv_file_name_only = ntpath.basename(self.path_to_csv).split('.')[0]

        output_file_name = f'{csv_file_name_only}_SYBYL_atomtype_ker{self.kernel_index}_cutoff{self.cutoff}.csv'

        df_features.to_csv(f'{self.feature_folder}/{output_file_name}', index=False, float_format='%.5f')


def get_args(args):

    parser = argparse.ArgumentParser(description="Get GGL Features")

    parser.add_argument('-k', '--kernel-index', help='Kernel Index (see kernels/kernels.csv)',
                        type=int)
    parser.add_argument('-c', '--cutoff', help='distance cutoff to define binding site',
                        type=float, default=12.0)
    parser.add_argument('-f', '--path_to_csv',
                        help='path to CSV file containing PDBIDs and pK values')
  
    parser.add_argument('-dd', '--data_folder', type=str,
    					help='path to data folder directory')
    parser.add_argument('-fd', '--feature_folder', type=str,
    					help='path to the directory where features will be saved')

    args = parser.parse_args()

    return args


def cli_main():
    args = get_args(sys.argv[1:])

    GGL_Features = GeometricGraphLearningFeatures(args)

    GGL_Features.main()


if __name__ == "__main__":

    t0 = time.time()

    cli_main()

    print('Done!')
    print('Elapsed time: ', time.time()-t0)
