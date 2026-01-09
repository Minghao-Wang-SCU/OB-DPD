import pandas as pd
import numpy as np
import random
random.seed(42)
np.random.seed(42)
import math
import os
import tempfile
from tqdm import tqdm
import warnings
import shap
warnings.filterwarnings("ignore")
from collections import Counter
from Bio.Seq import Seq
from Bio.SeqUtils.ProtParam import ProteinAnalysis

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdMolDescriptors import CalcLabuteASA
import rdkit.Chem.GraphDescriptors as GraphDescriptors

import optuna
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from joblib import dump,load
import statsmodels.api as sm

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches
import seaborn as sns



def get_polar_heavy_atoms(mol):
    """
    统计分子中极性重原子的数量。
    
    参数:
        mol (rdkit.Chem.Mol): RDKit 分子对象
        
    返回:
        int: 极性重原子数量
    """
    polar_atoms = {7: True, 8: True, 9: True, 15: True, 16: True, 17: True, 35: True, 53: True}
    # 7: N, 8: O, 9: F, 15: P, 16: S, 17: Cl, 35: Br, 53: I
    
    polar_count = 0
    
    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        if atomic_num in polar_atoms:
            polar_count += 1
    
    return polar_count

def get_nonpolar_heavy_atoms(mol):
    """
    统计分子中非极性重原子的数量。
    
    参数:
        mol (rdkit.Chem.Mol): RDKit 分子对象
        
    返回:
        int: 非极性重原子数量
    """
    nonpolar_atoms = {6: True}
    # 6: C
    
    nonpolar_count = 0
    
    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        if atomic_num in nonpolar_atoms:
            nonpolar_count += 1
    
    return nonpolar_count

def get_hydrogen_atoms(mol):
    """
    统计分子中氢原子的数量（包括显式和隐式氢原子）。
    
    参数:
        mol (rdkit.Chem.Mol): RDKit 分子对象
        
    返回:
        int: 氢原子数量
    """
    hydrogen_count = 0
    
    # 统计显式氢原子
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:  # 氢原子
            hydrogen_count += 1
    
    # 统计隐式氢原子
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 1:  # 重原子
            num_hydrogens = atom.GetTotalNumHs()
            hydrogen_count += num_hydrogens
    
    return hydrogen_count

def create_features(mol, feature_names):
    feature_list = []
    feature_names_list = []
    if feature_names == 'all':
        all_descriptors = {name: func(mol) for name, func in Descriptors.descList}
        for name, value in all_descriptors.items():
            feature_names_list.append(name)
            feature_list.append(value)
        feature_names_list.append('polar_heavy_atoms')
        feature_list.append(get_polar_heavy_atoms(mol))
        feature_names_list.append('nonpolar_heavy_atoms')
        feature_list.append(get_nonpolar_heavy_atoms(mol))
        feature_names_list.append('hydrogen_atoms')
        feature_list.append(get_hydrogen_atoms(mol))
        
    else:
        for f in feature_names:
            feature_names_list.append(f)
            if 'HeavyAtomCount' == f:
                feature = Descriptors.HeavyAtomCount(mol)
            elif 'HeavyAtomMolWt' == f:
                feature = Descriptors.HeavyAtomMolWt(mol)
            elif 'MolLogP' == f:
                feature = Descriptors.MolLogP(mol)
            elif 'TPSA' == f:
                feature = Descriptors.TPSA(mol)
            elif 'BCUT2D_LOGPHI' == f:
                feature = Descriptors.BCUT2D_LOGPHI(mol)
            elif 'BCUT2D_LOGPLOW' == f:
                feature = Descriptors.BCUT2D_LOGPLOW(mol)
            elif 'BalabanJ' == f:
                feature = GraphDescriptors.BalabanJ(mol)
            elif 'BertzCT' == f:
                feature = GraphDescriptors.BertzCT(mol)
            elif 'Chi0' == f:
                feature = Descriptors.Chi0(mol)
            elif 'Chi1' == f:
                feature = Descriptors.Chi1(mol)
            elif 'Kappa1' == f:
                feature = Descriptors.Kappa1(mol)
            elif 'Kappa2' == f:
                feature = Descriptors.Kappa2(mol)
            elif 'Kappa3' == f:
                feature = Descriptors.Kappa3(mol)
            elif 'LabuteASA' == f:
                feature = CalcLabuteASA(mol)
            else:
                print(f"不支持的特征名: {f}")
                feature = None
            feature_list.append(feature)
        
    return feature_list,feature_names_list


def Pred_Solubility_Parameter(pred_smiles_list,is_peg_list):
    """
    #输入pred_smiles_list
    返回list 包含每个对应的溶解度参数
    """
    print('-'*40+'Pred_Solubility_Parameter '+'-'*40)
    peg_smiles = ['CCOC','COCC','COC','CCO','OCC']
    loaded_model = load('finally_xgb_regression.joblib')
    pred_features_mat = []
    for i in range(len(pred_smiles_list)):
        mol = Chem.MolFromSmiles(pred_smiles_list[i])
        features,feature_names = create_features(mol, 'all')
        pred_features_mat.append(features)
    pred_data = pd.DataFrame(pred_features_mat,columns=feature_names)
    data_with_features = pd.concat([pd.Series(pred_smiles_list),pred_data],axis=1)
    data_with_features.to_excel(r'pred_data_with_features.xlsx',index=False)
    pred_X = data_with_features.iloc[:,1:]
    scaler = load('scaler_params.joblib')
    pred_X = scaler.transform(pred_X)
    top_10_feature_indices = pd.read_csv(r'top_10_feature_indices.csv').values.tolist()
    y_pred = loaded_model.predict(pred_X[:, top_10_feature_indices].reshape(-1,len(top_10_feature_indices)))
    print('is_peg_list',is_peg_list )
    for i,flag in enumerate(is_peg_list):
        if flag == 1 :
            print('pred_smiles_list[i]',pred_smiles_list[i])
            if pred_smiles_list[i] in peg_smiles:
                y_pred[i] = max(y_pred)
            if 'CCOCCOCCO' in pred_smiles_list[i]:
                y_pred[i] = max(y_pred)
    print('-'*40+'Pred_Solubility_Parameter '+'-'*40)
    return y_pred