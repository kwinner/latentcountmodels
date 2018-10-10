import numpy as np
import forward as fwd
import os
import pandas as pd
import re


def load_results(result_dir):
    result_filename = os.path.join(result_dir, 'results.df')

    results = pd.read_table(result_filename, sep=' ')

    return(results)

# def load_params(result_dir):
#     params_filename = os.path.join(result_dir, 'params.txt')
#
#     with open(params_filename, 'r') as params_file:
#         params_lines = params_file.readlines()
#
#         for line in params_lines:
#             line_split = [term.strip() for term in line.strip().split(':')]


def str2mat(str):
    # remove leading/trailing brackets
    str = re.search('(?=\\[*)[^\\[].*[^\\]](?=\\]*)', str).group(0)

    if '];[' in str: # 2D array
        # convert to numeric list
        mat = np.array([[float(i) for i in row.split(',')] for row in str.split('];[')])
    else:
        mat = np.array([float(i) for i in str.split(',')])

    return(mat)