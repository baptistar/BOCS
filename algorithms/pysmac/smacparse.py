import numpy as np
import ast
from itertools import izip


def param_pairs_to_float_np(param_pairs):
    x = np.zeros(len(param_pairs), dtype=np.float)
    for param_name, param_value in param_pairs:
        #strip "-x" from the name
        param_index = int(param_name[len("x"):])
        param_value = float(param_value.strip("'"))
        x[param_index] = param_value
    return x
 

def param_pairs_to_int_np(param_pairs):
    x = np.zeros(len(param_pairs), dtype=np.int)
    for param_name, param_value in param_pairs:
        #strip "-x_int" from the name
        param_index = int(param_name[len("x_int"):])
        param_value = int(param_value.strip("'"))
        x[param_index] = param_value
    return x


def value_to_literal(value):
    """ 
        Tries to convert a value to either a 
        float, int or boolean.
    """
    try:
        return ast.literal_eval(value)
    except:
        return value


def param_pairs_to_dict(param_pairs):
    x = {}
    for param_name, param_value in param_pairs:
        param_name = param_name[len("-x_categorical"):]
        param_value = param_value.strip("'")
        x[param_name] = value_to_literal(param_value)
    return x


def param_pairs_to_params(param_pairs):
    params = {}
    #let's sort the parameters by their type:
    for name, val in param_pairs:
        if name.startswith("x_int"):
            if not "x_int" in params:
                params["x_int"] = []
            params["x_int"].append((name, val))
        elif name.startswith("x_categorical"):
            if not "x_categorical" in params:
                params["x_categorical"] = []
            params["x_categorical"].append((name, val))
        elif name.startswith("x"):
            if not "x" in params:
                params["x"] = []
            params["x"].append((name, val))
        else:
            assert False, "unkown parameter type %s" % name

    if "x" in params:
        params["x"] = param_pairs_to_float_np(params["x"])
    if "x_int" in params:
        params["x_int"] = param_pairs_to_int_np(params["x_int"])
    if "x_categorical" in params:
        params["x_categorical"] = param_pairs_to_dict(params["x_categorical"])
    return params

 
def parse_smac_param_string(param_string):
    """
        SMAC format:
            cvfold-0 0 18000.0 2147483647 4 -x0 '6.9846789681200185' -x1 '13.43140264469383'
    """
    param_pieces = param_string.strip().split()[5:]
    param_pairs = [pair for pair in izip(*[iter(param_pieces)]*2)]
    #strip the '-'
    param_pairs = [(name[1:], value) for name, value in param_pairs]
    params = param_pairs_to_params(param_pairs)

    return params


def parse_smac_cv_fold(param_string):
    """
        SMAC format:
            cvfold-0 0 18000.0 2147483647 4 -x0 '6.9846789681200185' -x1 '13.43140264469383'
    """
    fold_descriptor = param_string.strip().split()[0]
    if not "cvfold-" in fold_descriptor:
        logging.warn("failed parsing fold")
        return 0
    return int(fold_descriptor.split("-")[1])


def parse_smac_trajectory_string(param_string):
    """
        SMAC format:
          12.95960999999998, 0.542677, 2.103, 100, 2.9599830000000003,  x0='3.2343299868854594', x1='1.8820288649037564'

        returns: (x, fval)
    """
    columns = param_string.strip().split(",")

    fval = float(columns[1])

    params_raw = columns[5:]
    param_pairs = [param.strip().split("=") for param in params_raw]

    output_params = param_pairs_to_params(param_pairs)

    return (output_params, fval)

