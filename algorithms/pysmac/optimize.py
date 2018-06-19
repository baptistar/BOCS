import numpy as np

import time
import logging

from socket import timeout

from pysmac.smacrunner import SMACRunner
from pysmac.smacremote import SMACRemote


def check_param_dimensions(x0, xmin, xmax):
    assert(x0.shape == xmin.shape), "shape of x0 and xmin don't agree"
    assert(x0.shape == xmax.shape), "shape of x0 and xmax don't agree"
    return True

def format_params(x0, xmin, xmax, nptype):
    x0 = np.asarray(x0, dtype=nptype)
    xmin = np.asarray(xmin, dtype=nptype)
    xmax = np.asarray(xmax, dtype=nptype)
    return x0, xmin, xmax

def check_categorical_params(params):
    assert isinstance(params, dict), "Categorical parameters must be a dict of lists."
    for key, value in params.iteritems():
        assert isinstance(value, list), "Categorical parameters must be a dict of lists."

def fmin(objective,
         x0=[], xmin=[], xmax=[],
         x0_int=[], xmin_int=[], xmax_int=[],
         x_categorical={},
         custom_args={},
         max_evaluations=100, seed=1,
         cv_folds=None,
         update_status_every=500,
         smac_rf_num_trees=100,
         smac_rf_full_tree_bootstrap=True,
         smac_intensification_percentage=0.0
         ):
    """
        min_x f(x) s.t. xmin < x < xmax

        objective: The objective function that should be optimized.
                   Designed for objective functions that are:
                   costly to calculate + don't have a derivative available.
        x0: initial guess
        xmin: minimum values 
        xmax: maximum values
        x0_int: initial guess of integer params
        xmin_int: minimum values of integer params
        xmax_int: maximum values of integer params
        x_categorical: dictionary of categorical parameters
        custom_args: a dict of custom arguments to the objective function
        max_evaluations: the maximum number of evaluations to execute
        seed: the seed that SMAC is initialized with
        cv_folds: set if you want to use cross-validation. The objective function will get an new `cv_fold` argument.
        update_status_every: the number of num_evaluationss, between status updates

        Advanced
        --------
        smac_rf_num_trees: number of trees to create in random forest.
        smac_rf_full_tree_bootstrap: bootstrap all data points into trees.
        smac_intensification_percentage: percent of time to spend intensifying versus model learning.

        returns: best parameters found
    """
    x0, xmin, xmax = format_params(x0, xmin, xmax, np.float)
    check_param_dimensions(x0, xmin, xmax)

    x0_int, xmin_int, xmax_int = format_params(x0_int, xmin_int, xmax_int, np.int)
    check_param_dimensions(x0_int, xmin_int, xmax_int)

    if cv_folds is not None:
        assert cv_folds > 1, "cv_folds needs to be either None or greater than 1."

    #check_objective_function(objective, )

    check_categorical_params(x_categorical)

    smacremote = SMACRemote()

    smacrunner = SMACRunner(x0, xmin, xmax,
                            x0_int, xmin_int, xmax_int,
                            x_categorical,
                            smacremote.port, max_evaluations, seed,
                            cutoff_time=86400,
                            rf_num_trees=smac_rf_num_trees,
                            rf_full_tree_bootstrap=smac_rf_full_tree_bootstrap,
                            intensification_percentage=smac_intensification_percentage)
    current_fmin = None
    num_evaluations = 0

    try:
        while not smacrunner.is_finished():
            try:
                params = smacremote.next()
            except timeout:
                #Timeout, check if the runner is finished
                continue
            params = smacremote.get_next_parameters()
            fold = smacremote.get_next_fold()

            start = time.clock()
            assert all([param not in custom_args.keys() for param in params.keys()]), ("Naming collision between"
                                                                                       "parameters and custom arguments")
            function_args = {}
            function_args.update(params)
            function_args.update(custom_args)
            if cv_folds is not None:
                function_args["cv_fold"] = fold

	    print(function_args)
            performance = objective(**function_args)
            num_evaluations += 1

            assert performance is not None, ("objective function did not return "
                "a result for parameters %s" % str(function_args))
            assert np.isreal(performance), ("objective function did not return a number: "
                 + str(performance))

            if current_fmin is None or performance < current_fmin:
                current_fmin = performance
                fmin_changed = True
            else:
                fmin_changed = False

            if fmin_changed or (num_evaluations % update_status_every) == 0:
                print "Number of evaluations %d, current fmin: %f" % (num_evaluations, current_fmin)
            runtime = time.clock() - start

            smacremote.report_performance(performance, runtime)

    except KeyboardInterrupt:
        logging.warn("received keyboard interrupt ... aborting")
        smacrunner.stop()

    print "Number of evaluations %d, fmin: %f" % (num_evaluations, current_fmin)

    return smacrunner.get_best_parameters()


