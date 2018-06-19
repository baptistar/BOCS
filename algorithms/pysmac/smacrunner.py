import tempfile
import shutil
import os
import sys
import time
import logging
import signal
from subprocess import Popen
from pkg_resources import resource_filename

from pysmac.smacparse import parse_smac_trajectory_string

import numpy as np

def check_java_version():
    import re
    from subprocess import STDOUT, check_output
    out = check_output(["java", "-version"], stderr=STDOUT).split("\n")
    if len(out) < 1:
        print "Failed checking Java version. Make sure Java version 7 or greater is installed."
        return False
    m = re.match('java version "\d+.(\d+)..*', out[0])
    if m is None or len(m.groups()) < 1:
        print "Failed checking Java version. Make sure Java version 7 or greater is installed."
        return False
    java_version = int(m.group(1))
    if java_version < 7:
        error_msg = "Found Java version %d, but Java version 7 or greater is required." % java_version
 
        raise RuntimeError(error_msg)
check_java_version()


class SMACRunner(object):
    """
        Interface to the SMAC library:
        see: http://www.cs.ubc.ca/labs/beta/Projects/SMAC/
    """

    SMAC_VERSION = "smac-v2.08.00-development-723"

    def __init__(self,
                 x0, xmin, xmax,
                 x0_int, xmin_int, xmax_int,
                 x_categorical,
                 port, max_evaluations, seed,
                 cutoff_time,
                 rf_num_trees,
                 rf_full_tree_bootstrap,
                 intensification_percentage
                 ):
        """
            Start up SMAC.

            x0: initial guess
            xmin: minimum values
            xmax: maximum values 
            x0_int: initial guess of integer parameters
            xmin_int: minimum values of ingeter parameters
            xmax_int: minimum values of ingeter parameters
            x_categorical: categorical parameters
            port: the port to communicate with SMACRemote
            max_evaluations: the maximum number of evaluations

            cutoff_time: CPU time before cut off
            rf_num_trees: number of trees to create in random forest.
            rf_full_tree_bootstrap: bootstrap all data points into trees.
            intensification_percentage: percent of time to spend intensifying versus model learning.
        """
        self.x0 = x0
        self.xmin = xmin
        self.xmax = xmax

        self.x0_int = x0_int
        self.xmin_int = xmin_int
        self.xmax_int = xmax_int

        self.x_categorical = x_categorical

        self.port = port
        self.max_evaluations = max_evaluations
        self.seed = seed
        self.cv_folds = None

        self.cutoff_time = cutoff_time
        self.rf_num_trees = rf_num_trees
        self.rf_full_tree_bootstrap = rf_full_tree_bootstrap
        self.intensification_percentage = intensification_percentage

        self._create_working_dir()
        self._generate_scenario_file()
        self._generate_instance_file()
        self._generate_parameter_file()
        self._start_smac()

    def __del__(self):
        #print "Shutting down SMACRunner."
        if hasattr(self, "_working_dir"):
            shutil.rmtree(self.working_dir)
        if hasattr(self, "_smac_process"):
            self.smac_process.poll()
            if self.smac_process.returncode == None:
                self.smac_process.kill()

    def is_finished(self):
        """
            Has the SMAC run finished?
        """
        self.smac_process.poll()
        if self.smac_process.returncode is None:
            return False
        else:
            return True

    def get_best_parameters(self):
        """
            Get the best parameters that were found along with the according function value.

            returns: (x, f(x))
        """
        with open(os.path.join(self.result_dir, "traj-run-1.txt"), "r") as traj_file:
            best_config_str = traj_file.readlines()[-1].strip()
            return parse_smac_trajectory_string(best_config_str)
        return (np.asarray([]), 0)

    def _create_working_dir(self):
        self.working_dir = tempfile.mkdtemp()
        self.exec_dir = os.path.join(self.working_dir, "exec")
        self.out_dir = os.path.join(self.working_dir, "out")
        os.mkdir(self.exec_dir)
        os.mkdir(self.out_dir)
        #also see the rungroup parameter
        self.result_dir = os.path.join(self.out_dir, "result")

    def _generate_scenario_file(self):
        """
            Generate a scenario file in order to run SMAC.
        """
        parameters = {'working_dir': self.working_dir,
                      'exec_dir': self.exec_dir,
                      'out_dir': self.out_dir,
                      'cutoff_time': self.cutoff_time}
        fdata = """
algo = echo 0
execdir = %(exec_dir)s
outdir = %(out_dir)s
deterministic = 1
rungroup = result
run_obj = quality
overall_obj = mean
cutoff_time = %(cutoff_time)d
cutoff_length = max
validation = false
paramfile = %(working_dir)s/params.pcs
instance_file = %(working_dir)s/instances.txt
""" % parameters
        self.scenario_file_name = os.path.join(self.working_dir, "smac-scenario.txt")
        with open(self.scenario_file_name, "w")  as self.scenario_file:
            self.scenario_file.write(fdata)

    def _generate_instance_file(self):
        instance_file_name = os.path.join(self.working_dir, "instances.txt")
        with open(instance_file_name, "w") as instance_file:
            if self.cv_folds is None:
                instance_file.write("cvfold-0")
            else:
                for i in range(0, self.cv_folds):
                    instance_file.write("cvfold-%d\n" % i)
            
    def _generate_parameter_file(self):
        param_definitions = []
        for i, (min_val, max_val, default_val) in enumerate(zip(self.xmin, self.xmax, self.x0)):
            param_definitions.append("x%d [%f, %f] [%f]" % (i, min_val, max_val, default_val))
        for i, (min_val, max_val, default_val) in enumerate(zip(self.xmin_int, self.xmax_int, self.x0_int)):
            param_definitions.append("x_int%d [%d, %d] [%d]i" % (i, min_val, max_val, default_val))
        for name, values in self.x_categorical.iteritems():
            str_values = map(str, values)
            param_definitions.append("x_categorical_%s {%s} [%s]" % (name,
                ", ".join(str_values),
                str_values[0]))
        param_file_name = os.path.join(self.working_dir, "params.pcs")
        with open(param_file_name, "w")  as param_file:
            param_file.write("\n".join(param_definitions))

    def _smac_classpath(self):
        smac_folder = resource_filename("pysmac", 'smac/%s' % SMACRunner.SMAC_VERSION)
        smac_conf_folder = os.path.join(smac_folder, "conf")
        smac_patches_folder = os.path.join(smac_folder, "patches")
        smac_lib_folder = os.path.join(smac_folder, "lib")

        logging.debug("SMAC lib folder: %s", smac_folder)

        classpath = [fname for fname in os.listdir(smac_lib_folder) if fname.endswith(".jar")]
        classpath = [os.path.join(smac_lib_folder, fname) for fname in classpath]
        classpath = [os.path.abspath(fname) for fname in classpath]
        classpath.append(os.path.abspath(smac_conf_folder))
        classpath.append(os.path.abspath(smac_patches_folder))

        logging.debug("SMAC classpath: %s", ":".join(classpath))

        return classpath

    def _start_smac(self):
        """
            Start SMAC in IPC mode. SMAC will wait for udp messages to be sent.
        """

        cmds = ["java",
                "-Xmx1024m",
                "-cp",
                ":".join(self._smac_classpath()),
                "ca.ubc.cs.beta.smac.executors.SMACExecutor",
                "--scenario-file", self.scenario_file_name,
                "--num-run","1",
                "--totalNumRunsLimit", str(self.max_evaluations),
                "--tae", "IPC",
                "--ipc-mechanism", "TCP",
                "--ipc-remote-port", str(self.port),
                "--seed", str(self.seed),
                "--rf-num-trees", str(self.rf_num_trees),
                "--rf-full-tree-bootstrap", str(self.rf_full_tree_bootstrap),
                "--intensification-percentage", str(self.intensification_percentage)
                ]
        with open(os.devnull, "w") as fnull:
            #logging.debug(" ".join(cmds))
            if logging.getLogger().level <= logging.DEBUG:
                self.smac_process = Popen(cmds, stdout = sys.stdout, stderr = sys.stdout)
            else:
                self.smac_process = Popen(cmds, stdout = fnull, stderr = fnull)

    def stop(self):
        if not self.is_finished():
            self.smac_process.send_signal(signal.SIGINT)
            self.smac_process.wait()
