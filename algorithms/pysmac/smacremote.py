import socket
import sys
import logging

from pysmac.smacparse import parse_smac_param_string, parse_smac_cv_fold

class SMACRemote(object):
    IP = "127.0.0.1"
    TCP_PORT = 5050
    #The size of a udp package
    #note: set in SMAC using --ipc-udp-packetsize
    TIMEOUT = 3

    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(SMACRemote.TIMEOUT)

        self.conn = None

        #try to find a free port:
        for self.port in range(SMACRemote.TCP_PORT, SMACRemote.TCP_PORT+1000):
            try:
                self.sock.bind((SMACRemote.IP, self.port))
                self.sock.listen(1)
                break
            except:
                pass
        logging.debug("Communicating on port: %d", self.port)

    def __del__(self):
        if self.conn is not None:
            self.conn.close()
        self.sock.close()

    def _connect(self):
        self.sock.settimeout(SMACRemote.TIMEOUT)
        self.conn, addr = self.sock.accept()
        self.conn.settimeout(SMACRemote.TIMEOUT)

    def _disconnect(self):
        self.conn.close()
        self.conn = None

    def send(self, data):
        assert self.conn is not None

        logging.debug("> " + str(data))
        self.conn.sendall(data)
        
        self._disconnect()

    def receive(self):
        logging.debug("Waiting for a message from SMAC.")
        self._connect()

        #data = self._conn.recv(4096) # buffer size is 4096 bytes
        fconn = self.conn.makefile('r') 
        data = fconn.readline()

        logging.debug("< " + str(data))
        return data

    def next(self):
        """
            Fetch the next input from SMAC.
        """
        self.next_param_string = self.receive()

    def get_next_parameters(self):
        """
            Extract the next set of parameters.

            returns: an array of parameters.
        """
        return parse_smac_param_string(self.next_param_string)

    def get_next_fold(self):
        """
            Extract the next fold.

            returns: the index of the next fold
        """
        return parse_smac_cv_fold(self.next_param_string)
    
    def report_performance(self, performance, runtime):
        """
            Report the performance for the current run of the algorithm.

            performance: performance of the algorithm.
            runtime: the runtime of the call in seconds.
        """
        #format:
        #Result for ParamILS: <solved>, <runtime>, <runlength>, <quality>, <seed>, <additional rundata>
        #e.g. Result for ParamILS: UNSAT, 6,0,0,4
        
        #runtime must be strictly positive:
        runtime = min(0, runtime)
        data = "Result for ParamILS: SAT, %f, 0, %f, 4" % (runtime, performance)
        logging.debug("Response to SMAC:")
        logging.debug(str(data))

        self.send(data)
