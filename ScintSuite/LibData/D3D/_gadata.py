"""
Generic class to read GA data

Extracted from D3D knowledge hub
"""

import MDSplus 
import numpy
import time
import sys
import logging
logger = logging.getLogger('ScintSuite.data')
class gadata:
    """GA Data Obj"""
    def __init__(self,signal,shot,tree=None,connection=None,nomds=False):
		
        # Save object values
        self.signal = signal
        self.shot = shot
        self.zdata = -1
        self.xdata = -1
        self.ydata = -1
        self.zunits = ''
        self.xunits = ''
        self.yunits	= ''
        self.rank = -1
        self.connection	= connection

        ## Retrieve Data 
        t0 =  time.time()
        found = 0

		# Create the MDSplus connection (thin) if not passed in  
        if self.connection is None:
            self.connection = MDSplus.Connection('atlas.gat.com')
            logger.debug('Connection to atlas.gat.com successful')

        # Retrieve data from MDSplus (thin)
        if nomds == False:
            try:     
                if tree != None:
                    tag = self.signal
                    fstree 	= tree 
                    found   = 1
                else:
                    tag = self.connection.get('findsig("'+self.signal+'",_fstree)').value
                    fstree = self.connection.get('_fstree').value 
                logger.debug('Opening tree %s for shot %d'%(fstree,shot))
                self.connection.openTree(fstree,shot)
                self.zdata = self.connection.get('_s = '+tag).data()
                self.zunits = self.connection.get('units_of(_s)').data()  
                self.rank = len(self.zdata.shape)	
                self.xdata = self.connection.get('dim_of(_s)').data()
                self.xunits = self.connection.get('units_of(dim_of(_s))').data()
                if self.xunits == '' or self.xunits == ' ': 
                    self.xunits = self.connection.get('units(dim_of(_s))').data()
                if self.rank > 1:
                    self.ydata = self.connection.get('dim_of(_s,1)').data()
                    self.yunits = self.connection.get('units_of(dim_of(_s,1))').data()
                    if self.yunits == '' or self.yunits == ' ':
                        self.yunits = self.connection.get('units(dim_of(_s,1))').data()

                found = 1	
                logger.debug('Found signal %s in MDSplus tree %s'%(self.signal,fstree))
                # MDSplus seems to return 2-D arrays transposed.  Change them back.
                if self.rank == 2: 
                    logger.debug('Transposing zdata')
                    self.zdata = self.zdata.T
                    if len(self.ydata.shape) == 2: 
                        logger.debug('Transposing ydata')
                        self.ydata = self.ydata.T
                    if len(self.xdata.shape) == 2: 
                        logger.debug('Transposing xdata')
                        self.xdata = self.xdata.T

            except Exception as e:
                logger.debug('Get exception: %s'%(e,))
                logger.debug('Signal not in MDSplus: %s' % (signal,))
                pass


            # Retrieve data from PTDATA
            if found == 0:
                self.zdata = self.connection.get('_s = ptdata2("'+signal+'",'+str(shot)+')')
                if len(self.zdata) != 1:
                    self.xdata = self.connection.get('dim_of(_s)')
                    self.rank = 1
                    found = 1
                    logger.debug('Found signal %s in PTDATA'%(self.signal))
                else:
                    logger.debug('Signal not in PTDATA: %s' % (signal,))

            # Retrieve data from Pseudo-pointname 
            if found == 0:
                self.zdata = self.connection.get('_s = pseudo("'+signal+'",'+str(shot)+')')
                if len(self.zdata) != 1:
                    self.xdata = self.connection.get('dim_of(_s)')
                    self.rank = 1
                    found = 1
                    logger.debug('Found signal %s as pseudo-pointname'%(self.signal))

            if found == 0: 
                logger.error('No such signal: %s'%(signal,))
                #print "   No such signal: %s" % (signal,)
                return
        logger.debug('GADATA Retrieval Time : %f'%(time.time() - t0))
                #print '   GADATA Retrieval Time : ',time.time() - t0
