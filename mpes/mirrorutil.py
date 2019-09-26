#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: L. Rettig
"""

import os
import shutil
import dask as d
from dask.diagnostics import ProgressBar

class CopyTool(object):
    """ File collecting and sorting class.
    """

    def __init__(self, source='/', dest='/', ntasks=None, **kwds):

        self.source = source
        self.dest = dest
        self.safetyMargin = kwds.pop('safetyMargin', 1 * 2**30) # Default 500 GB safety margin
        self.pbenv = kwds.pop('pbenv', 'classic')
        
        if (ntasks is None) or (ntasks < 0):
            # Default to 25 concurrent copy tasks
            self.ntasks = 25
        else:
            self.ntasks = int(ntasks)
        

    def copy(self, sdir, forceCopy=False, scheduler='threads', **compute_kwds):

        numFiles = countFiles(sdir)

        if numFiles > 0:

            ddir = getTargetDir(sdir, self.source, self.dest)

            makedirs(ddir)

            numCopied = 0

            for path, dirs, filenames in os.walk(sdir):
                # Check space left
                size = 0
                for sfile in filenames:
                    if (not os.path.exists(os.path.join(path.replace(sdir, ddir), sfile))):
                        size += os.path.getsize(os.path.join(sdir,sfile))
                if (size == 0 and not forceCopy):
                    # nothing to copy, just return directory
                    return ddir
                else:
                    total, used, free = shutil.disk_usage(ddir)
                    if (size > free - self.safetyMargin):
                        print("Target disk full, only " + str(free/2**30) + " GB free, but " + str(size/2**30) + " GB needed!")
                        return
                    for directory in dirs:
                        destDir = path.replace(sdir,ddir)
                        makedirs(os.path.join(destDir, directory))

                    copyTasks = [] # Core-level jobs
                    for sfile in filenames:

                            srcFile = os.path.join(path, sfile)
                            destFile = os.path.join(path.replace(sdir, ddir), sfile)
                            if (not os.path.exists(destFile) or forceCopy):
                                copyTasks.append(d.delayed(shutil.copy2)(srcFile, destFile))
                        
                    print("Copy Files...")
                    with ProgressBar():
                        d.compute(*copyTasks, scheduler=scheduler, num_workers=self.ntasks, **compute_kwds)
                    print("Copy finished!")

                return ddir

    def size(self, sdir):
        for path, dirs, filenames in os.walk(sdir):
            # Check space left
            size = 0
            for sfile in filenames:
                size += os.path.getsize(os.path.join(sdir,sfile))
            return size
                
    def cleanUpOldestScan(self, remove = None, force = False):
        
        # get list of all Scan directories (leaf directories)
        scan_dirs = list()
        for root,dirs,files in os.walk(self.dest):
            if not dirs:
                scan_dirs.append(root)
                
        #print(scan_dirs)
        scan_dirs = sorted(scan_dirs, key=os.path.getctime)
        #print(scan_dirs)
        oldestScan = None
        for scan in scan_dirs:
            size = 0
            for path, dirs, filenames in os.walk(scan):
                for sfile in filenames:
                    size += os.path.getsize(os.path.join(scan,sfile))
            if size > 0:
                oldestScan = scan
                break
        if oldestScan == None:
            print("No scan with data found to remove!")
            return
        
        print("I would delete the scan, freeing " + str(size/2**30) + " GB space:")
        print(oldestScan)    
        if (remove == oldestScan or force):
            shutil.rmtree(oldestScan)
            print ("Removed sucessfully!")
        else:
            print("To proceed, please call:")
            print("cleanUpOldestScan(remove=\'" + oldestScan + "')")
                

# private Functions
def getTargetDir(sdir, source, dest):
    if (not os.path.isdir(sdir)):
        print ("Only works for Directories!")
        return
    
    dirs = []
    head, tail = os.path.split(sdir)
    dirs.append(tail)
    while (not os.path.samefile(head, source)):
        if os.path.samefile(head, '/'):
            print ("sdir needs to be inside of source!")
            return
        
        head, tail = os.path.split(head)
        dirs.append(tail)
    
    dirs.reverse()
    ddir = dest
    for d in dirs:
        ddir =  os.path.join(ddir,d)
    return ddir

def countFiles(directory):
    files = []
 
    if os.path.isdir(directory):
        for path, dirs, filenames in os.walk(directory):
            files.extend(filenames)
 
    return len(files)
    
def makedirs(dest):
    if not os.path.exists(dest):
        os.makedirs(dest)
        
        
        