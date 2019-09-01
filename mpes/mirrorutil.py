#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: L. Rettig
"""

import os
from . import utils as u
import shutil

class CopyTool(object):
    """ File collecting and sorting class.
    """

    def __init__(self, source='/', dest='/', forceCopy=False, **kwds):

        self.source = source
        self.dest = dest
        self.forceCopy = forceCopy
        self.safetyMargin = kwds.pop('safetyMargin', 2 * 2**30) # Default 10 GB safety margin
        self.pbenv = kwds.pop('pbenv', 'classic')
        

    def copy(self, sdir):

        tqdm = u.tqdmenv(self.pbenv)
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
                if (size == 0 and not self.forceCopy):
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

                    print("Copy Files...")
                    for sfile in tqdm(filenames):
                        srcFile = os.path.join(path, sfile)

                        destFile = os.path.join(path.replace(sdir, ddir), sfile)

                        if (not os.path.exists(destFile) or self.forceCopy):
                            shutil.copy2(srcFile, destFile)

                        numCopied += 1
                    print("Copy finished!")

                return ddir
                
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
        
        
        