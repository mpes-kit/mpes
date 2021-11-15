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
        self.gid = kwds.pop('gid', 5050)
        
        if (ntasks is None) or (ntasks < 0):
            # Default to 25 concurrent copy tasks
            self.ntasks = 25
        else:
            self.ntasks = int(ntasks)
        

    def copy(self, source, forceCopy=False, scheduler='threads', **compute_kwds):
        """ Local file copying method.
        """
        
        if not os.path.exists(source):
            print("Source not found!")
            return
        
        filenames = []
        dirnames = []
        
        if os.path.isfile(source):
            # Single file
            sdir = os.path.dirname(os.path.realpath(source))
            ddir = getTargetDir(sdir, self.source, self.dest, gid=self.gid, mode=0o775, create=True)
            filenames.append(source)
            
        elif os.path.isdir(source):
            sdir = source
            ddir = getTargetDir(sdir, self.source, self.dest, gid=self.gid, mode=0o775, create=True)
            #dirs.append(sdir)
            for path, dirs, files in os.walk(sdir):
                for file in files:
                    filenames.append(os.path.join(path, file))
                for dir in dirs:
                    dirnames.append(os.path.join(path, dir))      
                            
        #actual copy loop
        if filenames:
            # Check space left
            size_src = 0
            size_dst = 0
            for sfile in filenames:
                size_src += os.path.getsize(sfile)
                if (os.path.exists(sfile.replace(sdir, ddir))):
                    size_dst += os.path.getsize(sfile.replace(sdir, ddir))
            if (size_src == 0 and not forceCopy):
                # nothing to copy, just return directory
                return ddir
            else:
                total, used, free = shutil.disk_usage(ddir)
                if (size_src - size_dst > free - self.safetyMargin):
                    print("Target disk full, only " + str(free/2**30) + " GB free, but " + str((size_src - size_dst)/2**30) + " GB needed!")
                    return
        
            # make directories
            for directory in dirnames:
                destDir = directory.replace(sdir,ddir)
                mymakedirs(destDir, gid=self.gid, mode=0o775)

            copyTasks = [] # Core-level jobs
            for srcFile in filenames:
                destFile = srcFile.replace(sdir, ddir)
                size_src = os.path.getsize(srcFile)
                size_dst = os.path.getsize(srcFile)
                if (not os.path.exists(destFile) or size_dst != size_src or forceCopy):
                    if (os.path.exists(destFile)):
                        # delete existing file, to fix permission issue
                        copyTasks.append(d.delayed(mycopy)(srcFile, destFile, gid=self.gid, mode=0o664, replace=True))
                    else:
                        copyTasks.append(d.delayed(mycopy)(srcFile, destFile, gid=self.gid, mode=0o664))
            
            # run the copy tasks
            if len(copyTasks)>0:
                print("Copy Files...")
                with ProgressBar():
                    d.compute(*copyTasks, scheduler=scheduler, num_workers=self.ntasks, **compute_kwds)
                print("Copy finished!")

            if os.path.isfile(source):
                return destFile
            elif os.path.isdir(source):
                return ddir


    def size(self, sdir):
        """ Calculate file size.
        """
        
        for path, dirs, filenames in os.walk(sdir):
            # Check space left
            size = 0
            for sfile in filenames:
                size += os.path.getsize(os.path.join(sdir,sfile))
            return size
                
    def cleanUpOldestScan(self, remove = None, force = False):
        """ Remove scans in old directories.
        """        
        
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
def getTargetDir(sdir, source, dest, gid, mode, create=False):
    """ Retrieve target directories.
    """
    
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
        if create==True and not os.path.exists(ddir):
            mymakedirs(ddir, mode, gid)
    return ddir

    
# replacement for os.makedirs, which is independent of umask
def mymakedirs(path, mode, gid):
    """ Create directories.
    """

    if not path or os.path.exists(path):
        return []
    (head, tail) = os.path.split(path)
    res = mymakedirs(head, mode, gid)
    os.mkdir(path)
    os.chmod(path, mode)
    os.chown(path, -1, gid)
    res += [path]
    return res
    
def mycopy(source, dest, gid, mode, replace=False):
    """ Copy function with option to delete the target file firs (to take ownership).
    """

    if replace:
        if (os.path.exists(dest)):
            os.remove(dest)
    shutil.copy2(source, dest)
    # fix permissions and group ownership:
    os.chown(dest, -1, gid)
    os.chmod(dest, mode)