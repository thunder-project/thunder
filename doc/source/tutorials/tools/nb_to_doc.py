#! /usr/bin/env python
"""
Convert IPython notebook to a sphinx doc page.
"""
import os
import sys

def convert_nb(nbname):
    os.system("ipython nbconvert --to rst source/%s.ipynb --output %s --template=tools/custom.tpl" % (nbname, nbname))
    os.system("mv %s.rst build/%s.rst" % (nbname, nbname))
    os.system("mv %s_files build/%s_files" % (nbname, nbname))

if __name__ == "__main__":
    for nbname in sys.argv[1:]:
        convert_nb(nbname.replace('.ipynb', ''))
