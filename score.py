# -*- coding: utf-8; -*-

from convert import write_starsem;

import tempfile;
import subprocess;
import sys;
import os;
import io;
import json

__author__ = "oe"
__version__ = "2018"

def starsem_score(gold, system, file = sys.stdout):
  one = tempfile.mkstemp(prefix = ".*sem.score.");
  two = tempfile.mkstemp(prefix = ".*sem.score.");
  with open(one[0], "w") as stream:
    write_starsem(gold, stream = stream);
  with open(two[0], "w") as stream:
    write_starsem(system, stream = stream);
  command = ["perl", "eval.cd-sco.pl", "-g", one[1], "-s", two[1]];
  with file if isinstance(file, io.IOBase) else open(file, "w") as stream:
    subprocess.run(command, stdout = stream);
  os.unlink(one[1]);
  os.unlink(two[1]);


if __name__ == '__main__':
  with open(sys.argv[1]) as gold, open(sys.argv[2]) as system:
    starsem_score(map(json.loads, gold), map(json.loads, system))