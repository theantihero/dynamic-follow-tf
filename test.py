import subprocess
import os
os.chdir("C:\Git\dynamic-follow-tf")
subprocess.call(["g++", "file.cpp"])
tmp=subprocess.call("a.exe")
print("printing result")
print(tmp)