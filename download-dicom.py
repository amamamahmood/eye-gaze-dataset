import os
path = "https://physionet.org/files/mimic-cxr/2.0.0/"

myfile = open("dcm-list.txt","r")


f = open("dcm-path-list.txt", "w")



myline = myfile.readline()
while myline:
    myline = myfile.readline()
    
    f.write(os.path.join(path, myline))
myfile.close()   
f.close()
