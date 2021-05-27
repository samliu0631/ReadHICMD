import os
import torch

if __name__ == "__main__":
    testlist = [1,2,3,4]
    readlist = []
    testdict = {"0":testlist}
    modelname = "features.pt"
    if os.path.exists(modelname):
        test = torch.load(modelname)
        for line in f.readlines():
            line = line.strip("\n")
            readlist.append(line)
        f.close()
    else:
        f= open("features.txt","w",encoding ="UTF-8")
        for linelist in testlist:
            f.write( str(linelist) )
            f.write("\n")
        f.close()
    
