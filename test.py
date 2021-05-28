import os
import torch

if __name__ == "__main__":
    testlist = [1,2,3,4]
    readlist = []
    testdict = {"0":testlist}
    modelname = "features.pt"
    if os.path.exists(modelname):
        state_dict = torch.load(modelname)
        print(state_dict)
    else:
        modelname = "features.pt"
        torch.save( testlist, modelname )
    
