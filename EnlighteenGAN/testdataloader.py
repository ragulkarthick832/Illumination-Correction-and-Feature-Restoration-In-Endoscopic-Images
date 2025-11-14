from dataloader import Endo3InputDataset

dataset = Endo3InputDataset(root="../../Dataset", split="train", img_size=256)

print(len(dataset))
inp, gt = dataset[0]
print(inp.shape)   # should be torch.Size([9,256,256])
print(gt.shape)    # should be torch.Size([3,256,256])
