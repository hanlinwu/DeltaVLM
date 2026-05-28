"""Mask Dataset"""
import os,torch,numpy as np
from torch.utils.data import Dataset,DataLoader
from PIL import Image
from processor import Blip2ImageTrainProcessor,BlipImageEvalProcessor

class ChangeMaskDataset(Dataset):
    def __init__(s,root,split="train",isz=224,msz=256,train=True):
        s.root=os.path.join(root,split);s.isz=isz;s.msz=msz
        s.a=os.path.join(s.root,"A");s.b=os.path.join(s.root,"B");s.l=os.path.join(s.root,"label")
        s.files=sorted([f for f in os.listdir(s.l) if f.endswith(".png")]) if os.path.exists(s.l) else []
        s.proc=Blip2ImageTrainProcessor(image_size=isz) if train else BlipImageEvalProcessor(image_size=isz)
        print(f"ChangeMaskDataset[{split}]:{len(s.files)}")
    def __len__(s):return len(s.files)
    def __getitem__(s,i):
        n=s.files[i]
        try:a=Image.open(os.path.join(s.a,n)).convert("RGB");b=Image.open(os.path.join(s.b,n)).convert("RGB");l=Image.open(os.path.join(s.l,n))
        except:return s[(i+1)%len(s)]
        a,b=s.proc(a,b);l=np.array(l.resize((s.msz,s.msz),Image.NEAREST))
        m=((l.sum(2) if l.ndim==3 else l)>0).astype(np.float32)
        return {"image_A":a,"image_B":b,"gt_mask":torch.from_numpy(m).unsqueeze(0),"name":n}
    def collate(s,x):return{k:torch.stack([i[k] for i in x]) if isinstance(x[0][k],torch.Tensor) else [i[k] for i in x] for k in x[0]}

def build_loaders(root,bs=8,nw=4):
    tr=ChangeMaskDataset(root,"train");va=ChangeMaskDataset(root,"val",train=False)
    return DataLoader(tr,bs,True,num_workers=nw,collate_fn=tr.collate,pin_memory=True,drop_last=True),DataLoader(va,bs,False,num_workers=nw,collate_fn=va.collate)

