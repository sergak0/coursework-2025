#!/usr/bin/env python
import argparse
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import os
from tqdm import tqdm

class PairedImageDataset(Dataset):
    def __init__(self, root: str, phase: str = "train"):
        root = Path(root)
        self.paths_A = sorted((root / phase / "real").glob("*"))
        self.paths_B = sorted((root / phase / "blended").glob("*"))
        files_a = os.listdir(root / phase / "real")
        files_b = os.listdir(root / phase / "blended")
        files_b = [el for el in files_b if el in files_a and '.jpg' in el]
        files_a = [el for el in files_a if el in files_b and '.jpg' in el]
        
        self.paths_B = [el for el in self.paths_B if str(el)[str(el).rfind('/') + 1:] in files_b]
        self.paths_A = [el for el in self.paths_A if str(el)[str(el).rfind('/') + 1:] in files_a]
        self.tf = transforms.Compose([
            transforms.Resize((1024, 1024), Image.BICUBIC),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,)*3, (0.5,)*3),
        ])
    def __len__(self):
        return len(self.paths_A)
    def __getitem__(self, idx):
        A = self.tf(Image.open(self.paths_A[idx]).convert("RGB"))
        B = self.tf(Image.open(self.paths_B[idx]).convert("RGB"))
        return {"A": A, "B": B}

def sn(m):
    return nn.utils.spectral_norm(m)

class Down(nn.Module):
    def __init__(self, cin, cout, norm=True):
        super().__init__()
        layers: List[nn.Module] = [sn(nn.Conv2d(cin, cout, 4, 2, 1, bias=False))]
        if norm:
            layers.append(nn.InstanceNorm2d(cout))
        layers.append(nn.LeakyReLU(0.2, True))
        self.block = nn.Sequential(*layers)
    def forward(self, x):
        return self.block(x)

class Up(nn.Module):
    def __init__(self, cin, cout, dropout=False):
        super().__init__()
        layers = [sn(nn.ConvTranspose2d(cin, cout, 4, 2, 1, bias=False)),
                  nn.InstanceNorm2d(cout), nn.ReLU(True)]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)
    def forward(self, x, skip):
        return torch.cat([self.block(x), skip], 1)

class UNetGenerator(nn.Module):
    def __init__(self, in_c=3, out_c=3, base=64):
        super().__init__()
        # Encoder 10 layers
        self.d1 = Down(in_c, base, norm=False)        # 512→512? Actually 1024→512 spatial dims
        self.d2 = Down(base, base*2)                  # 512
        self.d3 = Down(base*2, base*4)                # 256
        self.d4 = Down(base*4, base*8)                # 128
        self.d5 = Down(base*8, base*8)                # 64
        self.d6 = Down(base*8, base*8)                # 32
        self.d7 = Down(base*8, base*8)                # 16
        self.d8 = Down(base*8, base*8)                # 8
        self.d9 = Down(base*8, base*8)                # 4
        self.d10= Down(base*8, base*8, norm=False)    # 2→1 (bottleneck, no norm)

        # Decoder
        self.u1 = Up(base*8,  base*8,  dropout=True)   # 1→2
        self.u2 = Up(base*16, base*8,  dropout=True)   # 2→4
        self.u3 = Up(base*16, base*8,  dropout=True)   # 4→8
        self.u4 = Up(base*16, base*8)                  # 8→16
        self.u5 = Up(base*16, base*8)                  # 16→32
        self.u6 = Up(base*16, base*8)                  # 32→64  (still 512 out)
        self.u7 = Up(base*16, base*4)                  # 64→128 (512→256 out)
        self.u8 = Up(base*8,  base*2)                  # 128→256 (256→128 out)
        self.u9 = Up(base*4,  base)                    # 256→512 (128→64 out)

        # Final conv: input = 64 + 64 (cat) = 128
        self.outc = nn.Sequential(sn(nn.ConvTranspose2d(base*2, out_c, 4, 2, 1)), nn.Tanh())

    def forward(self, x):
        d1=self.d1(x); d2=self.d2(d1); d3=self.d3(d2); d4=self.d4(d3); d5=self.d5(d4)
        d6=self.d6(d5); d7=self.d7(d6); d8=self.d8(d7); d9=self.d9(d8); d10=self.d10(d9)
        u1=self.u1(d10,d9); u2=self.u2(u1,d8); u3=self.u3(u2,d7); u4=self.u4(u3,d6)
        u5=self.u5(u4,d5); u6=self.u6(u5,d4); u7=self.u7(u6,d3); u8=self.u8(u7,d2); u9=self.u9(u8,d1)
        return self.outc(u9)

class PatchD(nn.Module):
    def __init__(self, inc=6, base=64, nlayers=4):
        super().__init__()
        layers=[sn(nn.Conv2d(inc, base, 4,2,1)), nn.LeakyReLU(0.2,True)]
        nf=base
        for i in range(1,nlayers):
            mult=min(2**i,8)
            layers+=[sn(nn.Conv2d(nf, base*mult,4,2,1,bias=False)), nn.InstanceNorm2d(base*mult), nn.LeakyReLU(0.2,True)]
            nf=base*mult
        layers.append(sn(nn.Conv2d(nf,1,4,1,1)))
        self.model=nn.Sequential(*layers)
    def forward(self,x):
        return self.model(x)
        
class MultiScaleD(nn.Module):
    def __init__(self, inc=6):
        super().__init__()
        self.Ds=nn.ModuleList([PatchD(inc) for _ in range(3)])
        self.down=nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
    def forward(self,A,B):
        x=torch.cat([A,B],1); outs=[]
        for D in self.Ds:
            outs.append(D(x)); x=self.down(x)
        return outs
        
class GANLoss(nn.Module):
    def __init__(self):
        super().__init__(); self.register_buffer("real",torch.tensor(1.0)); self.register_buffer("fake",torch.tensor(0.0)); self.crit=nn.BCEWithLogitsLoss()
    def forward(self,pred,is_real):
        tgt=(self.real if is_real else self.fake).expand_as(pred); return self.crit(pred,tgt)

def init_weights(net):
    for m in net.modules():
        if isinstance(m,(nn.Conv2d,nn.ConvTranspose2d,nn.Linear)):
            nn.init.normal_(m.weight,0.,0.02); 
            if m.bias is not None: nn.init.constant_(m.bias,0)

def train(
    dataroot: str,
    epochs: int = 400,
    batch_size: int = 1,
    lr: float = 2e-4,
    lambda_L1: float = 100.0,
    device: str = "cuda",
    out_dir: str = "./outputs",
    resume: str | None = None,
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"▶ Training on {device}")

    tr_loader = DataLoader(PairedImageDataset(dataroot,"train"), batch_size=batch_size,
                           shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(PairedImageDataset(dataroot,"val"), batch_size=1, shuffle=False)

    netG = UNetGenerator().to(device)
    netD = MultiScaleD().to(device)
    init_weights(netG); init_weights(netD)

    criterionGAN = GANLoss().to(device)
    criterionL1  = nn.L1Loss()
    optG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5,0.999))
    optD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(0.5,0.999))
    scaler = torch.cuda.amp.GradScaler()
    start_epoch = 0

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = out_dir/"samples"; sample_dir.mkdir(exist_ok=True)
    if resume:
        ckpt=torch.load(resume,map_location="cpu")
        netG.load_state_dict(ckpt["G"]); netD.load_state_dict(ckpt["D"])
        optG.load_state_dict(ckpt["optG"]); optD.load_state_dict(ckpt["optD"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch=ckpt["epoch"]+1
        print(f"▶ Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        netG.train(); netD.train()
        pbar=tqdm(tr_loader,desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            realA=batch["A"].to(device,non_blocking=True)
            realB=batch["B"].to(device,non_blocking=True)
            
            optG.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                fakeB = netG(realA)
                pred_fake = netD(realA, fakeB)
                loss_G_GAN = sum(criterionGAN(p,True) for p in pred_fake)/len(pred_fake)
                loss_G_L1 = criterionL1(fakeB, realB)*lambda_L1
                loss_G = loss_G_GAN + loss_G_L1
            scaler.scale(loss_G).backward()
            scaler.step(optG)

            optD.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                pred_real = netD(realA, realB)
                loss_D_real = sum(criterionGAN(p,True) for p in pred_real)/len(pred_real)
                pred_fake_det = netD(realA, fakeB.detach())
                loss_D_fake = sum(criterionGAN(p,False) for p in pred_fake_det)/len(pred_fake_det)
                loss_D = (loss_D_real + loss_D_fake)*0.5
            scaler.scale(loss_D).backward()
            scaler.step(optD)
            scaler.update()

            pbar.set_postfix(G=loss_G.item(), D=loss_D.item())

        netG.eval();
        with torch.no_grad():
            for i, vb in enumerate(val_loader):
                if i>=4: break
                A = vb["A"].to(device); B = vb["B"].to(device); F = netG(A)
                grid=torch.cat([A, F, B],0)
                save_image((grid+1)/2, sample_dir/f"ep{epoch+1:04d}_idx{i}.png", nrow=3)
                
        ckpt={"epoch":epoch, "G":netG.state_dict(), "D":netD.state_dict(),
              "optG":optG.state_dict(), "optD":optD.state_dict(), "scaler":scaler.state_dict()}
        torch.save(ckpt, out_dir/"latest.pt")
        if (epoch+1)%5==0 or epoch+1==epochs:
            torch.save(ckpt, out_dir/f"epoch_{epoch+1:04d}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pix2Pix 1024×1024 Trainer")
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lambda_L1", type=float, default=100.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out_dir", default="./outputs")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    train(
        dataroot=args.dataroot,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_L1=args.lambda_L1,
        device=args.device,
        out_dir=args.out_dir,
        resume=args.resume,
    )
