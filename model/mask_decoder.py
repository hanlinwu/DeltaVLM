"""
AnyUp-style Mask Decoder for DeltaVLM
Reference: AnyUp, Change-Agent
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, ng=8):
        super().__init__()
        p = k // 2
        ng_in, ng_out = min(ng, in_ch), min(ng, out_ch)
        self.block = nn.Sequential(
            nn.GroupNorm(ng_in, in_ch), nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, k, padding=p, padding_mode='reflect', bias=False),
            nn.GroupNorm(ng_out, out_ch), nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, k, padding=p, padding_mode='reflect', bias=False))
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
    def forward(self, x):
        return self.block(x) + self.shortcut(x)

class LearnedFeatureUnification(nn.Module):
    def __init__(self, out_ch, k=5):
        super().__init__()
        self.out_ch, self.k = out_ch, k
        self.basis = nn.Parameter(torch.randn(out_ch, 1, k, k) * 0.02)
    def forward(self, f):
        b, c, h, w = f.shape
        p = self.k // 2
        x = F.conv2d(F.pad(f, (p,p,p,p), mode='reflect'), self.basis.repeat(c,1,1,1), groups=c)
        mask = torch.ones(1,1,h,w, dtype=x.dtype, device=x.device)
        denom = F.conv2d(F.pad(mask, (p,p,p,p), value=0), torch.ones(1,1,self.k,self.k, dtype=x.dtype, device=x.device))
        x = x / (denom + 1e-8)
        return F.softmax(x.view(b, self.out_ch, c, h, w), dim=1).mean(dim=2)

class LocalWindowAttention(nn.Module):
    def __init__(self, dim, nh=4, ws=7):
        super().__init__()
        self.dim, self.nh, self.ws = dim, nh, ws
        self.hd, self.scale = dim // nh, (dim // nh) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.rpb = nn.Parameter(torch.zeros((2*ws-1)**2, nh))
        nn.init.trunc_normal_(self.rpb, std=0.02)
        coords = torch.stack(torch.meshgrid(torch.arange(ws), torch.arange(ws), indexing='ij'))
        cf = coords.flatten(1)
        rc = cf[:,:,None] - cf[:,None,:]
        rc = rc.permute(1,2,0).contiguous()
        rc[:,:,0] += ws-1; rc[:,:,1] += ws-1; rc[:,:,0] *= 2*ws-1
        self.register_buffer("rpi", rc.sum(-1))
    def forward(self, x):
        B, H, W, C = x.shape
        ws = self.ws
        ph, pw = (ws - H % ws) % ws, (ws - W % ws) % ws
        if ph > 0 or pw > 0:
            x = F.pad(x, (0,0,0,pw,0,ph))
        Hp, Wp = x.shape[1], x.shape[2]
        x = x.view(B, Hp//ws, ws, Wp//ws, ws, C).permute(0,1,3,2,4,5).contiguous().view(-1, ws*ws, C)
        Bw = x.shape[0]
        qkv = self.qkv(x).reshape(Bw, ws*ws, 3, self.nh, self.hd).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn + self.rpb[self.rpi.view(-1)].view(ws*ws, ws*ws, -1).permute(2,0,1).unsqueeze(0)
        out = (F.softmax(attn, dim=-1) @ v).transpose(1,2).reshape(Bw, ws*ws, C)
        out = self.proj(out).view(B, Hp//ws, Wp//ws, ws, ws, C).permute(0,1,3,2,4,5).contiguous().view(B, Hp, Wp, C)
        return out[:,:H,:W,:].contiguous() if ph > 0 or pw > 0 else out

class UpsampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_attn=True):
        super().__init__()
        self.use_attn = use_attn
        self.up = nn.Sequential(nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1), nn.GroupNorm(min(8,out_ch), out_ch), nn.SiLU())
        self.ref = ResBlock(out_ch, out_ch)
        if use_attn:
            self.attn = LocalWindowAttention(out_ch, 4, 7)
            self.norm = nn.LayerNorm(out_ch)
    def forward(self, x):
        x = self.ref(self.up(x))
        if self.use_attn:
            B,C,H,W = x.shape
            x = x + self.attn(self.norm(x.permute(0,2,3,1))).permute(0,3,1,2)
        return x

class ChangeMaskDecoder(nn.Module):
    def __init__(self, in_ch=1408, hid=128, out_sz=256, n_stages=4, use_attn=True):
        super().__init__()
        self.in_ch, self.hid, self.out_sz = in_ch, hid, out_sz
        self.lfu = LearnedFeatureUnification(hid, 5)
        self.proj = nn.Sequential(nn.Conv2d(hid, hid, 3, padding=1), nn.GroupNorm(8, hid), nn.SiLU(), ResBlock(hid, hid))
        self.stages = nn.ModuleList()
        cur = hid
        for i in range(n_stages):
            nxt = max(hid // (2**(i+1)), 32)
            self.stages.append(UpsampleBlock(cur, nxt, use_attn and i < n_stages-1))
            cur = nxt
        self.head = nn.Sequential(nn.Conv2d(cur, cur, 3, padding=1), nn.GroupNorm(min(8,cur), cur), nn.SiLU(), nn.Conv2d(cur, 1, 1))
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self, f, out_sz=None):
        if f.dim() == 3:
            B, N, C = f.shape
            if N == 514:
                H = W = 16
                f = (f[:,257:,:][:,1:,:] - f[:,:257,:][:,1:,:]).permute(0,2,1).view(B,C,H,W)
            elif N == 257:
                f = f[:,1:,:].permute(0,2,1).view(B,C,16,16)
            else:
                H = W = int(math.sqrt(N))
                f = f.permute(0,2,1).view(B,C,H,W)
        x = self.proj(self.lfu(f))
        for s in self.stages: x = s(x)
        m = self.head(x)
        tgt = out_sz if out_sz else (self.out_sz, self.out_sz)
        if m.shape[-2:] != tgt:
            m = F.interpolate(m, size=tgt, mode='bilinear', align_corners=False)
        return torch.sigmoid(m)
    def get_loss(self, pred, gt, pw=2.0):
        if pred.shape != gt.shape:
            gt = F.interpolate(gt.float(), size=pred.shape[-2:], mode='nearest')
        loss = F.binary_cross_entropy(pred, gt.float(), reduction='none')
        if pw != 1.0:
            loss = loss * torch.where(gt > 0.5, pw, 1.0)
        return loss.mean()

def build_mask_decoder(in_ch=1408, out_sz=256, hid=128, light=False):
    return ChangeMaskDecoder(in_ch, 64 if light else hid, out_sz, 4, not light)

if __name__ == "__main__":
    print("Testing ChangeMaskDecoder...")
    f = torch.randn(2, 514, 1408)
    dec = build_mask_decoder()
    print(f"Params: {sum(p.numel() for p in dec.parameters()):,}")
    with torch.no_grad(): m = dec(f)
    print(f"In: {f.shape}, Out: {m.shape}")
    print("OK!")
