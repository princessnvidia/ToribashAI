#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from copy import deepcopy

import torch
import torch.nn as nn

ROOT = Path.home() / "Documents" / "ToribashAI"
MODEL_PATH = ROOT / "models" / "xioi_assassin_loop_gru_v42.pt"
REF_PATH = ROOT / "generated_replays" / "xioi_assassin_reference_v42_0_315.json"
TEMPLATE_RPL = ROOT / "generated_replays" / "xioi_427_assassincreedhunter_v37.rpl"
OUT_DIR = ROOT / "generated_replays"
STEAM_REPLAY = Path.home() / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash/replay"
SEQ_LEN = 8
POINT_ORDER = ["head","chest","lumbar","abs","left_shoulder","right_shoulder","left_hip","right_hip","left_foot","right_foot"]

class GRUAction(nn.Module):
    def __init__(self, state_dim: int, hidden: int, layers: int):
        super().__init__()
        self.gru = nn.GRU(state_dim, hidden, layers, batch_first=True, dropout=0.10 if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 20 * 5))
    def forward(self, x):
        z, _ = self.gru(x)
        return self.head(z[:, -1]).view(-1, 20, 5)

def make_state(fr, prev_action, origin):
    st=[]
    for name in POINT_ORDER:
        p=fr["points"].get(name,[0,0,0])
        st.extend([p[0]-origin[0], p[1]-origin[1], p[2]-origin[2]])
    st.extend(prev_action)
    return st

def load_model(device):
    ckpt=torch.load(MODEL_PATH,map_location=device)
    model=GRUAction(ckpt["state_dim"], ckpt.get("hidden",192), ckpt.get("layers",2)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print("Loaded", MODEL_PATH, "epoch", ckpt.get("epoch"))
    return model

def compact_pairs(action):
    return [[i,int(v)] for i,v in enumerate(action) if int(v)!=0]

def rewrite_rpl(actions_by_frame, out_path, fightname):
    lines=TEMPLATE_RPL.read_text(encoding="utf-8", errors="ignore").splitlines()
    out=[]; current=None; skip_joint0=False
    frame_re=re.compile(r"^FRAME\s+(\d+)\s*;")
    for line in lines:
        m=frame_re.match(line.strip())
        if m:
            if current is not None and current in actions_by_frame:
                pairs=compact_pairs(actions_by_frame[current])
                if pairs:
                    out.append("JOINT 0; " + " ".join(f"{j} {v}" for j,v in pairs))
            current=int(m.group(1)); skip_joint0=False
            out.append(line)
            continue
        if line.startswith("FIGHTNAME 0;"):
            out.append(f"FIGHTNAME 0; {fightname}"); continue
        if current is not None and current in actions_by_frame and line.strip().startswith("JOINT 0;"):
            continue
        out.append(line)
    if current is not None and current in actions_by_frame:
        # if last frame action was not inserted due no next frame, harmless duplicate guard omitted
        pass
    out_path.write_text("\n".join(out)+"\n", encoding="utf-8")

def main():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=load_model(device)
    ref=json.loads(REF_PATH.read_text(encoding="utf-8"))
    frames=ref["frames"]
    frame_numbers=[f["frame"] for f in frames]
    source_by_idx={i:f for i,f in enumerate(frames)}
    for seed_frames,total_frames in [(35,900),(70,900),(140,1200),(200,1200)]:
        seed_count=max(SEQ_LEN, sum(1 for f in frames if f["frame"]<=seed_frames))
        generated=[deepcopy(fr) for fr in frames[:seed_count]]
        prev_action=generated[-1].get("action", [0]*20)
        # Use source positions cyclically as body-state scaffold, while actions after seed are GRU-predicted.
        for step in range(seed_count, min(len(frames), 10**9)):
            if len(generated) >= total_frames//5:
                break
            src=deepcopy(frames[step % len(frames)])
            origin=generated[-SEQ_LEN]["points"].get("chest", [0,0,0])
            seq=[]
            for fr in generated[-SEQ_LEN:]:
                seq.append(make_state(fr, fr.get("action", prev_action), origin))
            x=torch.tensor([seq], dtype=torch.float32, device=device)
            with torch.no_grad():
                pred=model(x).argmax(dim=-1)[0].cpu().tolist()
            src["action"]=[int(v) for v in pred]
            src["joint_pairs"]=compact_pairs(pred)
            src["frame"] = len(generated)*5
            generated.append(src)
            prev_action=src["action"]
        actions={fr["frame"]: fr["action"] for fr in generated}
        name=f"xioi_assassin_loop_gru_free_v42_seed{seed_frames}"
        out=OUT_DIR/(name+".rpl")
        rewrite_rpl(actions, out, name)
        for dest in [STEAM_REPLAY/(name+".rpl"), STEAM_REPLAY/"parkour"/(name+".rpl")]:
            dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(out,dest)
        print("Wrote", out)
if __name__=="__main__":
    main()
