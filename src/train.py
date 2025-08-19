import argparse, time, pickle
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from model import GPT, GPTConfig

DATA_DIR = Path("data/tokens")
CKPT_DIR = Path("checkpoints/southpark-gpt-mini")
CKPT_DIR.mkdir(parents=True, exist_ok=True)

#ucitavamo binarne tokene iz datoteke bez dodatnog prevaranja tipa
def load_bin(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.uint32)

#omogucuje modelu da uci predvidati sljedeci token za svaki polozaj u bloku
class TokenBlockDataset(Dataset):
    def __init__(self, ids: np.ndarray, block_size: int):
        self.ids = torch.from_numpy(ids.astype(np.int64))
        self.block_size = block_size
    def __len__(self):
        return max(1, len(self.ids) - self.block_size - 1)
    def __getitem__(self, idx):
        x = self.ids[idx:idx+self.block_size]
        y = self.ids[idx+1:idx+self.block_size+1]
        return x, y

#racunamo prosjecni loss na validacijskom skupu za procjenu kvalitete modela
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    losses = []
    for x, y in loader:
        x = x.to(device) #ulazni token
        y = y.to(device) #sljdeci token za svaku poziciju
        _, loss = model(x, y)
        losses.append(loss.item())
        if len(losses) >= 50:
            break
    model.train()
    return sum(losses) / len(losses) if losses else float("inf")

def main(args):
    with open(DATA_DIR / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    vocab_size = int(meta["vocab_size"])

    train_ids = load_bin(DATA_DIR / "train.bin")
    val_ids   = load_bin(DATA_DIR / "val.bin")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[i] device={device}  vocab_size={vocab_size}  train_tokens={train_ids.size}  val_tokens={val_ids.size}")

    #definiramo arhitekturu i parametre za treniranje
    config = GPTConfig(
        vocab_size=vocab_size,
        n_layer=6,            
        n_head=6,
        n_embd=384,           
        block_size=args.block_size,
        dropout=args.dropout
    )
    model = GPT(config).to(device)

    #dohvacanje podataka u batchevima za treniranje i evaluaciju
    train_ds = TokenBlockDataset(train_ids, block_size=args.block_size)
    val_ds   = TokenBlockDataset(val_ids,   block_size=args.block_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, drop_last=True)

    #optimizator koji definira kako ce model uciti i updateati se kroz treniranje
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr, betas=(0.9, 0.95),
        weight_decay=args.weight_decay
    )

    #ucitavanje chekpointa, uz uvjet da je konfiguracija dobro postavljena
    ckpt_last = CKPT_DIR / "model-last.pt"
    ckpt_best = CKPT_DIR / "model-best.pt"
    best_val = float("inf")
    start_step = 0

    def same_cfg(ckpt_cfg: dict) -> bool:
        return (
            ckpt_cfg.get("vocab_size") == config.vocab_size and
            ckpt_cfg.get("block_size") == config.block_size and
            ckpt_cfg.get("n_layer")    == config.n_layer and
            ckpt_cfg.get("n_head")     == config.n_head and
            ckpt_cfg.get("n_embd")     == config.n_embd
        )

    checkpoint = None
    if ckpt_last.exists():  #osigurac da model nastavi trening s ispravnim chekpointom
        print(f"[i] Loading checkpoint {ckpt_last}")
        tmp = torch.load(ckpt_last, map_location=device)
        if same_cfg(tmp["config"]):
            checkpoint = tmp
        else:
            print("[i] model-last.pt config mismatch — skip")
    if checkpoint is None and ckpt_best.exists():  #ako zadnji chekpoint nije kompatibilan osigurava da se ucita najbolji posljednji kompantibilni chekpoint
        print(f"[i] Loading checkpoint {ckpt_best}")
        tmp = torch.load(ckpt_best, map_location=device)
        if same_cfg(tmp["config"]):
            checkpoint = tmp
        else:
            print("[i] model-best.pt config mismatch — skip")

    #omogucuje nastavak treniranja od zadnjeg spremljenog chekpointa, ili ako ga nema onda od pocetka
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state"])
        best_val   = checkpoint.get("val_loss", float("inf"))
        start_step = checkpoint.get("step", 0)
        if "optim_state" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optim_state"])
            except Exception:
                print("[i] optimizer state incompatible — starting optimizer fresh")
        print(f"[i] Resuming from step {start_step} (best_val={best_val:.3f})")
    else:
        print("[i] No compatible checkpoint found; starting fresh.")

    #pocetak treniranja
    global_step = start_step
    t0 = time.time()

    #treniranje modela, pracenje loss-a, spremanje chekpointa
    for epoch in range(args.epochs):
        for x, y in train_loader:
            x = x.to(device); y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            global_step += 1
            if global_step % args.log_interval == 0:
                tok_per_step = args.batch_size * args.block_size
                dt = time.time() - t0
                print(f"step {global_step:6d} | loss {loss.item():.3f} | tok/s ~ {tok_per_step / max(1e-6, dt):.0f}")
                t0 = time.time()
            if global_step % args.eval_interval == 0:
                val_loss = evaluate(model, val_loader, device)
                print(f"[eval] step {global_step} | val_loss {val_loss:.3f}")
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save({
                        "model_state": model.state_dict(),
                        "optim_state": optimizer.state_dict(),
                        "config": config.__dict__,
                        "val_loss": val_loss,
                        "step": global_step
                    }, ckpt_best)
                    print(f"[ckpt] saved {ckpt_best}")
                torch.save({
                    "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "config": config.__dict__,
                    "val_loss": best_val,
                    "step": global_step
                }, ckpt_last)

            if global_step >= args.max_steps:
                break
        if global_step >= args.max_steps:
            break

    #spremanje zadnjeg stanja modela nakon kraja treniranja
    torch.save({
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "config": config.__dict__,
        "val_loss": best_val,
        "step": global_step
    }, ckpt_last)
    print(f"[done] saved {ckpt_last}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--block_size", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=3000)
    ap.add_argument("--log_interval", type=int, default=50)
    ap.add_argument("--eval_interval", type=int, default=300)
    ap.add_argument("--weight_decay", type=float, default=0.04)
    ap.add_argument("--dropout", type=float, default=0.3)
    args = ap.parse_args()
    main(args)
