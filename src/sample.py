import argparse, pickle, re
from pathlib import Path
import torch

#otvara meta.pkl datoteku i ucitava podatke
def load_meta(meta_path: Path):
    with open(meta_path, "rb") as f:
        return pickle.load(f)

def build_encoder_decoder(meta): #funkcija za enkodiranje i dekodiranje teksta u tokene i obrnuto
    tok = meta.get("tokenizer", "spm") #dohvaca tip tokenizatora
    if tok == "spm": #ucitava SentencePiece model ako je tokenizator tipa spm
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=meta["spm_model"])
        def encode(s: str): return sp.encode(s, out_type=int) #funkcija koja pretvara string u listu tokena
        def decode(ids):     return sp.decode(ids) #funkcija koja pretvara listu tokena u tekst
        vocab_size = int(sp.get_piece_size()) #velicina vokabulara
        return encode, decode, vocab_size, "spm"
    #koristeno na pocetku ali sam promjenio na spm
    #elif tok == "char": #ako je char level tokenizator
        #vocab = meta["vocab"] 
        #stoi = {ch: i for i, ch in enumerate(vocab)}
        #itos = {i: ch for ch, i in stoi.items()}
        #def encode(s: str):  return [stoi.get(ch, 0) for ch in s]
        #def decode(ids):     return "".join(itos.get(int(i), "�") for i in ids)
        #return encode, decode, len(vocab), "char"
    #else:
        #raise ValueError(f"Nepoznat tokenizer u meta.pkl: {tok}")

#ciscenje generiranog teksta
def clean_script(text: str) -> str:
    #uklanjamo cudne znakove, npr ?? i mjenjamo ih s obicnim znakovima
    text = (text
            .replace("\u2047", "?")
            .replace("“", '"').replace("”", '"')
            .replace("’", "'").replace("‘", "'"))
    #uklanja scenske upute u []
    text = re.sub(r"\[[^\]]*\]", "", text)
    #uklanja visestruke razmake i smanjuje visestruke prazne linije
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    #skracivanje predugih linija teksta
    lines = []
    for line in text.splitlines(): #iteriramo krou tekst, ako je linija kraca od 220 znakova uklanjamo razmake i dodajemo ju u lines
        if len(line) <= 220:
            lines.append(line.strip())
            continue
        #pokusavamo podijeliti na dijelova na kraju recenica, nakon interpukcijskih znakova ili razmaka
        parts = re.split(r"(?<=[.!?])\s+", line)
        buf = "" #buffer za novu liniju
        for p in parts:
            if len(buf) + len(p) + 1 <= 220: #provjerava moze li se trenutni dio dodati u buffer ali da ne prede 220 znakova
                buf = (buf + " " + p).strip() #ako moze dodaje se taj dio u buffer i uklanja nepotrebne razmake
            else:
                if buf: #ako bi dodavanje preslo 220 znakova, dodaje se trenutni buffer kao nova linija i zapocinje se novi buffer
                    lines.append(buf)
                buf = p
        if buf: #nakon sto su svi dijelovi obradeni dodaje preostali buffer u listu osim ako je prazan
            lines.append(buf)
    return "\n\n".join(l for l in lines if l) #spaja sve u jedan tekst i odvaja s 2 nova reda

# ---------- Sampling ----------
@torch.no_grad() #ne koristimo gradijente jer ne trenira nego samo generira tekst
def generate_until(model, idx, decode, *,     #funkcija za generiranje teksta iz modela
                   temperature=0.8,           #idx = pocetni tokeni (prompt), decode = funkcija za dekodiranje tokena nazad u tekst, * = svaki sljedeci argument mora biti zadan kao keyword argument (npr temperature = 0.8) 
                   top_k=50,  #temperature = nasumicnost generiranja, top_k = uzima 50 najvjerijatnijih tokena u svakom koraku
                   top_p=0.9, #uzima najmanji skup tokena cija suma vjerojatnosti prelazi 0.9
                   repetition_penalty=1.10, #penalizira ponavljanje istih tokena
                   stop_after_chars=1200, #zaustavlja generiranje nakon 1200 znakova
                   stop_regex=r"\n(?:THE END|END|CUT TO|SCENE)\b"): #regex za prepoznavanje kraja
    
    device = next(model.parameters()).device #dohvaca gdje se nalazi model (CPU ili GPU)
    model.eval() #stavlja model u eval mod, sto iskljucuje dropout i ostalo sto se koristi samo kod treniranja
    stop_re = re.compile(stop_regex, re.IGNORECASE) if stop_regex else None #ako je stop_regex zadan, kompajlira ga u regex objekt, ako nije stavlja ga na None, koristi se za provjeru je li generirani tekst dosao do kraja skripte

    while True:
        idx_cond = idx[:, -model.config.block_size:] #uzima block_size tokena iz trenutnog niza tokena
        logits, _ = model(idx_cond)  #prosljeduje te tokene modelu i dobija predvidanja, tj vjerojatnosti za svaki token na svakoj poziciji, dimenzija (B, T, V)
        logits = logits[:, -1, :]   #uzima predvidanja za zadnji token u sekvenci, dimenzija je sada (B, V)
        logits = logits / max(1e-6, temperature) #temperature scaling

        #repetition penalty za zadnjih do 200 tokena
        if repetition_penalty and repetition_penalty > 1.0:
            recent = idx[0, -min(200, idx.shape[1]):]
            logits[0, recent] /= repetition_penalty

        #samo k najvjerojatnijih tokena ostaju moguci za generiranje
        if top_k is not None:
            k = min(top_k, logits.size(-1))
            v, _ = torch.topk(logits, k=k)
            logits[logits < v[:, [-1]]] = -float("inf")

        #racuna vjerojatnosti svih tokena i sortira, zadrzava samo najmanji skup tokena cija je vjerojatnost veca top_p
        if top_p is not None:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumprobs = torch.cumsum(probs, dim=-1)
            cutoff = cumprobs > top_p
            cutoff[..., 1:] = cutoff[..., :-1].clone()
            cutoff[..., 0] = False
            sorted_logits[cutoff] = -float("inf")
            logits.scatter_(dim=-1, index=sorted_idx, src=sorted_logits)

        #generira sljedeci token
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)  # (1,1)
        idx = torch.cat([idx, next_id], dim=1)

        #provjerava uvjete za kraj generiranja
        text = decode(idx[0].tolist())

        if len(text) >= stop_after_chars:
            break
        if stop_re and stop_re.search(text):
            break
        if len(text) > 200 and "\n\n" in text[-200:]:
            break

    return text

# ---------- Main ----------
def main(args):
    DATA_DIR = Path("data/tokens")
    meta = load_meta(DATA_DIR / "meta.pkl")
    encode, decode, vocab_size, tok_kind = build_encoder_decoder(meta)

    #ucitavamo chekpoint
    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    #pripremamo konfiguraciju i provjeravamo kompatibilnost prije instanciranja modela
    from model import GPT, GPTConfig
    cfg = GPTConfig(**ckpt["config"])
    if cfg.vocab_size != vocab_size:
        raise ValueError(
            f"Vocab mismatch: ckpt expects {cfg.vocab_size}, meta has {vocab_size}. "
        )

    #instanciramo model
    model = GPT(cfg)
    model.load_state_dict(ckpt["model_state"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    #pripremamo pocetne ID, tj prompt za model
    prompt = args.prompt
    ids = torch.tensor([encode(prompt)], dtype=torch.long, device=device)

    #generiramo novi tekst na temelju prompta i zadanih parametara
    raw = generate_until(
        model, ids, decode,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        stop_after_chars=args.stop_after_chars,
        stop_regex=args.stop_regex
    )

    #ciscenje i prikazivanje generiranog teksta
    cleaned = clean_script(raw) if args.clean else raw
    print(cleaned)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=str,
                    default="CARTMAN: You guys, seriously.\nKYLE: ")
    ap.add_argument("--checkpoint", type=str,
                    default="checkpoints/southpark-gpt-mini/model-best.pt")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--repetition_penalty", type=float, default=1.10)
    ap.add_argument("--stop_after_chars", type=int, default=1200)
    ap.add_argument("--stop_regex", type=str,
                    default=r"\n(?:THE END|END|CUT TO|SCENE)\b")
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args()
    main(args)
