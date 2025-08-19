#korpus za South Park .txt fileove koji parsira 2 formata:
#1) NAME: text
#2) NAME u jednoj liniji pa onda ispod tekst


import argparse
from pathlib import Path
import re
import random
import pickle
import numpy as np
import sentencepiece as spm

RAW_DIR = Path("data/raw") #direktorij s originalnim skriptama
PROC_DIR = Path("data/processed"); PROC_DIR.mkdir(parents=True, exist_ok=True) #direktorij za obradene skripte
TOK_DIR  = Path("data/tokens");    TOK_DIR.mkdir(parents=True, exist_ok=True) #putanja do tokena i meta podataka

LINE_COLON = re.compile(r'^\s*([^:]{1,120})\s*:\s*(.+)$') #regex za prepoznavanje formata 1)

NAME_ONLY = re.compile(r'^[A-Za-z][A-Za-z.\'-]*'
                       r'(?:\s+[A-Za-z][A-Za-z.\'-]*){0,3}\s*$') #regex za prepoznavanje formata 2)

SCENE_ONLY = re.compile(r'^\s*[\(\[].*[\)\]]\s*$') #regex za prepoznavanje scenskih napomena, ako je npr tekst u () ili []

#popis uobicajenih govornika, za lakse prepoznavanje imena govornika
COMMON_SPEAKERS = {
    "STAN", "STAN MARSH", "KYLE", "KYLE BROFLOVSKI", "CARTMAN", "ERIC CARTMAN",
    "KENNY", "KENNY MCCORMICK", "BUTTERS", "CHEF", "RANDY", "MR. GARRISON",
    "WENDY", "IKE", "OFFICER BARBRADY", "TOWELIE", "TWEEK", "JIMMY", "TIMMY",
    "MR. MACKEY", "GOD", "SATAN", "NARRATOR", "NEWS ANCHOR", "REPORTER"
}

def norm_name(name: str) -> str:
    n = name.strip().upper() #uklanjamo razmake s pocetka i kraja imena i pretvarmo u upper case
    n = "_".join(n.split()) #umjesto razmaka stavljamo _
    return n

#provjeravamo je li linija ime lika; ako nema :
def looks_like_name(line: str) -> bool:
    s = line.strip() #uklanjamo razmake s pocetka i kraja linije
    if not s:
        return False #ako je prazna linija
    #ako je linija neko ime iz COMMON_SPEAKERS onda vraca True
    if s in COMMON_SPEAKERS:
        return True
    if NAME_ONLY.match(s):    #ako je linija potencijalno ime dodatno provjeravamo da ako je ime samo 1 rijec i ima 2 ili manje slova onda vrati False a u suprotnom vraca True
        tokens = s.split()
        if len(tokens) == 1 and len(tokens[0]) <= 2:
            return False
        return True
    return False

def parse_file_text(text: str) -> list[str]: #vraca liniju oblika NAME: text
    out = [] #rezultat
    current_speaker = None
    buffer = []  #skupljamo recenice dok se ne promjeni govornik

    def flush(): #ako postoji govornik i buffer sadrzi neki tekst, spaja sve te linije u 1 redak i doda u out
        nonlocal current_speaker, buffer
        if current_speaker and buffer:
            text_join = " ".join(x.strip() for x in buffer if x.strip())
            if text_join:
                out.append(f"{current_speaker}: {text_join}")
        buffer = []

    for raw in text.splitlines(): #prolazimo kroz svaku liniju u tekstu
        line = raw.strip() #uklanjamo razmake s pocetka i kraja
        if not line: #ako je linija prazna poziva flush, resetira govornika i prelazi na sljedecu liniju
            flush()
            current_speaker = None
            continue

        #za oblik NAME: text
        m = LINE_COLON.match(line) #ako pronade taj oblik poziva flush, normalizira ime govornika, uzima tekst, i ako je oboje valjano postavlja govornika i buffer na tu recenicu, u suprotnom resetira govornika i buffer
        if m:
            flush()
            who = norm_name(m.group(1))
            utt = m.group(2).strip()
            if who and utt:
                current_speaker = who
                buffer = [utt]
            else:
                current_speaker = None
                buffer = []
            continue

        #ako je scenska napomena
        if SCENE_ONLY.match(line): #poziva flush, resetira govornika i ide na sljedecu liniju
            flush()
            current_speaker = None
            continue

        #ako je oblik NAME text (bez :)
        if looks_like_name(line): #poziva flush, resetira govornika i ide dalje
            flush()
            current_speaker = norm_name(line)
            buffer = []
            continue

        #obicna linija teksta
        if current_speaker: #ako ima govornika dodaje tu liniju u buffer, tj ona pripada tom govorniku, ako ne ignorira(npr ako je naracija)
            buffer.append(line)
        else:
            pass

    flush()
    return out

def build_corpus() -> tuple[str, int, int]:
    files = sorted(RAW_DIR.glob("*.txt")) #uzima sve datoteke iz raw direktorija
    if not files:
        raise SystemExit(f"Nema .txt datoteka u {RAW_DIR.resolve()}")

    parts = [] #inicijaliziramo varijable za dijelove korpusa, broj linija i broj parsiranih linija
    total_lines = 0
    kept = 0 #pokazuje koliko recenica je zapravo izvuceno za treniranje
    for fp in files: #ucitamo tekst, parsiramo ga funkcijom parse_file_text, zbrajamo broj parsiranih linija i ukupnih linija, dodajemo parsirane linije u listu parts
        text = fp.read_text(encoding="utf-8", errors="ignore")
        lines = parse_file_text(text)
        kept += len(lines)
        total_lines += len(text.splitlines())
        if lines:
            parts.append("\n".join(lines))

    corpus = "\n\n".join(parts) #spajamo sve dijelove u jedan veliki korpus
    (PROC_DIR / "southpark.txt").write_text(corpus, encoding="utf-8")
    print(f"[i] Fajlova: {len(files)} | Ulaznih linija: {total_lines:,} | Parsiranih replika: {kept:,} | Znakova korpusa: {len(corpus):,}") #ispisujemo broj fileova, ulaznih linija, parsiranih replika i znakova korpusa
    preview = "\n".join(corpus.splitlines()[:12]) #pokazuje prvih 12 linija korpusa
    print("[i] Preview prvih 12 linija:\n" + "-"*60 + f"\n{preview}\n" + "-"*60)
    return corpus, len(files), kept

def main(vocab_size: int, val_ratio: float, seed: int, overwrite: bool):
    random.seed(seed)
    np.random.seed(seed)

    if overwrite: #ako se pozove brise stare tokene, model i korpus
        for p in [TOK_DIR/"train.bin", TOK_DIR/"val.bin", TOK_DIR/"meta.pkl",
                  TOK_DIR/"spm.model", TOK_DIR/"spm.vocab",
                  PROC_DIR/"southpark.txt"]:
            try: p.unlink()
            except FileNotFoundError: pass

    corpus, n_files, kept_lines = build_corpus() #kreira i sprema korpus i ispusuje broj fileova i parsiranih replika
    if not corpus:
        raise SystemExit("Korpus je prazan nakon parsiranja.")

    #koristimo SentencePiece BPE tokenizer na korpusu 
    spm_prefix = str(TOK_DIR / "spm")
    spm.SentencePieceTrainer.train(
        input=str(PROC_DIR / "southpark.txt"),
        model_prefix=spm_prefix, #gdje se sprema model i vokabular
        vocab_size=vocab_size, #broj tokena
        model_type="bpe", 
        character_coverage=1.0, #koristi sve znakove iz korpusa
        bos_id=-1, eos_id=-1, pad_id=0, unk_id=1, #ID za padding i unknown tokena
        input_sentence_size=0, #koristi sve recenice iz korpusa
        shuffle_input_sentence=False #ne mijesaj recenice
    )
    sp = spm.SentencePieceProcessor(model_file=spm_prefix + ".model") #ucitava trenirani model eknodiranje korpusa u tokene

    #tokeniziramo korpus
    ids = np.array(sp.encode(corpus, out_type=int), dtype=np.uint32) #tokeniziramo korpus, pretvara tekst u ID tokena
    N = int(ids.size) #broj tokena u korpusu
    n_val = max(1, int(N * val_ratio)) #broj tokena u validacijskom skupu
    train_ids, val_ids = ids[:-n_val], ids[-n_val:] #dijeli tokene u train i val skup

    (TOK_DIR/"train.bin").write_bytes(train_ids.tobytes()) #sprema train tokene
    (TOK_DIR/"val.bin").write_bytes(val_ids.tobytes()) #sprema val tokene 

    #dictionary u kojem se spremaju podaci o korpusu i tokenizaciji
    meta = { 
        "tokenizer": "spm",
        "spm_model": str((TOK_DIR/"spm.model").as_posix()), #outanja do modela
        "vocab_size": int(sp.get_piece_size()), #velicina vokabulara
        "ids_dtype": "uint32", #tip podataka
        "split": {
            "type": "token",
            "val_ratio": val_ratio, #omjer val tokena
            "N_total": N, #uk broj tokena
            "N_train": int(train_ids.size), #broj train tokena
            "N_val": int(val_ids.size), #broj val tokena
        },
        "diagnostics": {
            "n_files": n_files, #broj obradenih fileova
            "kept_lines": kept_lines #broj parsiranih replika
        }
    }
    with open(TOK_DIR/"meta.pkl", "wb") as f:  ##otvara meta.pkl datoteku i sprema dictionary u nju
        pickle.dump(meta, f)

    print(f"vocab_size={meta['vocab_size']} | train_tokens={train_ids.size:,} | val_tokens={val_ids.size:,}") #ispisuje velicinu vokabulara, broj train i val tokena

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab_size", type=int, default=4000)
    ap.add_argument("--val_ratio",  type=float, default=0.10)
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--overwrite",  action="store_true")
    args = ap.parse_args()
    main(vocab_size=args.vocab_size, val_ratio=args.val_ratio, seed=args.seed, overwrite=args.overwrite)
