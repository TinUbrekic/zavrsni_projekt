## 1. Arhitektura
Projekt se sastoji od nekoliko dijelova:
- **prepare_southpark.py**
  - Učitava sve epizode u tekstualnom obliku i tokenizira ih pomoću *SentencePiece* biblioteke
  - Rezultat su `.bin` datoteke podijeljene na *train* i *val* skup

- **model.py**
  - Implementacija transformer arhitekture (GPT-like model)
  - Parametri modela:
    - `n_layer = 6` (broj slojeva)
    - `n_head = 6` (broj attention glava)
    - `n_embd = 384` (dimenzija embedding vektora)
    - `block_size = 256` (maksimalan broj tokena koje model može vidjeti odjednom)
    - `dropout = 0.2` (regularizacija za sprječavanje overfittinga)

- **train.py**
  - Trening modela na korpusu dobivenom iz *prepare_southpark.py*
  - Podržava checkpointing (`model_best.pt`, `model_last.pt`)
  - Koristi AdamW optimizator

- **sample.py**
  - Generiranje novih skripti na temelju istreniranog modela

## 2. Tehnologija
- Python 3.11
- PyTorch
- NumPy
- SentencePiece
- VS Code
- Git i GitHub

## 3. Pokretanje projekta
### 1. Kloniranje repozitorija
git clone https://github.com/TinUbrekic/zavrsni_projekt.git
cd zavrsni_projekt
### 2. Kreiranje i aktivacija virtualnog okruženja
python -m venv .venv
.\.venv\Scripts\activate  
### 3. Instalacija potrebnih paketa
pip install torch sentencepiece numpy
### 4. Obrada South Park skripti
python src/prepare_southpark.py --vocab_size 4000 --val_ratio 0.1
### 5. Treniranje modela
python src/train.py --block_size 256 --batch_size 12 --lr 1e-4 --max_steps 6000 --log_interval 50 --eval_interval 300
### 6. Generiranje teksta
python src/sample.py --out_len 1000

## 4. Primjer korištenja
### Početak treniranja
python src/train.py --block_size 256 --batch_size 12 --lr 1e-4 --max_steps 10000 --log_interval 50 --eval_interval 300
### Generiranje skripte
python src/sample.py --clean --prompt "CARTMAN:You guys seriously" --temperature 0.7 --top_k 50 --top_p 0.9 --stop_after_chars 1500

**Primjer outputa:**
STAN: Dude, why is Cartman acting so weird?  
KYLE: [the other boys rush up to the house] You're not gonna do it.  
KYLE: [whispers] I don't know how much is that?  
CARTMAN: Uh, I guess I think you've been a pretty good idea. [turns and walks away]  
STAN: "There.  
KYLE: You can't have to go to me again.  
STAN: Dude, it's just that was just a-fucker.  
STAN: Well, I have to hang on in. You're not gonna kick my ass.  
KYLE: Yeah, that's why you guys are?  
STAN: Yeah, what?  
STAN: No, dude, but I'm gonna get out of here and get some trouble.  
STAN: But then I'm gonna try to go home and get some more time, Kenny.  
STAN: You're not gonna take your butthole!  
KYLE: [being interviewed] Hey, you're not gonna get some food.  
KYLE: Dude, that's not cool! I know.  
CARTMAN: What's wrong with you?  
CARTMAN: You're gonna be at least a little boy who you have to be a little girl!  
CARTMAN: Oh, my God, I'm going to see you.  
