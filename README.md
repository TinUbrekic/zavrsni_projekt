## 1. Arhitektura
Projekt se sastoji od nekoliko dijelova:
- **prepare_southpark.py**
  - Učitava sve .txt epizode iz data/raw/, čisti scenske upute i normalizira dijalog u oblik IME: rečenica.
  - Trenirani tokenizator: SentencePiece BPE.
  - Cijeli korpus se tokenizira u uint32 ID-jeve i dijeli na train.bin i val.bin.

- **model.py**
  - Ulaz/izlaz  
    - Ulaz: niz token ID-jeva dimenzije (B, T) gdje je B=batch_size, T=block_size (u modelu je 256).  
    - Izlaz: logiti dimenzije (B, T, vocab_size) – distribucija nad idućim tokenom za svaku poziciju.
  - Slojevi
    1. Token embedding - pretvara ID u vektor.
    2. Pozicijski embedding - informacija o poziciji u sekvenci.
    3. Transformer blokovi (6 slojeva):  
       Svaki blok se sastoji od dva glavna dijela.  
       - Multi-Head Self-Attention: ulazne reprezentacije se dijele na više "glava" pažnje. Svaka glava uči gledati na različite odnose među tokenima u sekvenci, pri čemu se koristi kauzalna maska da bi model na svakoj             poziciji mogao gledati samo unatrag (na prethodne tokene), a ne unaprijed. Nakon toga se svi rezultati spajaju i vraćaju u prostor iste dimenzije.  
       - Feed-Forward mreža: nakon pažnje, svaka pozicija prolazi kroz malu dvostruku neuronsku mrežu koja transformira vektor i daje mu veću izražajnost.  

       Oba dijela (i pažnja i feed-forward mreža) imaju dodane rezidualne veze (koje pomažu u stabilnosti učenja), normalizaciju slojeva i dropout radi regularizacije i sprječavanja overfittinga.

    4. Izlazni sloj: nakon svih blokova, postoji završni linearni sloj koji svaku reprezentaciju prevodi u vjerojatnosnu distribuciju nad cijelim vokabularom, tj. određuje koji je sljedeći token najvjerojatniji.
 - Parametri modela
  - n_layer = 6 - broj slojeva   
  - n_head = 6 - broj attention glava   
  - n_embd = 384 - dimenzija vektora ugradnje  
  - block_size = 256 - maksimalna duljina konteksta u tokenima  
  - dropout = 0.2 - regularizacija   


- **train.py**
 - Batching:  
    - Iz train.bin uzimaju se maksimalne duljine sekvence dužine block_size.  
    - Ulaz x su prvih T tokena, cilj y je x pomaknut za 1 token (predviđanje sljedećeg tokena).
  - Cilj (loss): Koristi se Cross-Entropy Loss.
  - Optimizator: Koristi se AdamW optimizator.
  - Praćenje i validacija:  
    - loss se ispisuje svakih 50 koraka  
    - val_loss se računa na val_loader periodično (svakih 300 koraka) uz @torch.no_grad().  
    - Sprema se model-best.pt (najniži val_loss) i model-last.pt (zadnji korak), zajedno s konfiguracijom i korakom (step) radi nastavka treniranja.
  - Sekvenca za učenje: block_size = 256 definira maksimalni broj tokena iz povijesti koje model vidi pri predikciji.

- **sample.py**
  - Učitava checkpoint i meta.pkl, kreira model i enkoder/dekoder iz SentencePiece modela.
  - Dekodiranje: podržan je sampling s temperature, top-k i top-p varijablama i laganim repetition penalty.  
    Generiranje se zaustavlja po uvjetima: dosegnuta duljina (stop_after_chars) i/ili heuristike (prazan redak, detekcija „END/CUT TO/SCENE” i slično).
  - Primjer:  
    python src/sample.py --prompt "CARTMAN: You guys, seriously." --temperature 0.7 --top_k 50 --top_p 0.9 --stop_after_chars 1500

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
python src/sample.py --prompt "STAN: Dude, why is Cartman acting so weird?" --temperature 0.7 --top_k 40 --top_p 0.9 --stop_after_chars 1000

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

## 5. Specifikacije računala
**Procesor** - Intel Core i5-8400 @ 2.89 GHz
**RAM** - 16 GB
**Grafička kartica** - Nvidia GTX 1050 Ti (nije korištena, model je treniran na CPU)
###Vrijeme treniranja: Model je trenirao 15000 koraka, batch size je bio 8, a block size 256. Ne početku je learning rate(LR) bio 1e-4, zatim je nakon 1500 koraka smanjen na 3e-5 i tako je bio do 4000 koraka. Od 4000 do 6500 koraka LR je bio 1e-5. Od 6500 do 10000 koraka je LR bio 3e-6, i od 10000 do 15000 je LR bio 1e-7. Vrijeme potrebno za ovo treniranje je bilo između 5 i 6 sati.

## 6. Rezultati treniranja
Model je trenirao na korpusu od ~1.4 milijuna tokena(221 epizoda South Parka).
Chekpointi u kojima je spremljen model(model_best i model_last) su veličine oko 160MB.
Najbolja validacijska loss vrijednost koja je postignuta tijekom treniranja je ~4.0.

## 7. Treniranje modela
Kao što je već navedeno model je trenirao na korpusu od ~1.4 milijuna tokena. Proces treniranja može se opisati kroz sljedeće korake:
- **Batch**
  Podaci su podjeljeni u sekvence duljine 256(block_size).
  Jedan batch se sastoji od batch_size=8 takvih sekvenci.
- **Forward Pass**
  Za svaku sekvencu model pokušava predvidjeti sljedeći token.
  Batch na ulazu ima x, tj. ulaze tokene, i y, tj. ciljane tokene pomaknute za 1 udesno. Model želi minimizirati razliku između predikcija i stvarnih vrijednosti.
- **Loss funkcija**
  Korišten je Cross-Entropy Loss između distribucije vjerojatnosti predviđenih tokena i stvarnih tokena.
- **Optimizator**
  Korišten je AdamW optimizator s početnim LR = 1e-4 i weight_decay = 0.04. Tijekom treniranja LR je smanjivan na prethodno navedeni način kako bi se smanjila mogućnost overfittinga.
- **Epochs**
  S obzirom da je korpus dosta velik, treniranje je praćeno brojem koraka, a ne punim epohama. Broj koraka je bio 15000 i bilo je nekoliko faza treniranja. U prvoj fazi model je prošao 1500 koraka, zatim je u drugoj         prošao   4000 koraka. U trećoj fazi je išao do 6500 koraka, u četvrtoj do 10000 koraka i u posljednoj do 15000 koraka. Svakih 50 koraka se ispisivao loss, te se svakih 300 koraka računao val_loss, koji se onda spremao u   checkpointe.
- **Checkpoints**
  Tijekom treniranja model se spremao u 2 checkpointa:
  - model_last.pt - zadnji korak treniranja
  - model_best.pt - najbolja validacijska loss vrijednost tijekom treniranja
Nakon završetka treniranja, model je generirao razgovor koji se nalazi u odjeljku **Primjer outputa**.

## 8. Tokenizer
U ovom projektu korišten je SentencePiece s algoritmom BPE za tokenizaciju korpusa South Park epizoda. Tokenizator služi da bi pretvarao tekstualne skripte u numeričke ID-ove koje model može obraditi, zatim za održavanje vokabulara(koji je jednak 4000), čime se smanjuje veličina embedding matrice i ubrzava treniranje.
**Rad tokenizatora**:
- Tijekom pripreme podataka tokenizeruči vokabular korpusa.
- Tekst epizoda se pretvara u sekvencu ID-ova i sprema u .bin datoteke za treniranje i validaciju.
- Tijekom generiranja izlazni ID-ovi se pretvaraju natrag u tekst pomoću SentencePiece modela.
