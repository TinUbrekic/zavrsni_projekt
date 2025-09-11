U ovom projektu napravljen je jezični model koji za dani prompt generira scenarij za epizodu South Parka. 
Model je treniran na transkriptima epizoda, točnije na transkriptima prvih 17 seozna serije.
Model je temeljen na arhitekturi transformera, čime je omogućeno učinkovito učenje i generiranje teksta na temelju velikih količina podataka.

Podaci za projekt nabavljeni su sa: https://southpark.fandom.com/wiki/Portal:Scripts

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
- **Procesor** - Intel Core i5-8400 @ 2.89 GHz
- **RAM** - 16 GB
- **Grafička kartica** - Nvidia GTX 1050 Ti (nije korištena, model je treniran na CPU)
- **Vrijeme treniranja:** Model je trenirao 15000 koraka, batch size je bio 8, a block size 256. Ne početku je learning rate(LR) bio 1e-4, zatim je nakon 1500 koraka smanjen na 3e-5 i tako je bio do 4000 koraka. Od 4000 do 6500 koraka LR je bio 1e-5. Od 6500 do 10000 koraka je LR bio 3e-6, i od 10000 do 15000 je LR bio 1e-7. Vrijeme potrebno za ovo treniranje je bilo između 5 i 6 sati.

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

## 9. Clip gradient
Clip gradient je korišten u projektu kako bi se spriječilo da gradijenti tijekom treniranja postanu preveliki, tj. da model nebi napravio "prevelike korake" te da učenje ne postane nestabilno. Pomoću clip gradienta je norma gradijenta ograničena na maksimalnu vrijednost i time treniranje ostaje stabilno i loss se smanjuje postepeno. 

## 10. Usporedba modela
Tijekom treniranja su spremljena 2 modela. Prvi model je došao do 4500 koraka, dok je drugi model došao do 15000 koraka. 
Oba modela su pokrenuta na sljedećem promptu: python src\sample.py --prompt "STAN: Dude, why is Cartman acting so weird?" --temperature 0.7 --top_k 40 --top_p 0.9 --stop_after_chars 1000
Usporedba generiranja skripte 2 modela:
1. Model od 4500 koraka
   
   STAN: Dude, why is Cartman acting so weird? [the boys reach the door] CARTMAN: What?! KYLE: [walks off] You're going to get it! KYLE: I think of you'm sorry, and you got a good. CARTMAN: Yeah, that's okay, that's          what's like it's a little friend! KYLE: We've got to get it! KYLE: [Ssighs] Don't have you see how much more than to go to take this! KYLE: Dude, I can't wanna go home. I didn't think that's that was, Cartman. CARTMAN:    I want to do you. KYLE: No, it, I don't understand. You'll be fat, but I'm just... because I guess. CHEF: What's right now. I just don't tell you, that. KYLE: How? STAN: No, you want to talk about you guys? KYLE: No!      CARTMAN: No! STAN: That's you, I'm gonna have a great idea. KYLE: Okay. CARTMAN: Well, we're not so we're doing it're too! He's house, and Stan's mom is a little boy's house, but then, and the room. KYLE: Kyle, Kyle is    talking to the bushes. He goes back to see the floor. The boys are still ready to see, and Cartman, and Kenny's face. They'll find it
3. Model od 15000 koraka
   
   STAN: Dude, why is Cartman acting so weird? CARTMAN: No! I don't think it was really gonna be my best friend, but if you don't need any idea what you're doing?! KYLE: What?! [Kyle and Kyle laugh at the table] Hey, you     guys, I just wanted to make it, Cartman. I'm gonna tell you what you're talking about. I mean, I have a little fantastic game, and you know. CARTMAN: [hangs up and starts writing something] You know, Kenny? CARTMAN: I     don't think about it's gonna make it. It's like that, Kyle, I'll get away. CARTMAN: [turns and walks away] Oh, dude, I'm gonna be able to go. KYLE: You're not gonna take a pet of my mom. STAN: Dude, I don't wanna get a    lot, you guys. We're in the living room. CARTMAN: Yeah, we're gonna play with you. CARTMAN: No, I can't go home. STAN: You guys, we're gonna go to your house. CARTMAN: Hey, we're gonna find out! [points to Kyle] STAN:     What are you doing? KYLE: Dude, this is awesome! I have been trying to take out of my moms. CARTMAN: [to Stan] Oh, no, what are

## 11. Usporedba modela i stvarne skripte
Usporedit ćemo generiranje modela i stvarnu skriptu tako što ćemo uzeti rečenicu iz skripte i dati modelu kao prompt, te vidjeti koliko je slično pravoj skripti.
1. prompt: STAN: That's my new dog Sparky. He followed me to the bus stop.
   
   Model: STAN: That's my new dog Sparky. He followed me to the bus stop. STAN: [Gretched.] I don't know what I'm going on. KYLE: We'll do that! KYLE: What? KYLE: You have to do, dude. He's gonna do with you. STAN: Dude,            don't let us go back to your room and I'm not gonna kill him. Cartman turns around and walks away. CARTMAN: What are we supposed to do?! CARTMAN: Dude, you guys, I think it's a big thing. STAN: [To Kyle.] Well,            I guess we can't be for all of you guys. I've gotta play, but he's gonna be cool. KYLE: Yeah, we're not gonna tryin' to play with him. STAN: Yeah, but, yeah, I know, I'll just go. CARTMAN: Yeah. I'm just asking            you. STAN: Yeah, I'll be here at Kenny. CARTMAN: What are you doing? CARTMAN: Yeah, because... CARTMAN: I'm gonna start watching television. CARTMAN: Why the hell would you hear that? CARTMAN: Yeah, but we're              gonna be fine. We'll go find out and go get to your house! CARTMAN: [Feeling.] Oh, I'm gonna have to watch it. CARTMAN: [Frugs.] I'm sorry, Kyle. KYLE: You're not gonna stay here! We're gonna get to do it.                 CARTMAN: Don't worry, but we're going to do it. STAN: Yeah, but you're going to be a little bit more time. KYLE: You can't make me sick, Kyle. CARTMAN: Okay
   
   Skripta: STAN: That's my new dog Sparky. He followed me to the bus stop. KYLE: Wow! Cool! STAN: Good boy, Sparky. Who's my best buddy? Who's your boy? Who's your buddy? CARTMAN: Eh. You're making me sick, dude. STAN:               He's part Doberman and part wolf. He's the toughest dog on the mountain. CARTMAN: No way. Everybody knows that Sylvester is the toughest dog in South Park. SYLVESTER: Arrrrrr. STAN: He's not meaner than                    Sparky. CARTMAN: Oh yeah? Let's see. Hey, Sylvester!
            Sylvester comes over. STAN: Sparky'll kick his ass. CARTMAN: I'll put a dollar on Sylvester. KYLE: You're on, dude!
            Sylvester starts after Sparky. Both dogs growl. STAN: That's it, Sparky! Kick his ass!
            Sparky lunges after Sylvester and out of view. Aggressive panting can be heard. The boys stand there, shocked. CARTMAN Huh. He's doing something to his ass. He's not kicking his ass, but he's definitely doing              something to his ass. STAN: Sparky! Bad dog!
   
3. prompt: CARTMAN: Mr. Garrison, what's Vietnam?
   
   Model: STAN: Mr. Garrison, what's Vietnam? MR._GARRISON: Well, you can't believe it was the newestest thing! You have to take that money to the children in front of the South Park, m'kay? MR._GARRISON: Uh-kay,                    children, I think we need to go to the school and start to school tomorrow. MR._MACKEY: How are you gonna make a lot of your little friends? [the kids look at each other] MR._GARRISON: We don't know what to say,           Mr. Garrison? MR._GARRISON: Yeah, yeah. Now, I hope that you all do, Mr. Garrison. MR._GARRISON: Oh, it's not true. MR._GARRISON: Well, children, we can see why I do something today, but I'm just, uh, "Gett's a            very difficyer." CARTMAN: Hey, come on, children. MR._MACKEY: You can't stop being dying, m'kay. MR._GARRISON: That's great. MR._GARRISON: You can't let me see what you want, Mr. Hat? MR._GARRISON: Yeah, I can't           go to hell to hell. STAN: He's so important for me. CARTMAN: Well, if you want to go to your mother. MR._GARRISON: Now, why does you get out of the school bus? MR._GARRISON: Well, I have to make sure to be                 learning the lawy, but that I don't know what that-can-do-sucker! MR._GARRISON: How much do you think you should be all about your son? MR._GARRISON: I know
   
   Skripta: CARTMAN: Mr. Garrison, what's Vietnam? MR. GARRISON: "What's Vietnam?" A question a child might ask, but not a childish question. Heheheheheheheheheh. Children, for the next few days, we'll be learning all                 about Vietnam. Chances are that somebody in your own lives was affected by this incredible war. MR. HAT: That's right, Mr. Garrison. The Vietnam War was sticky and icky. Kyle raises his hand. KYLE: Mr.                     Garrison, were you in Vietnam? Mr. Garrison frowns as he seems to recall an incident in that war. An injured man is dragged to a helicopter as fresh troops wait to replace him. INJURED MAN: Aaawwwgh...                     Aaawwwgh… LEADER: Come on, men! Let's go! Jump out of the chopper!
            The class is waiting. Now Mr. Garrison is smiling as the next scene begins. A group of men is seen disrobed and bathing. At the end of it, he laughs to himself. MAN 1: Who's next to take a shower? MAN 2: Me-               ee! MAN 3: I am. MAN 1: You just took one last week, silly. MAN 2:Oh, where can I hide this big pipe? MR. GARRISON: No, I wasn't in Vietnam, but sometimes, I like to pretend I was. Anyway, children, I'm going              to assign you all a paper. CLASS: Awww..! CARMAN: Son of a b****.
            MR. GARRISON: I want you all to find somebody in your own life who was in Vietnam and interview them about it. CLYDE: What if we don't know anybody who was in Vietnam? MR. GARRISON: Then you get an F, fail the             third grade, and have to get a job cleaning septic tanks to support your drug habit. CLYDE: Oh. STAN: Dude, my uncle Jimbo was in Vietnam. KYLE: Hey, yeah! He and Ned do that stupid TV show. Back to Huntin'                and Killin, in a special segment... JIMBO: And now, time for Jimbo's Mysteries of the Unexplained. The letters are wavy and uneven. The boys come to see the taping.

3. prompt: STAN: Boy band?
   
      Model: STAN: Boy band? CARTMAN: No, you guys, I guess we'll try and get away with it. The boys look at each other. STAN: We're not gonna have a lot more time. CARTMAN: Yeah! [Kenny gets up, then walks away] You've                 got it, Kenny. KYLE: You can't take it to the bus. CARTMAN: Well, it's pretty cool. STAN: Well, we're gonna get our parents. We're gonna have to go to the bus. STAN: We'll have to get to play Truth or Det.                 CARTMAN: You can't believe it. KYLE: Dude, dude, what are you doing? KYLE: Cartman, what do you mean? KYLE: Well, let's see, we can't go with the clubhouse. KYLE: I'm going to go to hell out of here. STAN:                 What?! I don't wanna play with the boys. CARTMAN: You guys, you know what you need to know, but I'll be okay. KYLE: But what if we're all gonna do is that. STAN: Hey, you guys, Kyle. CARTMAN: What's going                  on? CARTMAN: How much? CARTMAN: What? STAN: Yeah, dude. We're gonna watch this game. [walks away] STAN: Well, I don't want to go, dude. STAN: Huh? KYLE: What? KYLE: Come on, you guys! CARTMAN: That's a                     minute. Now, maybe we have to get some of a bit of time. STAN: Dude, what are we supposed to go. KYLE: He's just here! STAN: [Stan and Kenny are still waiting for a few seconds

      Skripta: STAN: Boy band? CARTMAN: Boy band. KYLE: I'm not being in any [crosses his arms over his chest] f**gy boy band! CARTMAN: There's nothing f**gy about $10 million, asshole! This was a message from God! STAN:                 Dude, we don't have any musical talent. CARTMAN: That didn't stop any of the other boy bands, dumbass! [pulls out a tape from his back pocket] I've got prerecorded music we can sing to, just like they do.                  All we need to do is practice our choreography over and over and over! KENNY: (That sounds totally fuckin' stupid.) CARTMAN: Shut up, Kenny. And then, I know I can get us a gig at the South Park Mall.                      [intense] So everybody get in a straight line, we're gonna listen to a song from the top, and take it-- KYLE: Wait a minute. There's only four of us. CARTMAN: So? KYLE: So, all boy bands have five members.                 CARTMAN: What? YLE: 'N Sync, Backstreet Boys, New Kids on the Block. All had five members. TAN: He's right. ARTMAN: [throws down the tape] Dammit! Okay, okay okayokayokay. We'll put off practice for now,                   and hold auditions for a fifth member. Get the word out that auditions will be tomorrow morning!
                Marsh residence, night. The family is gathered for dinner, enjoying ham and turkey SHARON: Did you have fun at Eric's house today, Stanley? STAN: Well, I guess. SHARON: What did you do? STAN: Well, Cartman                wants to start a boy band, so we're gonna rehearse and then try to perform at the South Park Mall. [Randy reacts, and Shelley takes notice] SHARON: Oh well, that sounds nice. RANDY: [to Sharon, angrily] No,                it does NOT sound nice! [to Stan] Stanley, you are gonna have no part in that boy band. STAN: Well but, Dad, all my friends are doing it. RANDY: [rises] If all your friends jumped off a cliff, [jabs his                     finger at Stan] would you do that too? STAN: [voice shaking] Cartman says we can make $10 million. RANDY: You are not gonna be in a boy band, Stanley! AND THAT IS FINAL! [storms out] SHELLEY: Geez, what's                 up Dad's ass?
                Cartman residence, next morning, "BOY BAND AUDITIONS TODAY!!" Music begins. Inside, Butters is singing his song in front of the sofa. 
