import math
import torch
import torch.nn as nn
import torch.nn.functional as F

#definiramo konfiguraciju GPT modela
class GPTConfig:
    def __init__(self, vocab_size, n_layer=6, n_head=6, n_embd=384, block_size=256, dropout=0.2):
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.block_size = block_size
        self.dropout = dropout

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0   #provjeravamo da je n_embd djeljiv s n_head
        self.n_head = config.n_head  #broj glava
        #definiramo "slojeve" koji generiraju key, query i value attention matrice
        self.key = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.query = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)   #"sloj" koji vraca rezultat attention matrice u originalni prostor embedanja
        #dropout slojevi za regularizaciju, jedan za attention matricu, drugi self.proj
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)
        mask = torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size)  #donjetrokutasta matrica koja sluzi da svaki token vidi sebe i prethodne ali ne i buduce
        self.register_buffer("mask", mask) #koristimo masku kao buffer

    def forward(self, x):
        B, T, C = x.size() #B batch size, T broj tokena, C dimenzija embedanja
        H = self.n_head #broj glava za attention
        head_dim = C // H
        k = self.key(x).view(B, T, H, head_dim).transpose(1, 2)   #mijenjamo osi da bi dobili (B, H, T, head_dim), radimo to za key, query i value
        q = self.query(x).view(B, T, H, head_dim).transpose(1, 2) 
        v = self.value(x).view(B, T, H, head_dim).transpose(1, 2) 
        att = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)     #racunamo attention score
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))  #mjesta gdje je maska=0 stavljamo na -inf
        att = F.softmax(att, dim=-1) #normaliziramo attention score po zadnjoj osi pomocu softmaxa
        att = self.attn_drop(att) #primjenjujemo dropout na attention matricu(radimo regularizaciju)
        y = att @ v  #racunamo weighted sum za svaki token                                           
        y = y.transpose(1, 2).contiguous().view(B, T, C)   #vracamo natrag dimenzije na (B, T, C)
        y = self.resid_drop(self.proj(y)) #vracamo rezultat u originalni prostor embedanja
        return y

class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.fc   = nn.Linear(config.n_embd, 4 * config.n_embd)  #prvi sloj, prosirujemo dimenziju embedanja na 4 puta vecu jer onda model ima veci kapacitet za ucenje
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd)  #drugi sloj, vracamo dimenziju embedanja na originalnu
        self.drop = nn.Dropout(config.dropout) #dropout sloj za regularizaciju(smanjuje overfiting)
    def forward(self, x):
        x = self.fc(x) #prolazimo kroz prvi sloj
        x = F.gelu(x) #koristimo aktivacijsku funkciju GELU
        x = self.proj(x) #prolazimo kroz drugi sloj
        x = self.drop(x) #primjenjujemo dropout
        return x

class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd) #prvi sloj za normalizaciju, stabilizira ucenje prije causal self-attention
        self.attn = CausalSelfAttention(config) #omogucujemo tokenima da gledaju samo sebe ili prethodne tokene
        self.ln2 = nn.LayerNorm(config.n_embd) #drugi sloj za normalizaciju
        self.mlp = MLP(config) #MLP sloj
    def forward(self, x):
        x = x + self.attn(self.ln1(x)) #normaliziramo sloj pa ga saljemo u casual-self-attention sloj i onda ga dodajemo natrag na ulaz
        x = x + self.mlp(self.ln2(x))  #normaliziramo sloj pa ga saljemo kroz MLP sloj i dodajemo ga natrag na ulaz
        return x

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd) #embeding sloj za tokene, pretvara id tokena u vektor
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd) #embeding sloj za poziciji, daje informaciju o poziciji svakog vektora u sekvenci
        self.drop = nn.Dropout(config.dropout) #dropout sloj za regularizaciju
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)]) #lista transformer blokova, zadano u konfiguraciji
        self.ln_f = nn.LayerNorm(config.n_embd) #zavrsni sloj za normalizaciju
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False) #sloj koji mapira embedanje natrag u prostor vokabulara
        self.apply(self._init_weights) #inicijalizacija tezine svih slojeva

    #koristimo normalnu distribuciju za inicijaliziranje tezine
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, "sequence length > block_size" #dohvacamo batch size i duljinu sekvence i gledamo da nije veca od maksimalne
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)  #generiramo pozicijske indekse za svaki token, oblika (1, T)
        x = self.tok_emb(idx) + self.pos_emb(pos) #zbrajamo token i pozicijske embedinge
        x = self.drop(x) #primjenjujemo dropout
        for block in self.blocks:
            x = block(x) #prolazimo kroz sve transformer blokove
        x = self.ln_f(x) #zavrsni sloj za normalizaciju
        logits = self.head(x) #sloj koji daje predikcije za svaki token u vokabularu
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) #racunamo loss
        return logits, loss
