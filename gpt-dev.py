import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eval_interval = 300
block_size = 8
batch_size = 32
max_iters = 3000
learning_rate = 1e-2
eval_iters = 200
enable_verbose = False

# download data from: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
# output: input_shakespeare.txt
#!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

with open('input_shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read() # should be simple plain text file

if enable_verbose:
    print('input length in chars:', len(text))

# Build Vocab
vocab = sorted(list(set(text)))
vocab_size = len(vocab)
if enable_verbose:
    print('----')
    print('vocab length:', vocab_size)
    print('vocab: ', ''.join(vocab))

stoi = {c: i for i, c in enumerate(vocab)}
itos = {i: c for i, c in enumerate(vocab)}
encode = lambda x: [stoi[c] for c in x]
decode = lambda x: ''.join([itos[c] for c in x])

# Build datasets
data = encode(text)

n_split = int(0.9 * len(data))
train_data = data[:n_split]
val_data = data[n_split:]

x = train_data[:block_size]
y = train_data[1:block_size+1]

if enable_verbose:
    print('----')
    print('Example of a training sample:')
    for i in range(block_size):
        print(x[:i+1], '-->', y[i])

# seed random number generator for torch
torch.manual_seed(1337)

def get_batch(split):
    data = train_data if split == 'train' else val_data if split == 'val' else None
    
    if data is None:
        raise ValueError('split must be either train or val')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.tensor([data[i:i+block_size] for i in ix]).to(device)
    y = torch.tensor([data[i+1:i+block_size+1] for i in ix]).to(device)
    return x, y
    
@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[i] = loss.item()
        out[split] = loss.mean()
    model.train()
    return out

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        
    def forward(self, x, targets=None):
        x = self.embed(x)
        logits = x

        loss = None
        if targets != None:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        if targets is None:
            return logits
        else:
            return logits, loss
    
    def generate(self, x, n):
        # input is B x T
        for i in range(n):
            logits = self(x) # B x T x C
            # only use last prediction
            logits = logits[:, -1, :] # B x C
            # sample from distribution
            probs = F.softmax(logits, dim=-1) # B x C
            # sample next token
            idx_next = torch.multinomial(probs, num_samples=1) # B x 1
            x = torch.cat([x, idx_next], dim=1) # B x (T+1)

        return x
    
n_embed = vocab_size
model = BigramLanguageModel(vocab_size, n_embed).to(device)
if enable_verbose:
    print('----')
    print('Model Desc:', model)

    # Init predition
    idx = torch.zeros(1, 1, dtype=torch.long).to(device)
    print('----')
    print('Initial predictions:', idx)
    [print(p) for p in [decode(pred) for pred in model.generate(idx, 100).tolist()]]

# Train model
print('----')
print('Training model...')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for step in range(max_iters):
    
    if step % eval_interval == 0:
        eval_loss = estimate_loss()
        print(f'{step=}, train: {eval_loss["train"]}, val: {eval_loss["val"]}')

    xb, yb = get_batch('train')
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    

# Eval model
print('----')
print('Evaluating model...')
print(decode(model.generate(torch.zeros((1,1), device=device, dtype=torch.long), 200)[0].tolist()))

