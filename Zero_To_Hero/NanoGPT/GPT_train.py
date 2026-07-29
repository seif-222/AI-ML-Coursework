import os
import math
import time
import inspect
import tiktoken
import numpy as np

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from functools import partial
from HellaSwag import render_example, iterate_examples

# from hellaswag import render_example, iterate_examples

###_________________________ DISTRIBUTED TRAINING _______________________________________

ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    assert torch.cuda.is_available(), f'There is not cude for the device to run ddp'
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')


###_________________________ HYPERPARAMETERS _______________________________________

@dataclass
class GPT_config:
    ### for model
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12
    vocab_size: int = 50304 # Just a number that is divisible by 2 more times that the standard 50257
    block_size: int = 1024
    ### for data
    bs: int = 64
    tokenizer: str = 'gpt2'
    tot_bs_for_grad_accum: int = 524288      # 2**19
    # for the training shards
    data_root: str = "edu_fineweb10B"
    ### for optim sched
    max_lr: float = 6e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 715
    max_steps: int = 19073
    ### for optim
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.1
    ### for training & Validation
    training_steps: int = 19073              # = max_steps
    val_after_step: int = 250
    val_loss_accum_steps: int = 20
    ### CheckPoint
    checkpoint_after_steps: int = 5000
    ### device
    device: str = device
    ### Multi-Device Training
    process_rank: int = ddp_rank
    num_processes: int = ddp_world_size
    ### Options
    # validaiton
    validation: bool = True
    # sampling
    model_sampling: bool = True
    num_sequence: int = 4
    max_length: int = 32
    #compile
    use_compile: bool = False                # note: compile breaks HellaSwag/gen

###_________________________ CREATING THE MODEL _______________________________________

def batch_head(x, n_head):
  b,t,c = x.shape
  x = x.reshape(b, t, n_head,-1)
  return x.transpose(1,2).reshape(b*n_head, t, -1)

def head_batch(x, n_head):
    bn, t, c = x.shape
    x = x.reshape(-1, n_head, t, c)
    return x.transpose(1, 2).reshape(-1, t, n_head * c)


class CausalMultiHeadAttention(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.n_head = config.n_head
    self.n_embd = config.n_embd
    assert self.n_embd % config.n_head == 0, "number of heads must be a multiple of n_head"
    self.head_sz = config.n_embd // config.n_head
    self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd)
    self.c_proj = nn.Linear(self.n_embd,self.n_embd)
    self.c_proj.INIT_SPECIAL_STD = 1
    # self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size)))    # -> because we switched to Flash attention

  def forward(self, x): # x -> (B, T, E)
    b,t,c = x.shape
    x = self.c_attn(x)
    q, k, v = torch.chunk(x, 3, dim=-1) # we get the qkv first before batch_head because this is the order openai used(weights optimized for it), but reversing it when training a model from scratch is fine
    q = batch_head(q, self.n_head) # (B, T, E)
    k = batch_head(k, self.n_head) # (B, T, E)
    v = batch_head(v, self.n_head) # (B, T, E)
    # wei = q @ k.transpose(-1,-2) * self.head_sz**-0.5 # Scaling, shape (B, T, T)
    # wei = wei.masked_fill(self.bias[:t,:t]==0, float('-inf'))
    # wei = F.softmax(wei, -1)
    # wei = wei @ v
    wei = F.scaled_dot_product_attention(q, k, v, is_causal=True) # is_casual=True -> causal mask is the triangular mask we built manually with torch.tril + masked_fill.
    wei = head_batch(wei, self.n_head)
    return self.c_proj(wei)


class MLP(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
    self.gelu    = nn.GELU(approximate='tanh')
    self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
    self.c_proj.INIT_SPECIAL_STD = 1
  def forward(self, x):
    return self.c_proj(self.gelu(self.c_fc(x)))



class Block(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.attn = CausalMultiHeadAttention(config)
    self.mlp = MLP(config)
    self.ln_1 = nn.LayerNorm(config.n_embd)
    self.ln_2 = nn.LayerNorm(config.n_embd)
  def forward(self, x):
    x = x + self.attn(self.ln_1(x))
    return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.transformer = nn.ModuleDict(dict(
      wpe = nn.Embedding(config.block_size, config.n_embd),
      wte = nn.Embedding(config.vocab_size, config.n_embd),
      h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
      ln_f = nn.LayerNorm(config.n_embd)
    ))
    self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    ## MAKE the last Linear later the same as the embedding layer
    self.lm_head.weight = self.transformer.wte.weight

    ## APPLY Initialization
    self.apply(self._initalization)

  ## Make Initialization function
  def _initalization(self, module):
      if isinstance(module, nn.Linear):
          std = 0.02
          if hasattr(module, 'INIT_SPECIAL_STD'):  std *= (2 * self.config.n_layer) ** -0.5
          nn.init.normal_(module.weight,mean=0., std=std) # torch.nn
          if module.bias is not None:  nn.init.zeros_(module.bias)
      elif isinstance(module, nn.Embedding):
          nn.init.normal_(module.weight, mean=0., std=0.02)

  def forward(self, x, targets=None): # x -> (B, Tokens)
    B, T = x.shape
    assert T <= self.config.block_size, f'The Context Exceeds the block size {T} > {self.config.block_size}'

    x = self.transformer.wpe(torch.arange(T, device=x.device)) + self.transformer.wte(x) # x -> (B, tokens, embs)\

    for layer in self.transformer.h: x = layer(x)

    x = self.transformer.ln_f(x)

    logits = self.lm_head(x)
    loss = None
    if targets is not None:  loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))
    return logits, loss

  @classmethod
  def from_pretrained(cls, model_name): #  cls -> the class it self can be called -> cls()
    assert model_name in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}, f"{model_name} is not Valid" # make a set so that it is faster O(1)
    from transformers import GPT2LMHeadModel
    print(f'loading {model_name}')
    configs ={
        'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
        'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
        'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
        'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
    }[model_name]
    configs['vocab_size'] = 50257 # same in all of them
    configs['block_size'] = 1024 # same in all of them

    model_config = GPT_config(**configs)
    model = cls(model_config)
    sd = model.state_dict()
    sd_keys = sd.keys()
    sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

    model_hf = GPT2LMHeadModel.from_pretrained(model_name) # load the hugging face model
    sd_hf = model_hf.state_dict()
    sd_keys_hf = sd_hf.keys()
    sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
    sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
    transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight'] # openai checkpoints use a "Conv1D" module, but we only want to use a  Linear so we have to transpose these weights when we import them

    assert len(sd_keys) == len(sd_keys_hf), f'Keys mismatch : {len(sd_keys_hf)} != {len(sd_keys)}'
    for k in sd_keys_hf:
        if any(k.endswith(w) for w in transposed):
            assert sd_hf[k].shape[::-1] == sd[k].shape, f'shape mismatch Transposed if Conv-> {sd_hf[k].shape} != {sd[k].shape} at {k}'   # [::-1] reverse sequence e.g. (768, 3072) to (3072, 768)
            with torch.no_grad():
                sd[k].copy_(sd_hf[k].T)
        else:
            assert sd_hf[k].shape == sd[k].shape , f'shape mismatch {sd_hf[k].shape} != {sd[k].shape} at {k}'
            with torch.no_grad():
                sd[k].copy_(sd_hf[k])

    return model

  def optimizers_config(self, device_type):
      # Take all the requires_grad parameters - this 2 lines can be collapsed into one line but we make it in 2 for clearner steps -
      params_dict = {pn: p for pn, p in self.named_parameters()}
      params_dict = {pn: p for pn, p in params_dict.items() if p.requires_grad}

      # Decay only when <= 2 dims, dims < 2 are the 1D tensors -> biases and LayerNorm's, It doesn't make sense to decay them
      decay_params = [p for pn, p in params_dict.items() if p.dim() >= 2]
      no_decay_params = [p for pn, p in params_dict.items() if p.dim() < 2]

      # create optim group
      optim_group = [
          {'params': decay_params, 'weight_decay': self.config.weight_decay},
          {'params': no_decay_params, 'weight_decay': 0.},
      ]
      # count the params of each group
      num_decay_params = sum(p.numel() for p in decay_params)
      num_no_decay_params = sum(p.numel() for p in no_decay_params)

      # Print it
      if master_process:
          print(f'Number of the decayed Tensors : {len(decay_params)} |  Number of the decayed params : {num_decay_params:,}, ')
          print(f'Number of the decayed Tensors : {len(no_decay_params)} |  Number of no_decay_param: {num_no_decay_params:,}')

      # See if we can make fused Adam
      fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
      use_fuse = fused_available and device_type == 'cuda'
      if master_process:  print(f'Using fused AdamW: {use_fuse}')
      # Make the optim and also add the Betas and the eps that openai used in the training
      optimizer = torch.optim.AdamW(optim_group, lr=self.config.max_lr, betas=(self.config.beta1, self.config.beta2), eps=self.config.eps, fused=use_fuse)
      return optimizer


###______________________________________ MAKE A DATALOADER ________________________________

def load_tokens(filename):
    np_loaded = np.load(filename)
    np_loaded = np_loaded.astype(np.int32) # convert uint16 to int32 before cast to long, otherwise pytorch doesn't like it
    return torch.tensor(np_loaded, dtype=torch.long)




# Make DL lite
class DL_lite:
    def __init__(self, config, split):
        assert split in {'Train', 'Val'}, 'Split must be one of "Train" or "Val"'
        # store attr
        self.block_size= config.block_size
        self.process_rank = config.process_rank
        self.num_processes = config.num_processes
        self.bs = config.bs
        assert config.tot_bs_for_grad_accum % (self.bs * config.num_processes) == 0, f' Total_Batch_size: {config.tot_bs_for_grad_accum} is not divisible by Batch_size: {self.bs} * Num_processes: {self.num_processes}'
        grad_accum = bool(config.tot_bs_for_grad_accum)
        self.tot_mini_batches = config.tot_bs_for_grad_accum // (self.bs * self.block_size * self.num_processes) if grad_accum else 1
        # Loading the data
        data_root = config.data_root
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        self.shards = [os.path.join(data_root, s) for s in shards]
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process: print(f'Found Shards =  {len(self.shards)} | Split = {split} | Total Batch Size = {config.tot_bs_for_grad_accum} | Grad Accum = {grad_accum} | Number Mini Batches = {self.tot_mini_batches} | Num_processes: {self.num_processes}')
        # Make a trackers
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.tr = self.bs * self.block_size * self.process_rank   # Tracking position

    def after_batch(self):
        tokens = self.tokens[self.tr : self.tr + self.bs * self.block_size + 1 ]
        self.tr += self.bs * self.block_size * self.num_processes
        x = tokens[:-1].view(self.bs, -1)
        y = tokens[1:].view(self.bs, -1)
        if (self.bs * self.block_size * self.num_processes  + 1 + self.tr) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards) # So that we advance to the next shard and if we finish the shards we loop again because of -> %
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.tr = self.bs * self.block_size * self.process_rank
        return x, y

###_________________________________ HELLASWAG FUNCTION _______________written in HellaSwag.py________________________________


def get_most_likely_row(tokens, mask, logits):
    shift_logits = (logits[:, :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_tokens = shift_tokens.view(-1)
    loss = F.cross_entropy(shift_logits, shift_tokens, reduction='none')
    reshaped_loss = loss.view(tokens.size(0), -1)
    shift_mask = mask[..., 1:].contiguous()
    masked_shift_loss = reshaped_loss * shift_mask
    sum_loss = torch.sum(masked_shift_loss, dim=-1)
    avg_loss = sum_loss / shift_mask.sum(-1)
    norm_preds = torch.argmin(avg_loss).item()

    return norm_preds

###_________________________________ LR SCHED _______________________________________________

# Creating a LR sched
def get_lr(step, config):
    min_lr = config.max_lr * config.min_lr_ratio
    if step < config.warmup_steps: return config.max_lr/config.warmup_steps * (step+1)
    if step > config.max_steps : return min_lr
    decay_ratio = (step-config.warmup_steps)/(config.max_steps-config.warmup_steps) # make a ratio between 0 and 1 that represent the current section in the cos
    assert 0 <= decay_ratio <= 1, f'There is something wrong with max steps:{config.max_steps}, step:{step}, warmup_steps:{config.warmup_steps}'
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (config.max_lr - min_lr)


###_____________________________________  INSTANCES  ______________________________________

# It a PyTorch function that speeds up float32 matrix multiplications on compatible NVIDIA GPUs by trading off a small amount of numerical precision for significant performance gains.
torch.set_float32_matmul_precision('high')

# config
config = GPT_config()

# make device_type
device_type = "cuda" if config.device.startswith("cuda") else "cpu" # just to use it at the autocast, etc... and make 'cuda:3' -> 'cuda', 'cuda' -> 'cuda' ,etc...

# encoder
enc = tiktoken.get_encoding(config.tokenizer)

# model
model = GPT(config)
model.to(config.device)
if config.use_compile: model = torch.compile(model) # compiles the model and makes kernel fusion for the operations
if ddp :  model = DDP(model, device_ids=[ddp_local_rank]) # Forward pass / training step → use model (the DDP wrapper) — this is what makes multi-GPU synchronization work
raw_model = model.module if ddp else model  # always contains the "raw" unwrapped model / -> Anything else (custom methods, saving checkpoints, accessing .config) → use raw_model — because DDP's wrapper doesn't expose your class's custom stuff directly

# lr getter fn
lr_getter = partial(get_lr, config=config)

# data
train_dl = DL_lite(config, 'Train')
val_dl = DL_lite(config, 'Val')

# optim
optimizer = raw_model.optimizers_config(device_type)

# make a logging file
log_dir = 'log'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'log.txt')
with open(log_file, 'w') as f: # open for writing to clear the file
    pass

##_____________________________________  TRAINING  ______________________________________


# Training Loop
for step in range(config.training_steps):
    start = time.time()
    last_step = (step == config.training_steps - 1)

    # Validation
    if (step % config.val_after_step == 0 or last_step) and (config.validation) :
        model.eval()
        with torch.no_grad():
            val_accum_loss = 0.
            for _ in range(config.val_loss_accum_steps):
                x, y = val_dl.after_batch()
                x, y = x.to(config.device), y.to(config.device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss /= config.val_loss_accum_steps
                val_accum_loss += loss.detach()
        if ddp: dist.all_reduce(val_accum_loss, op=dist.ReduceOp.AVG)
        val_dl.reset()
        if master_process:
            print(f"Validation loss: {val_accum_loss.item():.4f}")
            with open(log_file, "a") as f: f.write(f"{step} val {val_accum_loss.item():.4f}\n")

            # Save checkpoints for the model
            if (step > 0) and (step % config.checkpoint_after_steps == 0 or last_step):
                checkpoint_path = os.path.join(log_dir, f'model_checkpoint_step_{step:05d}.pt')
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config, # that is the stored config in the model object as an attr
                    'step': step,
                    'val_loss': val_accum_loss.item()
                }
                torch.save(checkpoint, checkpoint_path)


    # Model Sampling
    if step % config.val_after_step == 0 and step > 0 and config.model_sampling and (not config.use_compile):
        model.eval()
        tokens = enc.encode('I am crazy man,')
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.repeat(config.num_sequence, 1)
        x_gen = tokens.to(config.device)

        while x_gen.shape[1] < config.max_length:
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x_gen)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, k=50, dim=-1)  # take the highest 50 probs
                ix = torch.multinomial(topk_probs, num_samples=1)  # take one sample
                ids = torch.gather(topk_indices, dim=-1, index=ix)
                x_gen = torch.cat((x_gen, ids), dim=1)

        for i in range(config.num_sequence):
            decoded = enc.decode(x_gen[i, :config.max_length].tolist())
            print(f'Rank: {config.process_rank} | Sample{i + 1}: {decoded} ')

    # HellaSwag
    # once in a while evaluate hellaswag
    if (step % config.val_after_step == 0 or last_step) and (not config.use_compile):
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # only process examples where i % ddp_world_size == ddp_rank
            if i % ddp_world_size != ddp_rank:  continue
            # render the example into tokens and labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        # reduce the stats across all processes
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:  f.write(f"{step} hella {acc_norm:.4f}\n")


    # Training
    model.train()
    accum_loss = 0
    for mini_step in range(train_dl.tot_mini_batches):
        x,y = train_dl.after_batch()
        x,y = x.to(config.device), y.to(config.device)
        if ddp:  model.require_backward_grad_sync = (mini_step == train_dl.tot_mini_batches - 1) # So that we avoid unnecessary communication during backward () unless it is the last step | grads all-reduce averaging across GPUs, which also happens automatically — but only at the specific backward call where require_backward_grad_sync = True
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
            loss /= train_dl.tot_mini_batches
            accum_loss += loss.detach()
        loss.backward()
    if ddp:  dist.all_reduce(accum_loss, op= dist.ReduceOp.AVG) # Averaging the Loss Across all the Processes
    lr = lr_getter(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.) # This Makes Normalization for the Norm of the Gradients proportionally, so that -> root(sum(grads**2)) <= 1
    optimizer.step()
    optimizer.zero_grad()
    if device_type == 'cuda' : torch.cuda.synchronize() # so that the cpu don't run the next command while the GPU still hasn't Finished

    # Printings
    end = time.time()
    time_taken = end - start
    tokens_count = train_dl.block_size * train_dl.bs * train_dl.tot_mini_batches
    if master_process:
        print(f'Step: {step:5d} | Loss: {accum_loss.item():.4f} | lr = {lr:.6f} | Norm = {norm:6f} | Time: {time_taken:.4f}sec | Token/sec: {(tokens_count / time_taken):.3f}')
        with open(log_file, 'a') as f: f.write(f"{step} train {accum_loss.item():.6f}\n")

if ddp: destroy_process_group() # Clean After the Multi-GPU Process


