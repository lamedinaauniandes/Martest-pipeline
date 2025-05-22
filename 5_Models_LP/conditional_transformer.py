import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
     GPT2LMHeadModel,

)
 
class CondDataset(Dataset):
    def __init__(self, encodings, rewards, probs):
        """
        encodings: dict con 'input_ids' y 'attention_mask' (tensors pt)
        rewards:  lista o tensor de floats normalizados en [0,1]
        probs:    lista o tensor de floats normalizados en [0,1]
        """
        self.encodings = encodings
        self.rewards   = torch.tensor(rewards, dtype=torch.float)
        self.probs     = torch.tensor(probs,   dtype=torch.float)

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.encodings["input_ids"][idx],  # causal LM
            "reward":         self.rewards[idx],
            "prob_fault":     self.probs[idx],
        }

# dataset = CondDataset(encodings,rewards,prob_fault)

class CondGPT2(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        d = config.n_embd
        self.cond_proj = nn.Sequential(
            nn.Linear(2, d),
            nn.ReLU(),
            nn.Linear(d, d)
        )

    def forward(self, input_ids, attention_mask=None,
                labels=None, reward=None, prob_fault=None, **kwargs):


        cond = torch.stack([reward, prob_fault], dim=-1)     
        cond_emb = self.cond_proj(cond)                    

        inputs_embeds = self.transformer.wte(input_ids)    

        cond_exp = cond_emb.unsqueeze(1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds + cond_exp
        
        return super().forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )

    def prepare_inputs_for_generation(self,
            input_ids,
            attention_mask=None,
            **model_kwargs
        ):
        # Este método se invoca en cada paso de generate().
        # Sacamos reward y prob_fault de model_kwargs
        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "reward":         model_kwargs["reward"],
            "prob_fault":     model_kwargs["prob_fault"],
        }




if __name__ == '__main__': 
    from transformers import GPT2Config

    config = GPT2Config(
            vocab_size=996,
            n_embd=1024,
            n_layer=12,
            n_head=16,
            # bos_token_id=self.fast_tokenizer.pad_token_id,
            # eos_token_id=self.fast_tokenizer.pad_token_id,
            # pad_token_id=self.fast_tokenizer.pad_token_id
        )
    model = CondGPT2(config)
    print(model)