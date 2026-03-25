import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model_id = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.float32
).to(device).eval()

prompt= "Explain KV cache in one paragraph, with an analogy."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        use_cache=True,
        return_dict_in_generate=True,  # returns a ModelOutput object instead of a raw Tensor
    )
# KV cache tensors are shaped (batch, num_heads, seq_len, head_dim)
pkv = outputs.past_key_values
# transformers 5.x: DynamicCache stores layers in .layers list, each with .keys / .values
k0 = pkv.layers[0].keys    # shape: (batch, num_heads, seq_len, head_dim)
v0 = pkv.layers[0].values
print("num layers:", len(pkv.layers))
print("num heads:", k0.shape[1])
print("seq len:", k0.shape[2])
print("head dim:", k0.shape[3])
print("K shape (layer 0):", k0.shape)
print("V shape (layer 0):", v0.shape)

print(tokenizer.decode(outputs.sequences[0], skip_special_tokens=True))