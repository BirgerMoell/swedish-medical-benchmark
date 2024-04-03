from transformers import AutoTokenizer
import transformers
import torch

model = "birgermoell/eir"

from transformers import AutoTokenizer
import transformers
import torch

messages = [{"role": "user", "content": "Är anorektal endosonografi värdefull vid dyschezi?"}]

tokenizer = AutoTokenizer.from_pretrained(model)
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

outputs = pipeline(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
print(outputs[0]["generated_text"])