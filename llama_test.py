import transformers
import torch

model_id = "/mnt/Meta-Inspector/meta-llama/Meta-Llama-3.1-8B-Instruct/"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

outputs = pipeline(
    messages,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"][-1])


# import torch
# from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# # 设置设备
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(device)

# local_path = "/mnt/Meta-Inspector/meta-llama/Meta-Llama-3.1-8B-Instruct/"

# # 加载模型和tokenizer
# llama_tokenizer = AutoTokenizer.from_pretrained(local_path)
# llama_model = AutoModelForCausalLM.from_pretrained(local_path, torch_dtype=torch.bfloat16, device_map="auto")

# # 创建文本生成pipeline
# llama_pipeline = pipeline(
#     "text-generation",
#     model=llama_model,
#     tokenizer=llama_tokenizer,
#     device_map="auto",
#     model_kwargs={"torch_dtype": torch.bfloat16},
# )

# messages = [
#     {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
#     {"role": "user", "content": "Who are you?"},
# ]

# outputs = llama_pipeline(
#     messages,
#     max_new_tokens=256,
# )
# print(outputs[0]["generated_text"][-1])
