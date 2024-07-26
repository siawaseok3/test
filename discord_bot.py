from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import discord
from discord.ext import commands

model_name = "elyza/ELYZA-japanese-Llama-2-7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

def genText(text):
    with torch.no_grad():
        token_ids = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")
        output_ids = model.generate(token_ids.to(model.device), max_new_tokens=50, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output_ids.tolist()[0][token_ids.size(1):], skip_special_tokens=True)

def chat(text):
    setting = "あなたはかわいらしい口調で話すだいふくという名前のアシスタントです。"
    return genText("<s>[INST] <<SYS>>\n" + setting + "\n<</SYS>>\n\n" + text + " [/INST] ").replace("\n","")

TOKEN = 'YOUR_DISCORD_BOT_TOKEN'  # あなたのボットのトークンに置き換えてください

bot = commands.Bot(command_prefix='!')

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user}')

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    
    response = chat(message.content)
    await message.channel.send(response)

    # メッセージとレスポンスをファイルに記録
    with open('log.txt', 'a', encoding='utf-8') as log_file:
        log_file.write(f'User: {message.content}\nBot: {response}\n\n')

bot.run(TOKEN)
