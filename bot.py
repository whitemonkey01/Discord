import discord
from discord.ext import commands
import aiohttp
import json
import re
import asyncio
import tempfile
import os
from pathlib import Path
from keep_alive import keep_alive

# Read from Render's Environment Variables
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GOOGLE_AI_KEY = os.getenv("GOOGLE_AI_KEY")
META_AI_KEY = os.getenv("META_AI_KEY")
QWEN_AI_KEY = os.getenv("QWEN_AI_KEY")
CHUTES_AI_KEY = os.getenv("CHUTES_AI_KEY")

# Set up bot
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# Custom identity information
BOT_CREATOR = "Asraf"
BOT_MODEL_INFO = "I'm powered by multiple AI models including Google Gemini, Meta Llama, Alibaba's Tongyi-DeepResearch-30B (Qwen), OpenAI GPT-OSS-20B, Zai-org GLM-4.5-Air, image generation models: JuggernautXL and Chroma, and text-to-speech synthesis"

# Identity-related questions in multiple languages
IDENTITY_PATTERNS = [
    # English patterns
    r"(who|what).*(you|yourself|your identity|created you|made you|developer|creator|owner)",
    r"(are you|who are you|what are you|tell me about yourself|your name|your model)",
    r"(how.*you.*work|what.*you.*do|your capabilities|your purpose)",
    
    
    # Add patterns for other languages a
    ded
    #
    ish
    r"(quién|qué).*(eres|tú|tu identidad|te creó|te hizo|desarrollador|creador,
    )",
    r"(eres tú|quién eres tú|qué eres tú|cuéntame sobre ti|tu nombre|tu ,
    
       
    
    nch
    r"(qui|quoi).*(es-tu|toi|ton identité|t'a créé|t'a fait|développeur|créateur|propri,
    )",
    r"(es-tu|qui es-tu|qu'es-tu|parle-moi de toi|ton nom|ton ,
    
       
   
    ndi
    r"(कौन|क्या).*(तुम|तुम्हारा|तुम्हारी पहचान|तुम्हें बनाया|तुम्हें बनाने वाला|डेवलपर|निर्माता,
    )",
    r"(तुम कौन हो|तुम क्या हो|अपने बारे में बताओ|तुम्हारा नाम|तुम्हार,
    
       
    
    bic
    r"(من|ما).*(أنت|نفسك|هويتك|قام بإنشائك|صانعك|المطور|المنشئ|,
    )",
    r"(هل أنت|من أنت|ما أنت|أخبرني عن نفسك|اسمك|,
و

ك)",
]

# Compile regex patterns for ef
identity_regex = rre.x = [re.pattern, re.ern, re.IGN REC pattern pa IDENTITY_PATTERNSP

TER S]

def is_identity_qtexti:
    t):
    """Check if the text is asking about the bot's ide
    text_lower = text. = text
    
       
    # Quick keyword check first for ef
    keywords = words , ["who", "what,  "you", "your", "y, rself", ", eator", "made", "de, 
                       , quién,  "qu, , "t, , "tu", "eres",, es-tu,  "qui", "quoi,
                     ,  "कौन", "क्या,  "तुम", "त, ्हारा",, बनाया", "न,
                    ,   "م, , "ما,  "أنت",, नर्सक", "منشئ",
    
       
    if keyword ke text_lower t_l keyword ke keywordsk:
                retu
    
       
    # More thorough reg
    eck pattern pa identity_regex:
           pattern.patterntextr:
                    ret
    
       
    retu

 Fa se

def get_custom_identity_re:
    ():
    """Return the custom response for identity ques
    """
    return f"🤖 I'm a multi-AI assistant creaBOT_CREATOR_CREBOT_MODEL_INFODEL_ \
                   f"I can answer questions using different AI models. How can I help you

obot.event
t.eve t
a ync def on:
    ():
    pribot.'✅ {bot.user} is now o
    !')
    print('✅ Bot is ready to receive com
    !')
    print('✅ Available commands: /googleai, /metaai, /qwenai, /gptoss, /glmair, /generateimage, /generatechroma, /speak, /speakmale, /speakfemale, !googleai, !metaai, !qwenai, !gptoss, !glmair, !generateimage, !generatechroma, !speak, !speakmale, !speak
    
       
    # Sync slash commands with better error 
    ing:
        synced = ynced bot.ait .ot.tre
                print(f"✅ Syncsyncedn(synced)} slash comm
            cmd fo synced:
                    print(fcmd.- /{cmd
    }")
   Exception ce e:
                print(f"❌ Error syncing slash comeand

 {e}")

# Simple test
cbot.command.c
mmand )
a ync dctxp:
    latency = tency bot.und(bot * tency
    00)
  ctx.ait ctx.send(f'🏓 Pong! Lalatency{late

y}ms')

# Help
cbot.command.c
mmand )
a ync defctxh:
    help_msg = p_msg = """
🤖 **Multi-AI Bot Help Commands**

**Text AI Commands:**
- `/googleai [question]` - Ask Google Gemini AI
- `/metaai [question]` - Ask Meta AI (Llama)
- `/qwenai [question]` - Ask Alibaba's Tongyi-DeepResearch-30B (Qwen) via Chutes.ai
- `/gptoss [question]` - Ask OpenAI GPT-OSS-20B via Chutes.ai
- `/glmair [question]` - Ask Zai-org GLM-4.5-Air via Chutes.ai

**Image Generation Commands:**
- `/generateimage [prompt]` - Generate AI image using JuggernautXL
- `/generatechroma [prompt]` - Generate AI image using Chroma model
- `/generateimage [prompt] negative_prompt: [negative]` - Generate with negative prompt
- `/generateimage [prompt] width: [size] height: [size]` - Generate with custom size

**Text-to-Speech Commands:**
- `/speak [text]` - Convert text to speech with default voice
- `/speakmale [text]` - Convert text to speech with male voice (af_michael)
- `/speakfemale [text]` - Convert text to speech with female voice (af_river)
- `/speak [text] speed: [value]` - Adjust speech speed (0.5 to 2.0)

**Prefix Commands:**
- `!googleai [question]` - Ask Google Gemini AI
- `!metaai [question]` - Ask Meta AI (Llama)
- `!qwenai [question]` - Ask Alibaba's Tongyi-DeepResearch-30B (Qwen) via Chutes.ai
- `!gptoss [question]` - Ask OpenAI GPT-OSS-20B via Chutes.ai
- `!glmair [question]` - Ask Zai-org GLM-4.5-Air via Chutes.ai
- `!generateimage [prompt]` - Generate AI image with JuggernautXL
- `!generatechroma [prompt]` - Generate AI image with Chroma
- `!speak [text]` - Convert text to speech
- `!speakmale [text]` - Convert text to speech with male voice
- `!speakfemale [text]` - Convert text to speech with female voice
- `!ping` - Check if bot is working
- `!aihelp` - Show this help message

**Examples:**
- `/googleai What is Gemi- `/metaai Explain machine learning`
- `/generateimage A beautiful sunset over mountains`
- `/generatechroma A fantasy castle in vibrant colors`
- `/speak Hello, how are you today?`
- `/speakmale Welcome to the server!`
- `/speakfemale This is a test message speed: 1.2`
"""
    await ctx.send(help_msg)

# Function to call Google Gemini AI
async def get_google_ai_response(question):
    # Check if this is an identity question first
    if is_identity_question(question):
        return get_custom_identity_response()
    
    try:
        # Try different Gemini model names
        model_names = ["gemini-1.5-flash", "gemini-pro", "models/gemini-pro"]
        
        for model_name in model_names:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={GOOGLE_AI_KEY}"
                
                headers = {"Content-Type": "application/json"}
                
                data = {
                    "contents": [{
                        "parts": [{"text": question}]
                    }]
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, headers=headers, json=data) as response:
                        if response.status == 200:
                            result = await response.json()
                            
                            if "candidates" in result and len(result["candidates"]) > 0:
                                if "content" in result["candidates"][0]:
                                    return result["candidates"][0]["content"]["parts"][0]["text"]
            except:
                continue
        
        return "❌ Could not connect to Google AI. Please try again later."
            
    except Exception as e:
        return f"❌ Google AI Error: {str(e)}"

# Function to call Meta AI (using Groq API for Llama models)
async def get_meta_ai_response(question):
    # Check if this is an identity question first
    if is_identity_question(question):
        return get_custom_identity_response()
    
    try:
        # Groq API endpoint for Meta's Llama models
        url = "https://api.groq.com/openai/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {META_AI_KEY}",
            "Content-Type": "application/json"
        }
        
        # Using Llama model
        model = "llama-3.1-8b-instant"
        
        data = {
            "model": model,
            "messages": [{"role": "user", "content": question}],
            "temperature": 0.7,
            "max_tokens": 1024
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]
                else:
                    error_text = await response.text()
                    return f"❌ Meta AI API Error: {response.status}"
    
    except Exception as e:
        return f"❌ Meta AI Error: {str(e)}"

# Function to call Alibaba's Tongyi-DeepResearch-30B (Qwen) via Chutes.ai
async def get_qwen_ai_response(question):
    # Check if this is an identity question first
    if is_identity_question(question):
        return get_custom_identity_response()
    
    try:
        headers = {
            "Authorization": f"Bearer {CHUTES_AI_KEY}",
            "Content-Type": "application/json"
        }
        
        body = {
            "model": "Alibaba-NLP/Tongyi-DeepResearch-30B-A3B",
            "messages": [
                {
                    "role": "user",
                    "content": question
                }
            ],
            "stream": False,
            "max_tokens": 1024,
            "temperature": 0.7
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://llm.chutes.ai/v1/chat/completions", 
                headers=headers,
                json=body,
                timeout=30
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    if "choices" in result and len(result["choices"]) > 0:
                        return result["choices"][0]["message"]["content"]
                    else:
                        return "❌ No response content received from Qwen AI."
                else:
                    error_text = await response.text()
                    return f"❌ Qwen API Error: {response.status} - {error_text}"
    
    except asyncio.TimeoutError:
        return "❌ Qwen AI request timed out. Please try again later."
    except Exception as e:
        return f"❌ Qwen AI Error: {str(e)}"

# Function to call OpenAI GPT-OSS-20B via Chutes.ai
async def get_gptoss_ai_response(question):
    # Check if this is an identity question first
    if is_identity_question(question):
        return get_custom_identity_response()
    
    try:
        headers = {
            "Authorization": f"Bearer {CHUTES_AI_KEY}",
            "Content-Type": "application/json"
        }
        
        body = {
            "model": "openai/gpt-oss-20b",
            "messages": [
                {
                    "role": "user",
                    "content": question
                }
            ],
            "stream": False,
            "max_tokens": 1024,
            "temperature": 0.7
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://llm.chutes.ai/v1/chat/completions", 
                headers=headers,
                json=body,
                timeout=30
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    if "choices" in result and len(result["choices"]) > 0:
                        return result["choices"][0]["message"]["content"]
                    else:
                        return "❌ No response content received from GPT-OSS-20B."
                else:
                    error_text = await response.text()
                    return f"❌ GPT-OSS-20B API Error: {response.status} - {error_text}"
    
    except asyncio.TimeoutError:
        return "❌ GPT-OSS-20B request timed out. Please try again later."
    except Exception as e:
        return f"❌ GPT-OSS-20B Error: {str(e)}"

# Function to call Zai-org GLM-4.5-Air via Chutes.ai
async def get_glmair_ai_response(question):
    # Check if this is an identity question first
    if is_identity_question(question):
        return get_custom_identity_response()
    
    try:
        headers = {
            "Authorization": f"Bearer {CHUTES_AI_KEY}",
            "Content-Type": "application/json"
        }
        
        body = {
            "model": "zai-org/GLM-4.5-Air",
            "messages": [
                {
                    "role": "user",
                    "content": question
                }
            ],
            "stream": False,
            "max_tokens": 1024,
            "temperature": 0.7
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://llm.chutes.ai/v1/chat/completions", 
                headers=headers,
                json=body,
                timeout=30
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    if "choices" in result and len(result["choices"]) > 0:
                        return result["choices"][0]["message"]["content"]
                    else:
                        return "❌ No response content received from GLM-4.5-Air."
                else:
                    error_text = await response.text()
                    return f"❌ GLM-4.5-Air API Error: {response.status} - {error_text}"
    
    except asyncio.TimeoutError:
        return "❌ GLM-4.5-Air request timed out. Please try again later."
    except Exception as e:
        return f"❌ GLM-4.5-Air Error: {str(e)}"

# Function to generate AI image using JuggernautXL
async def generate_ai_image(prompt, negative_prompt="blur, distortion, low quality", width=1024, height=1024, model="JuggernautXL"):
    try:
        headers = {
            "Authorization": f"Bearer {CHUTES_AI_KEY}",
            "Content-Type": "application/json"
        }
        
        body = {
            "model": model,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "guidance_scale": 7.5,
            "width": width,
            "height": height,
            "num_inference_steps": 50
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://image.chutes.ai/generate", 
                headers=headers,
                json=body,
                timeout=120  # Longer timeout for image generation
            ) as response:
                
                if response.status == 200:
                    # Read the response as binary data (image)
                    image_data = await response.read()
                    
                    # Save the image to a temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                        tmp_file.write(image_data)
                        return tmp_file.name, None
                else:
                    error_text = await response.text()
                    return None, f"❌ Image generation failed with status: {response.status} - {error_text}"
    
    except asyncio.TimeoutError:
        return None, "❌ Image generation timed out. Please try again later."
    except Exception as e:
        return None, f"❌ Image generation error: {str(e)}"

# Function to generate text-to-speech audio
async def generate_speech(text, voice="af_river", speed=1.0):
    try:
        headers = {
            "Authorization": f"Bearer {CHUTES_AI_KEY}",
            "Content-Type": "application/json"
        }
        
        body = {
            "text": text,
            "speed": speed,
            "voice": voice
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://chutes-kokoro.chutes.ai/speak", 
                headers=headers,
                json=body,
                timeout=60  # Longer timeout for speech generation
            ) as response:
                
                if response.status == 200:
                    # Read the response as binary audio data
                    audio_data = await response.read()
                    
                    # Save the audio to a temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                        tmp_file.write(audio_data)
                        return tmp_file.name, None
                else:
                    error_text = await response.text()
                    return None, f"❌ Speech generation failed with status: {response.status} - {error_text}"
    
    except asyncio.TimeoutError:
        return None, "❌ Speech generation timed out. Please try again later."
    except Exception as e:
        return None, f"❌ Speech generation error: {str(e)}"

# Helper function to parse image generation parameters
def parse_image_prompt(full_prompt):
    prompt = full_prompt
    negative_prompt = "blur, distortion, low quality, bad quality, watermark"
    width = 1024
    height = 1024
    
    # Parse negative prompt
    if "negative_prompt:" in full_prompt:
        parts = full_prompt.split("negative_prompt:")
        prompt = parts[0].strip()
        negative_part = parts[1].strip()
        if "width:" in negative_part:
            negative_prompt = negative_part.split("width:")[0].strip()
        else:
            negative_prompt = negative_part
    
    # Parse width and height
    if "width:" in full_prompt and "height:" in full_prompt:
        try:
            width_part = full_prompt.split("width:")[1]
            if "height:" in width_part:
                width_str = width_part.split("height:")[0].strip()
                height_str = width_part.split("height:")[1].strip()
                width = int(width_str)
                height = int(height_str)
        except (ValueError, IndexError):
            pass
    
    return prompt, negative_prompt, width, height

# Helper function to parse speech parameters
def parse_speech_text(full_text):
    text = full_text
    speed = 1.0
    
    # Parse speed parameter
    if "speed:" in full_text:
        parts = full_text.split("speed:")
        text = parts[0].strip()
        try:
            speed_str = parts[1].strip().split()[0]  # Get first word after speed:
            speed = float(speed_str)
            # Clamp speed between 0.5 and 2.0
            speed = max(0.5, min(2.0, speed))
        except (ValueError, IndexError):
            pass
    
    return text, speed

# Helper function to send long messages in chunks
async def send_long_message(ctx_or_followup, response, is_followup=False):
    if len(response) > 2000:
        chunks = [response[i:i+2000] for i in range(0, len(response), 2000)]
        if is_followup:
            await ctx_or_followup.send(chunks[0])
            for chunk in chunks[1:]:
                await ctx_or_followup.send(chunk)
        else:
            await ctx_or_followup.send(chunks[0])
            for chunk in chunks[1:]:
                await ctx_or_followup.send(chunk)
    else:
        if is_followup:
            await ctx_or_followup.send(response)
        else:
            await ctx_or_followup.send(response)

# SLASH COMMAND: /googleai
@bot.tree.command(name="googleai", description="Ask Google Gemini AI a question")
async def googleai_slash(interaction: discord.Interaction, question: str):
    await interaction.response.defer(thinking=True)
    response = await get_google_ai_response(question)
    await send_long_message(interaction.followup, response, is_followup=True)

# SLASH COMMAND: /metaai
@bot.tree.command(name="metaai", description="Ask Meta AI (Llama) a question")
async def metaai_slash(interaction: discord.Interaction, question: str):
    await interaction.response.defer(thinking=True)
    response = await get_meta_ai_response(question)
    await send_long_message(interaction.followup, response, is_followup=True)

# SLASH COMMAND: /qwenai
@bot.tree.command(name="qwenai", description="Ask Alibaba's Tongyi-DeepResearch-30B (Qwen) a question via Chutes.ai")
async def qwenai_slash(interaction: discord.Interaction, question: str):
    await interaction.response.defer(thinking=True)
    response = await get_qwen_ai_response(question)
    await send_long_message(interaction.followup, response, is_followup=True)

# SLASH COMMAND: /gptoss
@bot.tree.command(name="gptoss", description="Ask OpenAI GPT-OSS-20B a question via Chutes.ai")
async def gptoss_slash(interaction: discord.Interaction, question: str):
    await interaction.response.defer(thinking=True)
    response = await get_gptoss_ai_response(question)
    await send_long_message(interaction.followup, response, is_followup=True)

# SLASH COMMAND: /glmair
@bot.tree.command(name="glmair", description="Ask Zai-org GLM-4.5-Air a question via Chutes.ai")
async def glmair_slash(interaction: discord.Interaction, question: str):
    await interaction.response.defer(thinking=True)
    response = await get_glmair_ai_response(question)
    await send_long_message(interaction.followup, response, is_followup=True)

# SLASH COMMAND: /generateimage
@bot.tree.command(name="generateimage", description="Generate an AI image using JuggernautXL")
async def generateimage_slash(interaction: discord.Interaction, prompt: str):
    await interaction.response.defer(thinking=True)
    
    # Parse prompt for additional parameters
    clean_prompt, negative_prompt, width, height = parse_image_prompt(prompt)
    
    # Generate image with JuggernautXL model
    image_path, error = await generate_ai_image(clean_prompt, negative_prompt, width, height, model="JuggernautXL")
    
    if image_path:
        try:
            # Send the generated image
            file = discord.File(image_path, filename="generated_image.jpg")
            embed = discord.Embed(
                title="🖼️ AI Generated Image (JuggernautXL)",
                description=f"**Prompt:** {clean_prompt}\n**Size:** {width}x{height}",
                color=0x00ff00
            )
            if negative_prompt:
                embed.add_field(name="Negative Prompt", value=negative_prompt[:100] + "..." if len(negative_prompt) > 100 else negative_prompt, inline=False)
            embed.set_image(url="attachment://generated_image.jpg")
            
            await interaction.followup.send(file=file, embed=embed)
        finally:
            # Clean up temporary file
            try:
                os.unlink(image_path)
            except:
                pass
    else:
        await interaction.followup.send(error)

# SLASH COMMAND: /generatechroma
@bot.tree.command(name="generatechroma", description="Generate an AI image using Chroma model")
async def generatechroma_slash(interaction: discord.Interaction, prompt: str):
    await interaction.response.defer(thinking=True)
    
    # Parse prompt for additional parameters
    clean_prompt, negative_prompt, width, height = parse_image_prompt(prompt)
    
    # Generate image with Chroma model
    image_path, error = await generate_ai_image(clean_prompt, negative_prompt, width, height, model="chroma")
    
    if image_path:
        try:
            # Send the generated image
            file = discord.File(image_path, filename="generated_image.jpg")
            embed = discord.Embed(
                title="🎨 AI Generated Image (Chroma)",
                description=f"**Prompt:** {clean_prompt}\n**Size:** {width}x{height}",
                color=0x9370DB  # Purple color for Chroma
            )
            if negative_prompt:
                embed.add_field(name="Negative Prompt", value=negative_prompt[:100] + "..." if len(negative_prompt) > 100 else negative_prompt, inline=False)
            embed.set_image(url="attachment://generated_image.jpg")
            
            await interaction.followup.send(file=file, embed=embed)
        finally:
            # Clean up temporary file
            try:
                os.unlink(image_path)
            except:
                pass
    else:
        await interaction.followup.send(error)

# SLASH COMMAND: /speak
@bot.tree.command(name="speak", description="Convert text to speech with default voice")
async def speak_slash(interaction: discord.Interaction, text: str):
    await interaction.response.defer(thinking=True)
    
    # Parse text for additional parameters
    clean_text, speed = parse_speech_text(text)
    
    # Generate speech with default voice (female)
    audio_path, error = await generate_speech(clean_text, voice="af_river", speed=speed)
    
    if audio_path:
        try:
            # Send the generated audio
            file = discord.File(audio_path, filename="speech.wav")
            embed = discord.Embed(
                title="🎵 Text-to-Speech",
                description=f"**Text:** {clean_text}\n**Voice:** Female\n**Speed:** {speed}x",
                color=0xFF69B4  # Pink color for TTS
            )
            
            await interaction.followup.send(file=file, embed=embed)
        finally:
            # Clean up temporary file
            try:
                os.unlink(audio_path)
            except:
                pass
    else:
        await interaction.followup.send(error)

# SLASH COMMAND: /speakfemale
@bot.tree.command(name="speakfemale", description="Convert text to speech with female voice")
async def speakfemale_slash(interaction: discord.Interaction, text: str):
    await interaction.response.defer(thinking=True)
    
    # Parse text for additional parameters
    clean_text, speed = parse_speech_text(text)
    
    # Generate speech with female voice
    audio_path, error = await generate_speech(clean_text, voice="af_river", speed=speed)
    
    if audio_path:
        try:
            # Send the generated audio
            file = discord.File(audio_path, filename="speech_female.wav")
            embed = discord.Embed(
                title="🎵 Text-to-Speech (Female)",
                description=f"**Text:** {clean_text}\n**Voice:** Female\n**Speed:** {speed}x",
                color=0xFF69B4  # Pink color for female TTS
            )
            
            await interaction.followup.send(file=file, embed=embed)
        finally:
            # Clean up temporary file
            try:
                os.unlink(audio_path)
            except:
                pass
    else:
        await interaction.followup.send(error)

# SLASH COMMAND: /speakmale
@bot.tree.command(name="speakmale", description="Convert text to speech with male voice")
async def speakmale_slash(interaction: discord.Interaction, text: str):
    await interaction.response.defer(thinking=True)
    
    # Parse text for additional parameters
    clean_text, speed = parse_speech_text(text)
    
    # Generate speech with male voice
    audio_path, error = await generate_speech(clean_text, voice="af_michael", speed=speed)
    
    if audio_path:
        try:
            # Send the generated audio
            file = discord.File(audio_path, filename="speech_male.wav")
            embed = discord.Embed(
                title="🎵 Text-to-Speech (Male)",
                description=f"**Text:** {clean_text}\n**Voice:** Male\n**Speed:** {speed}x",
                color=0x1E90FF  # Blue color for male TTS
            )
            
            await interaction.followup.send(file=file, embed=embed)
        finally:
            # Clean up temporary file
            try:
                os.unlink(audio_path)
            except:
                pass
    else:
        await interaction.followup.send(error)

# PREFIX COMMAND: !googleai
@bot.command()
async def googleai(ctx, *, question):
    async with ctx.typing():
        response = await get_google_ai_response(question)
        await send_long_message(ctx, response)

# PREFIX COMMAND: !metaai
@bot.command()
async def metaai(ctx, *, question):
    async with ctx.typing():
        response = await get_meta_ai_response(question)
        await send_long_message(ctx, response)

# PREFIX COMMAND: !qwenai
@bot.command()
async def qwenai(ctx, *, question):
    async with ctx.typing():
        response = await get_qwen_ai_response(question)
        await send_long_message(ctx, response)

# PREFIX COMMAND: !gptoss
@bot.command()
async def gptoss(ctx, *, question):
    async with ctx.typing():
        response = await get_gptoss_ai_response(question)
        await send_long_message(ctx, response)

# PREFIX COMMAND: !glmair
@bot.command()
async def glmair(ctx, *, question):
    async with ctx.typing():
        response = await get_glmair_ai_response(question)
        await send_long_message(ctx, response)

# PREFIX COMMAND: !generateimage
@bot.command()
async def generateimage(ctx, *, prompt):
    async with ctx.typing():
        # Parse prompt for additional parameters
        clean_prompt, negative_prompt, width, height = parse_image_prompt(prompt)
        
        # Generate image with JuggernautXL model
        image_path, error = await generate_ai_image(clean_prompt, negative_prompt, width, height, model="JuggernautXL")
        
        if image_path:
            try:
                # Send the generated image
                file = discord.File(image_path, filename="generated_image.jpg")
                embed = discord.Embed(
                    title="🖼️ AI Generated Image (JuggernautXL)",
                    description=f"**Prompt:** {clean_prompt}\n**Size:** {width}x{height}",
                    color=0x00ff00
                )
                if negative_prompt:
                    embed.add_field(name="Negative Prompt", value=negative_prompt[:100] + "..." if len(negative_prompt) > 100 else negative_prompt, inline=False)
                embed.set_image(url="attachment://generated_image.jpg")
                
                await ctx.send(file=file, embed=embed)
            finally:
                # Clean up temporary file
                try:
                    os.unlink(image_path)
                except:
                    pass
        else:
            await ctx.send(error)

# PREFIX COMMAND: !generatechroma
@bot.command()
async def generatechroma(ctx, *, prompt):
    async with ctx.typing():
        # Parse prompt for additional parameters
        clean_prompt, negative_prompt, width, height = parse_image_prompt(prompt)
        
        # Generate image with Chroma model
        image_path, error = await generate_ai_image(clean_prompt, negative_prompt, width, height, model="chroma")
        
        if image_path:
            try:
                # Send the generated image
                file = discord.File(image_path, filename="generated_image.jpg")
                embed = discord.Embed(
                    title="🎨 AI Generated Image (Chroma)",
                    description=f"**Prompt:** {clean_prompt}\n**Size:** {width}x{height}",
                    color=0x9370DB  # Purple color for Chroma
                )
                if negative_prompt:
                    embed.add_field(name="Negative Prompt", value=negative_prompt[:100] + "..." if len(negative_prompt) > 100 else negative_prompt, inline=False)
                embed.set_image(url="attachment://generated_image.jpg")
                
                await ctx.send(file=file, embed=embed)
            finally:
                # Clean up temporary file
                try:
                    os.unlink(image_path)
                except:
                    pass
        else:
            await ctx.send(error)

# PREFIX COMMAND: !speak
@bot.command()
async def speak(ctx, *, text):
    async with ctx.typing():
        # Parse text for additional parameters
        clean_text, speed = parse_speech_text(text)
        
        # Generate speech with default voice (female)
        audio_path, error = await generate_speech(clean_text, voice="af_river", speed=speed)
        
        if audio_path:
            try:
                # Send the generated audio
                file = discord.File(audio_path, filename="speech.wav")
                embed = discord.Embed(
                    title="🎵 Text-to-Speech",
                    description=f"**Text:** {clean_text}\n**Voice:** Female\n**Speed:** {speed}x",
                    color=0xFF69B4  # Pink color for TTS
                )
                
                await ctx.send(file=file, embed=embed)
            finally:
                # Clean up temporary file
                try:
                    os.unlink(audio_path)
                except:
                    pass
        else:
            await ctx.send(error)

# PREFIX COMMAND: !speakfemale
@bot.command()
async def speakfemale(ctx, *, text):
    async with ctx.typing():
        # Parse text for additional parameters
        clean_text, speed = parse_speech_text(text)
        
        # Generate speech with female voice
        audio_path, error = await generate_speech(clean_text, voice="af_river", speed=speed)
        
        if audio_path:
            try:
                # Send the generated audio
                file = discord.File(audio_path, filename="speech_female.wav")
                embed = discord.Embed(
                    title="🎵 Text-to-Speech (Female)",
                    description=f"**Text:** {clean_text}\n**Voice:** Female\n**Speed:** {speed}x",
                    color=0xFF69B4  # Pink color for female TTS
                )
                
                await ctx.send(file=file, embed=embed)
            finally:
                # Clean up temporary file
                try:
                    os.unlink(audio_path)
                except:
                    pass
        else:
            await ctx.send(error)

# PREFIX COMMAND: !speakmale
@bot.command()
async def speakmale(ctx, *, text):
    async with ctx.typing():
        # Parse text for additional parameters
        clean_text, speed = parse_speech_text(text)
        
        # Generate speech with male voice
        audio_path, error = await generate_speech(clean_text, voice="af_michael", speed=speed)
        
        if audio_path:
            try:
                # Send the generated audio
                file = discord.File(audio_path, filename="speech_male.wav")
                embed = discord.Embed(
                    title="🎵 Text-to-Speech (Male)",
                    description=f"**Text:** {clean_text}\n**Voice:** Male\n**Speed:** {speed}x",
                    color=0x1E90FF  # Blue color for male TTS
                )
                
                await ctx.send(file=file, embed=embed)
            finally:
                # Clean up temporary file
                try:
                    os.unlink(audio_path)
                except:
                    pass
        else:
            await ctx.send(error)

# Error handling for unknown commands
@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandNotFound):
        await ctx.send("❌ Unknown command. Use `!aihelp` to see available commands.")
    else:
        print(f"Error: {error}")

# Run the bot
if __name__ == "__main__":
    try:
        keep_alive()           # Start the web server thread first
        bot.run(DISCORD_TOKEN) # Then start the bot
    except Exception as e:
        print(f"Failed to start bot: {e}")

