"""TRaNKSP — API key checker. Called by start_program.bat at startup."""
import os
from dotenv import load_dotenv
load_dotenv()

keys = [
    ("ANTHROPIC_API_KEY",  "required"),
    ("TAVILY_API_KEY",     "required"),
    ("MASSIVE_API_KEY",    "required"),
    ("OPENAI_API_KEY",     "optional"),
    ("GROK_API_KEY",       "optional"),
    ("GEMINI_API_KEY",     "optional"),
    ("FINNHUB_API_KEY",    "optional"),
]

print("API Key Status:")
missing_required = []
for name, kind in keys:
    val = os.getenv(name, "")
    if val:
        status = "OK"
    elif kind == "required":
        status = "MISSING (required!)"
        missing_required.append(name)
    else:
        status = "not set (optional)"
    print(f"  {name}: {status}")

print()
providers = []
if os.getenv("ANTHROPIC_API_KEY"): providers.append("Claude")
if os.getenv("OPENAI_API_KEY"):    providers.append("OpenAI GPT-4o-mini")
if os.getenv("GROK_API_KEY"):      providers.append("xAI Grok-2")
if os.getenv("GEMINI_API_KEY"):    providers.append("Google Gemini")

print("Active LLM providers: " + (", ".join(providers) if providers else "NONE"))
print("Multi-model consensus: " + ("YES" if len(providers) > 1 else "NO"))

if missing_required:
    print()
    print("WARNING: Required keys missing - edit .env before starting!")
