# src/agent.py
import os
import json
import sys
from google import genai
from google.genai import types
from groq import Groq
from dotenv import load_dotenv

from tools import dispatch_tool
from context import (
    write_system_prompt,
    write_user_message,
    write_assistant_message,
    write_tool_result,
    compress_history,
    get_context_stats,
)

# ── Init ──────────────────────────────────────────────────
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("ERROR: GGROQ_API_KEY not found in .env")
    sys.exit(1)

client=Groq(api_key=api_key)


MAX_STEPS = 8  # Safety cap — prevents infinite loops


# ── LLM Call ─────────────────────────────────────────────

def call_llm(system_prompt: str, messages: list) -> str:
    """
    Sends conversation to Groq (Llama 3.1 70B).
    Same logic as before — just different provider.
    """
    # Build messages in the format Groq expects
    groq_messages = [{"role": "system", "content": system_prompt}]

    for msg in messages:
        groq_messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=groq_messages,
        temperature=0.1,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()

# ── Response Parser ───────────────────────────────────────

def parse_response(raw: str) -> dict:
    """
    Parses the LLM's JSON response.
    Handles messy outputs — strips markdown fences, whitespace, etc.
    Returns: {"tool": ..., "input": ...} OR {"answer": ...}
    """
    clean = raw.strip()

    # Strip markdown code fences if model got chatty
    if "```" in clean:
        parts = clean.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                clean = part
                break

    # Find first { to last } in case of extra text
    start = clean.find("{")
    end = clean.rfind("}") + 1
    if start != -1 and end > start:
        clean = clean[start:end]

    try:
        parsed = json.loads(clean)
        # Validate it has expected keys
        if "tool" in parsed or "answer" in parsed:
            return parsed
        else:
            return {"answer": raw}  # Unexpected keys — treat as answer
    except json.JSONDecodeError:
        return {"answer": raw}  # Can't parse — treat whole thing as answer


# ── Core Agent Loop ───────────────────────────────────────

def run_agent(user_input: str) -> str:
    """
    THE LOOP — this is the heart of every agentic system.

    Think → Decide (tool or answer?) → Act → Observe → Think → ...

    This exact pattern is what LangGraph, LangChain AgentExecutor,
    CrewAI, and every other framework implements for you.
    Today you're building it yourself.
    """
    divider = "─" * 52

    print(f"\n{divider}")
    print(f"  USER: {user_input}")
    print(divider)

    system_prompt = write_system_prompt()
    messages = [write_user_message(user_input)]
    tool_calls_made = []

    for step in range(1, MAX_STEPS + 1):
        print(f"\n  ┌─ Step {step}/{MAX_STEPS} ─────────────────────────")

        # Compress context if needed
        messages = compress_history(messages)
        print(f"  │ {get_context_stats(messages)}")

        # Call the LLM
        raw_response = call_llm(system_prompt, messages)
        print(f"  │ LLM raw: {raw_response[:120]}{'...' if len(raw_response) > 120 else ''}")

        # Parse what it decided
        parsed = parse_response(raw_response)

        # ── Branch 1: Tool Use ──
        if "tool" in parsed:
            tool_name = parsed.get("tool", "")
            tool_input = parsed.get("input", "")

            print(f"  │ 🔧 Tool: {tool_name}({repr(tool_input)})")

            result = dispatch_tool(tool_name, tool_input)
            tool_calls_made.append({"tool": tool_name, "input": tool_input, "result": result})

            print(f"  │ 📤 Result: {result[:120]}{'...' if len(result) > 120 else ''}")
            print(f"  └─────────────────────────────────────────────")

            # Feed result back into context
            messages.append(write_assistant_message(raw_response))
            messages.append(write_tool_result(tool_name, result))

        # ── Branch 2: Final Answer ──
        elif "answer" in parsed:
            final = parsed["answer"]
            print(f"  └─────────────────────────────────────────────")
            print(f"\n  ✅ ANSWER: {final}")
            print(f"\n  📊 Summary: {step} steps | {len(tool_calls_made)} tool calls")
            for tc in tool_calls_made:
                print(f"     └─ {tc['tool']}({repr(tc['input'][:40])})")
            print(divider)
            return final

        # ── Branch 3: Unexpected ──
        else:
            print(f"  └─ Unexpected format, treating as answer")
            return raw_response

    # Reached max steps without answer
    print(f"\n  ⚠️  Max steps reached ({MAX_STEPS}). Returning last response.")
    print(divider)
    return "Agent reached step limit without a final answer."


# ── Main ─────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 52)
    print("  AGENTIC AI — Week 1 Day 1")
    print("  Raw tool-calling loop. Zero frameworks.")
    print("=" * 52)

    questions = [
        "What is 2 raised to the power 15, divided by the square root of 225?",
        "What is a transformer model in machine learning? Give me a 2 line summary.",
        "How many bones are in the human body, and what is the square root of that number?",
    ]

    for i, q in enumerate(questions, 1):
        print(f"\n[Question {i} of {len(questions)}]")
        run_agent(q)
        if i < len(questions):
            input("\n  Press Enter for next question...\n")

    print("\n✅ Day 1 complete. Now write your NOTES.md and push to GitHub.")