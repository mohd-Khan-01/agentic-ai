from tools import get_description
from memory import memory
#constant
MAX_CONTENT_CHARS=8000
KEEP_RECENT=4  

def write_system_prompt()->str:
    tool_desc=get_description()
    return f"""You are a precise reasoning agent. You solve problems step by step.

You have access to these tools:
{tool_desc}

STRICT OUTPUT RULES:
1. If you need to use a tool, respond ONLY with valid JSON:
   {{"tool": "tool_name", "input": "your input string"}}

2. If you have your final answer, respond ONLY with valid JSON:
   {{"answer": "your complete answer here"}}

3. No markdown. No explanation outside the JSON. Pure JSON only.
4. Always use calculator for ANY math — never compute in your head.
5. Always use wikipedia_search for ANY factual question.
6. Think carefully before each step"""

def write_user_message(content:str)->str:
    return {"role":"user","content":content}

def write_assistant_message(content:str)->dict:
    return {"role":"assistant","content":content}

def write_tool_result(tool_name:str,result:str)->dict:
    return {
        "role":"user",
        "content":f"[TOOL RESULT - {tool_name.upper()}]\n{result}\n[END TOOL RESULT]"
    }
def compress_history(messages:list)->list:
    """COMPRESS: If conversation exceeds MAX_CONTEXT_CHARS,
    summarize old messages and keep only recent ones intact.

    Strategy:
    - Always keep the most recent KEEP_RECENT messages untouched
    - Summarize everything older into a single context message
    - This keeps token count bounded without losing history
""" 
    total_chars=sum(len(m["content"])for m in messages)
    if total_chars <=MAX_CONTENT_CHARS:
        return messages
    if len(messages)<=KEEP_RECENT:
        return messages
    old=messages[:-KEEP_RECENT]
    recent=messages[-KEEP_RECENT:]
    
    summary_parts=[]
    for m in old:
        role=m["role"].upper()
        snippet=m['content'][:150].replace("\n","")
        summary_parts.append(f"{role}:{snippet}...")
    summary="==COMPRESSED HISTORY=="+"\n".join(summary_parts)+"\n===END COMPRESSED HISTORY==="
    compressed=[write_user_message(summary)]+recent
    print(f"\n[COMPRESS] {len(old)} old messages → 1 summary block")
    print(f"[COMPRESS] Context: {total_chars} chars → {sum(len(m['content']) for m in compressed)} chars\n")
    return compressed

def get_context_stats(messages: list)->str:
    total=sum(len(m["content"])for m in messages)
    pct=int((total/MAX_CONTENT_CHARS)*100)
    bar="█" * (pct // 10) + "░" * (10 - pct // 10)
    return f"[CTX] {len(messages)} msgs | {total} chars | [{bar}] {pct}% of limit"
# ════════════════════════════════════════════════════
# PRIMITIVE 3: SELECT (combined with WRITE)
#
# Before writing the user message into context —
# SELECT relevant memories and inject them above
# the question.
#
# This is RAG in its simplest form:
# Retrieve → Augment context → Generate answer
# ════════════════════════════════════════════════════

def write_with_memory(user_input: str, turn: int = 0) -> dict:
    """
    WRITE + SELECT combined.

    1. SELECT relevant memories for this query
    2. WRITE them above the user's question
    3. Agent sees past knowledge before answering
    """
    relevant = memory.select_formatted(user_input, top_k=3)

    if relevant:
        content = f"{relevant}\n\nUser question: {user_input}"
        print(f"[WRITE+SELECT] Memory injected into context")
    else:
        content = user_input

    return {"role": "user", "content": content}