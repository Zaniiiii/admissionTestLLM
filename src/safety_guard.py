def construct_safe_prompt(query, retrieved_docs):
    # Join docs and strictly truncate to avoid token overflow
    context_str = "\n---\n".join(retrieved_docs)
    if len(context_str) > 4000:
        context_str = context_str[:4000] + "...[TRUNCATED]"
    
    # SYSTEM PROMPT (The Guardrail)
    # Llama-3 works best with clear, step-by-step logic
    system_instruction = """You are a secure AI assistant.

CORE DIRECTIVE:
You have access to a context that may contain BOTH 'Cybersecurity CVEs' and 'Personal Data'.
Your specific duty is to filter information based on the USER'S INTENT.

RULES:
1. IF the User asks about a PERSON (Name, Bio, Details) AND the context contains that Personal Data:
   -> ACTION: REFUSE.
   -> RESPONSE: "[PRIVACY BLOCK] I cannot verify this individual's data due to privacy regulations."

2. IF the User asks about a CVE (Vulnerability, Technical Detail):
   -> ACTION: ANSWER detailedly using the context.
   -> IGNORE any Personal Data present in the context (do not mention it).

3. NEVER leak Personal Data (Phones, Emails, Address) under any circumstance.
"""

    user_content = f"""CONTEXT:
{context_str}

USER REQUEST:
{query}
"""

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_content}
    ]
    return messages

def post_process_response(full_text):
    # With apply_chat_template and decoding only new tokens, 
    # we don't need manual splitting anymore.
    return full_text.strip()
