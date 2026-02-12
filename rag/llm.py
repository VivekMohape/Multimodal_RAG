from groq import Groq


def ask_groq(context, question, api_key):
    client = Groq(api_key=api_key)

    prompt = f"""
Answer using only the context below.

Context:
{context}

Question: {question}

Answer:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content
