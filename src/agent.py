import ast
import json
import operator
import sys
import ollama
from rag import retrieve, LLM

OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported expression")


def calculator(expression):
    try:
        return str(_eval(ast.parse(expression, mode="eval").body))
    except Exception as e:
        return f"error: {e}"


def search_docs(query):
    docs, metas = retrieve(query, k=4)
    return "\n\n".join(f"[{m['title']}] {d}" for d, m in zip(docs, metas))


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
                        "description": "Search the Bangla Wikipedia knowledge base about Bangladesh and its history, places, language and people.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "search query in Bangla"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate an arithmetic expression such as 2022 - 1861 or 3.5 * 4.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
]

FUNCS = {"search_docs": search_docs, "calculator": calculator}

SYSTEM = (
    "তুমি একজন সহায়ক এজেন্ট। তথ্যের জন্য search_docs এবং হিসাবের জন্য calculator ব্যবহার করো। "
    "প্রয়োজনে একাধিকবার টুল ব্যবহার করো। শুধু টুলের তথ্যের ভিত্তিতে বাংলায় উত্তর দাও। "
    "তথ্য না পেলে বলো 'তথ্য পাওয়া যায়নি'।"
)


def run(question, max_steps=5):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]
    for _ in range(max_steps):
        msg = ollama.chat(model=LLM, messages=messages, tools=TOOLS)["message"]
        messages.append(msg)
        if not msg.tool_calls:
            return msg.content
        for call in msg.tool_calls:
            name, args = call.function.name, call.function.arguments
            print(f"[tool] {name}({json.dumps(args, ensure_ascii=False)})")
            fn = FUNCS.get(name)
            result = fn(**args) if fn else f"unknown tool: {name}"
            messages.append({"role": "tool", "tool_name": name, "content": result})
    return "ধাপের সীমা শেষ হয়েছে।"


if __name__ == "__main__":
    print(run(" ".join(sys.argv[1:])))