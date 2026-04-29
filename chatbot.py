import os
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq

# ── Load API Key ─────────────────────────────────────
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ── Initialize LLM with streaming ────────────────────
llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0.7,
    streaming=True,
    api_key=GROQ_API_KEY
)

# ── Prompt Template (Travel-only restriction) ────────
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a Travel Agent chatbot. "
     "Answer ONLY travel-related queries (flights, hotels, destinations, travel tips). "
     "If the question is not travel-related, respond EXACTLY with: "
     "'I can’t help with it.'"),
    ("placeholder", "{history}"),
    ("human", "{input}")
])

# ── Memory (List-based) ──────────────────────────────
conversation_history = []
turn_count = 0

print("✈️ Travel Agent Chatbot Ready! (type 'exit' to quit)\n")

# ────────────────────────────────────────────────────
# MAIN LOOP
# ────────────────────────────────────────────────────
while True:

    # STEP 5: User input
    user_input = input("You: ").strip()

    if not user_input:
        continue

    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Travel Agent: Safe travels! Goodbye! ✈️")
        break

    conversation_history.append(HumanMessage(content=user_input))
    turn_count += 1

    # STEP 6: Streaming response
    chain = prompt | llm

    print("Travel Agent: ", end="", flush=True)
    full_response = ""

    for chunk in chain.stream({
        "history": conversation_history[:-1],
        "input": user_input
    }):
        print(chunk.content, end="", flush=True)
        full_response += chunk.content

    print()

    # STEP 7: Save response
    conversation_history.append(AIMessage(content=full_response))

    # STEP 8/9: Summarization every 5 turns
    if turn_count % 5 == 0 and len(conversation_history) > 2:
        print("\n[Summarizing chat history...]\n")

        summary_prompt = f"""Summarize this travel conversation in 2-3 sentences,
keeping important travel details:
{chr(10).join([f"{type(m).__name__}: {m.content}" for m in conversation_history])}"""

        summary_response = llm.invoke([HumanMessage(content=summary_prompt)])
        summary_text = summary_response.content

        conversation_history = [
            AIMessage(content=f"[Summary]: {summary_text}")
        ]

        print("[History compressed.]\n")
