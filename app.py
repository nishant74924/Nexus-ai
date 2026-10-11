import concurrent.futures
import os
import google.generativeai as genai
from groq import Groq
import streamlit as st

# Secrets / Environment variables se keys lena
gemini_key = os.getenv("GEMINI_API_KEY")
groq_key = os.getenv("GROQ_API_KEY")

if gemini_key:
  genai.configure(api_key=gemini_key)

groq_client = Groq(api_key=groq_key) if groq_key else None


# Model 1: Google Gemini Flash (Free)
def fetch_gemini(prompt):
  try:
    model = genai.GenerativeModel("gemini-1.5-flash")
    res = model.generate_content(prompt)
    return "Google Gemini", res.text
  except Exception as e:
    return "Google Gemini", f"Error: {e}"


# Model 2: Meta Llama 3.3 70B via Groq (Free)
def fetch_llama_70b(prompt):
  try:
    res = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
    )
    return "Llama 3.3 (70B)", res.choices[0].message.content
  except Exception as e:
    return "Llama 3.3 (70B)", f"Error: {e}"


# Model 3: Llama 3.1 8B via Groq (Free)
def fetch_llama_8b(prompt):
  try:
    res = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
    )
    return "Llama 3.1 (8B)", res.choices[0].message.content
  except Exception as e:
    return "Llama 3.1 (8B)", f"Error: {e}"


# Evaluator / Judge Model: Jo sabse behtar answer select karega
def judge_responses(prompt, answers):
  judge_prompt = f"""
    Sawal: {prompt}

    Answers:
    1. Gemini: {answers['Google Gemini']}
    2. Llama 3.3: {answers['Llama 3.3 (70B)']}
    3. Llama 3.1: {answers['Llama 3.1 (8B)']}

    Task:
    - Teeno answers ko evaluate karo (accuracy, clarity aur facts check karo).
    - Sabse accurate answer chun kar present karo ya teeno ka best combination banao.
    - Shuruat me 1 line me batao kaunsa model sabse sahi tha aur kyun, fir final detailed answer do.
    """
  try:
    model = genai.GenerativeModel("gemini-1.5-flash")
    res = model.generate_content(judge_prompt)
    return res.text
  except Exception as e:
    return f"Judge Evaluation Error: {e}"


# UI Design
st.set_page_config(page_title="Nexus-AI", layout="wide")
st.title("⚡ Nexus-AI: Multi-Model Best Selector")
st.caption(
    "Google Gemini + Meta Llama parallelly run hokar sabse accurate answer"
    " chunenge."
)

if not gemini_key or not groq_key:
  st.warning(
      "Kripya Streamlit Settings -> Secrets me GEMINI_API_KEY aur GROQ_API_KEY"
      " save karein!"
  )

user_prompt = st.text_area("Apna sawal yahan likhein:", height=100)

if st.button("Poochho aur Best Answer Chuno"):
  if not user_prompt.strip():
    st.warning("Pehle sawal type karein!")
  elif not gemini_key or not groq_key:
    st.error("API Keys missing hain.")
  else:
    with st.spinner("Teeno models se parallel response mangwaya ja raha hai..."):
      with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(fetch_gemini, user_prompt),
            executor.submit(fetch_llama_70b, user_prompt),
            executor.submit(fetch_llama_8b, user_prompt),
        ]
        results = dict([f.result() for f in futures])

    with st.spinner("Judge model best answer evaluate kar raha hai..."):
      best_answer = judge_responses(user_prompt, results)

    st.subheader("🎯 Final Best Answer")
    st.success(best_answer)

    with st.expander("Sabhi Models ke Raw Responses Dekhein"):
      col1, col2, col3 = st.columns(3)
      with col1:
        st.markdown("**Google Gemini**")
        st.write(results["Google Gemini"])
      with col2:
        st.markdown("**Llama 3.3 (70B)**")
        st.write(results["Llama 3.3 (70B)"])
      with col3:
        st.markdown("**Llama 3.1 (8B)**")
        st.write(results["Llama 3.1 (8B)"])
      
