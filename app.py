import concurrent.futures
import os
import google.generativeai as genai
from groq import Groq
import streamlit as st

# Environment variables se keys lena
gemini_key = os.getenv("GEMINI_API_KEY")
groq_key = os.getenv("GROQ_API_KEY")

if gemini_key:
  genai.configure(api_key=gemini_key.strip())

groq_client = Groq(api_key=groq_key.strip()) if groq_key else None


# Model 1: Google Gemini Flash
def fetch_gemini(prompt):
  try:
    model = genai.GenerativeModel("gemini-1.5-flash-latest")
    res = model.generate_content(prompt)
    return "Google Gemini", res.text
  except Exception as e:
    return "Google Gemini", f"Error: {e}"


# Model 2: Meta Llama 70B via Groq
def fetch_llama_70b(prompt):
  try:
    res = groq_client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
    )
    return "Llama 70B", res.choices[0].message.content
  except Exception as e:
    return "Llama 70B", f"Error: {e}"


# Model 3: Llama 8B via Groq
def fetch_llama_8b(prompt):
  try:
    res = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
    )
    return "Llama 8B", res.choices[0].message.content
  except Exception as e:
    return "Llama 8B", f"Error: {e}"


# Evaluator / Judge Model
def judge_responses(prompt, answers):
  judge_prompt = f"""
    Sawal: {prompt}

    Answers:
    1. Gemini: {answers.get('Google Gemini', '')}
    2. Llama 70B: {answers.get('Llama 70B', '')}
    3. Llama 8B: {answers.get('Llama 8B', '')}

    Task:
    - Teeno answers ko evaluate karo.
    - Sabse accurate aur helpful response chun kar final answer provide karo.
    - Shuruat me 1 line me batao kaunsa model best tha, fir final clean answer do.
    """
  try:
    model = genai.GenerativeModel("gemini-1.5-flash-latest")
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

user_prompt = st.text_area("Apna sawal yahan likhein:", height=100)

if st.button("Poochho aur Best Answer Chuno"):
  if not user_prompt.strip():
    st.warning("Pehle sawal type karein!")
  elif not gemini_key or not groq_key:
    st.error("Streamlit Secrets mein API keys missing hain.")
  else:
    with st.spinner("Models se parallel responses liye ja rahe hain..."):
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
        st.write(results.get("Google Gemini"))
      with col2:
        st.markdown("**Llama 70B**")
        st.write(results.get("Llama 70B"))
      with col3:
        st.markdown("**Llama 8B**")
        st.write(results.get("Llama 8B"))
