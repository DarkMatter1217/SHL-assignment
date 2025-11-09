import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage

try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    print("🔑 Gemini API key loaded successfully from Streamlit secrets.")
except Exception:
    raise ValueError(
        "❌ GEMINI_API_KEY not found in .streamlit/secrets.toml.\n\n"
        "Create a file at '.streamlit/secrets.toml' with:\n\n"
        'GEMINI_API_KEY = "AIzaSyYourRealGeminiKeyHere"'
    )

def get_gemini_llm(model_name: str = "gemini-2.5-flash", temperature: float = 0.2):
    """
    Returns a LangChain ChatGoogleGenerativeAI LLM configured with Gemini 2.5 Flash
    using the API key securely loaded from Streamlit secrets.

    Args:
        model_name (str): Gemini model name, default "gemini-2.5-flash"
        temperature (float): Response creativity level, default 0.2

    Returns:
        ChatGoogleGenerativeAI: LangChain LLM instance ready to use.
    """
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        google_api_key=GEMINI_API_KEY,
    )
    return llm

def test_gemini_connection():
    """
    Simple connection test for Gemini API using LangChain.
    Prints status & model response to verify everything works.
    """
    print("\n⏳ Testing Gemini 2.5 Flash connection via LangChain...\n")

    try:
        llm = get_gemini_llm(model_name="gemini-2.5-flash", temperature=0.1)

        response = llm.invoke([
            HumanMessage(
                content="Say '✅ Gemini 2.5 Flash key working perfectly via LangChain and Streamlit secrets.'"
            )
        ])

        print("✅ Connection successful!")
        print("Gemini Response:", response.content)

    except Exception as e:
        print("❌ Gemini connection test failed:")
        print(str(e))

if __name__ == "__main__":
    print("==============================================")
    print("   🔍 GEMINI 2.5 FLASH API KEY VALIDATION TEST")
    print("==============================================")
    test_gemini_connection()
    print("==============================================")
