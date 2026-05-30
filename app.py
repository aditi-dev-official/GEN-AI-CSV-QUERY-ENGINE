## For app.py
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from langchain_groq import ChatGroq
from langchain_experimental.agents import create_pandas_dataframe_agent

# ---- PAGE CONFIG ----
st.set_page_config(page_title="CSV AI Agent", page_icon="📊", layout="wide")

st.title("📊 CSV AI Agent (Groq + LangChain)")

# ---- SIDEBAR (Settings & Upload) ----
st.sidebar.header("⚙️ Configuration")

api_key = st.sidebar.text_input("1. Enter GROQ API Key", type="password")

# File Uploader in Sidebar
uploaded_file = st.sidebar.file_uploader("2. Upload your CSV file", type=["csv"])

temperature = st.sidebar.slider("Model Temperature", 0.0, 1.0, 0.0)

st.sidebar.divider()
st.sidebar.info("This agent uses Llama-3.3-70b to analyze and visualize your data.")

# ---- MAIN CONTENT ----
if uploaded_file:
    # Load Data
    df = pd.read_csv(uploaded_file)

    # Tabs for better organization
    tab1, tab2 = st.tabs(["💬 Chat with Data", "🎨 AI Visualizations"])

    with tab1:
        st.subheader("📄 Data Preview")
        st.dataframe(df.head(5))
        st.divider()
        
        user_input = st.text_area("🗨️ Ask a question about this data:", placeholder="e.g., What is the average value of the sales column?")

        if st.button("Generate Answer"):
            if not api_key:
                st.warning("⚠️ Please enter your GROQ API key in the sidebar.")
            elif not user_input:
                st.warning("⚠️ Please ask a question first.")
            else:
                try:
                    os.environ["GROQ_API_KEY"] = api_key
                    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=temperature)
                    agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)

                    with st.spinner("🤖 Thinking..."):
                        response = agent.run(user_input)
                        st.success("✅ Answer:")
                        st.write(response)
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    with tab2:
        st.subheader("📊 Chat to Generate Charts")
        st.write("Ask for a chart (e.g., 'Show a bar chart of the top 10 categories')")
        
        viz_input = st.text_input("Describe your visualization:", key="viz_input")

        if st.button("Generate Visualization"):
            if not api_key:
                st.warning("⚠️ Please enter your GROQ API key.")
            elif not viz_input:
                st.warning("⚠️ Please describe a visualization.")
            else:
                try:
                    os.environ["GROQ_API_KEY"] = api_key
                    # Lower temperature is better for generating code
                    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0) 
                    
                    agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)

                    with st.spinner("📈 Creating your chart..."):
                        # Updated prompt: instruct the agent to use the fig, ax pattern
                        full_prompt = (
                            f"Create a visualization for: {viz_input}. "
                            "Instruction: Use matplotlib and seaborn. Define the figure using 'fig, ax = plt.subplots()'. "
                            "Do not call plt.show()."
                        )
                        
                        agent.run(full_prompt)
                        
                        # Display the figure currently in matplotlib's buffer
                        st.pyplot(plt.gcf())
                        plt.close('all') # Clear memory to prevent chart overlapping

                except Exception as e:
                    st.error(f"Error generating plot: {str(e)}")

else:
    st.info("👈 Please upload a CSV file in the sidebar to get started.")
