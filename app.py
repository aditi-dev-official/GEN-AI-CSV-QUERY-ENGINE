import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI

# ==================================================
# CONFIG
# ==================================================

MODEL_NAME = "qwen/qwen3-32b"

st.set_page_config(
    page_title="CSV AI Agent",
    page_icon="📊",
    layout="wide"
)

st.title("📊 CSV AI Agent (OpenRouter + Qwen)")

# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.header("⚙️ Configuration")

api_key = st.sidebar.text_input(
    "OpenRouter API Key",
    type="password"
)

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

temperature = st.sidebar.slider(
    "Temperature",
    0.0,
    1.0,
    0.2
)

st.sidebar.divider()

st.sidebar.info(
    f"Model: {MODEL_NAME}"
)

# ==================================================
# OPENROUTER CLIENT
# ==================================================

def get_client(api_key):
    return OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )

# ==================================================
# MAIN APP
# ==================================================

if uploaded_file is not None:

    try:

        df = pd.read_csv(uploaded_file)

        tab1, tab2 = st.tabs(
            ["💬 Chat With Data", "📊 AI Visualizations"]
        )

        # ==========================================
        # TAB 1 - CHAT
        # ==========================================

        with tab1:

            st.subheader("📄 Dataset Preview")

            st.dataframe(df.head())

            user_input = st.text_area(
                "Ask a question about your data",
                placeholder="What is the average sales amount?"
            )

            if st.button("Generate Answer"):

                if not api_key:
                    st.warning("Please enter your OpenRouter API Key.")

                elif not user_input.strip():
                    st.warning("Please enter a question.")

                else:

                    try:

                        client = get_client(api_key)

                        sample_data = df.head(20).to_string()

                        prompt = f"""
You are an expert data analyst.

Dataset Columns:
{list(df.columns)}

Dataset Sample:
{sample_data}

User Question:
{user_input}

Provide a clear and concise answer.
"""

                        with st.spinner("Analyzing..."):

                            response = client.chat.completions.create(
                                model=MODEL_NAME,
                                messages=[
                                    {
                                        "role": "user",
                                        "content": prompt
                                    }
                                ],
                                temperature=temperature
                            )

                            answer = response.choices[0].message.content

                            st.success("Answer Generated")
                            st.write(answer)

                    except Exception as e:
                        st.error(f"Error: {e}")

        # ==========================================
        # TAB 2 - VISUALIZATIONS
        # ==========================================

        with tab2:

            st.subheader("📊 Generate AI Charts")

            viz_input = st.text_area(
                "Describe the chart you want",
                height=150,
                placeholder="""
Examples:
• Create a bar chart of sales by category
• Show a histogram of transaction amounts
• Create a line chart of monthly revenue
• Show top 10 products by sales
"""
            )

            if st.button("Generate Visualization"):

                if not api_key:
                    st.warning("Please enter your OpenRouter API Key.")

                elif not viz_input.strip():
                    st.warning("Please describe a chart.")

                else:

                    try:

                        client = get_client(api_key)

                        prompt = f"""
You are an expert Python data visualization engineer.

Dataset Columns:
{list(df.columns)}

User Request:
{viz_input}

Generate ONLY executable Python code.

Requirements:

- DataFrame name is df
- Use matplotlib and seaborn
- Start with:

fig, ax = plt.subplots(figsize=(10,6))

- Use only columns from the dataset
- Store final chart in variable fig
- Do not use markdown
- Do not use explanations
- Do not use comments
- Do not use plt.show()
"""

                        with st.spinner("Creating Chart..."):

                            response = client.chat.completions.create(
                                model=MODEL_NAME,
                                messages=[
                                    {
                                        "role": "user",
                                        "content": prompt
                                    }
                                ],
                                temperature=0
                            )

                            code = response.choices[0].message.content

                            code = (
                                code.replace("```python", "")
                                .replace("```", "")
                                .strip()
                            )

                            local_vars = {
                                "df": df,
                                "pd": pd,
                                "plt": plt,
                                "sns": sns
                            }

                            exec(code, {}, local_vars)

                            fig = local_vars.get("fig")

                            if fig is None:
                                fig = plt.gcf()

                            st.pyplot(fig)

                            plt.close(fig)

                            with st.expander(
                                "🔍 View Generated Python Code"
                            ):
                                st.code(
                                    code,
                                    language="python"
                                )

                    except Exception as e:
                        st.error(
                            f"Visualization Error: {e}"
                        )

    except Exception as e:
        st.error(f"CSV Error: {e}")

else:
    st.info("👈 Upload a CSV file to get started.")
