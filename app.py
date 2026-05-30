import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="CSV AI Agent",
    page_icon="📊",
    layout="wide"
)

st.title("📊 CSV AI Agent (OpenRouter + GPT)")

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("⚙️ Configuration")

api_key = st.sidebar.text_input(
    "1. Enter OpenRouter API Key",
    type="password"
)

uploaded_file = st.sidebar.file_uploader(
    "2. Upload CSV File",
    type=["csv"]
)

temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.2
)

st.sidebar.divider()

st.sidebar.info(
    "Upload a CSV file and ask questions or generate charts using GPT."
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

if uploaded_file:

    try:
        df = pd.read_csv(uploaded_file)

        tab1, tab2 = st.tabs(
            ["💬 Chat with Data", "📊 AI Visualizations"]
        )

        # ==================================================
        # TAB 1 - CHAT
        # ==================================================

        with tab1:

            st.subheader("📄 Dataset Preview")

            st.dataframe(df.head())

            st.divider()

            user_input = st.text_area(
                "Ask a question about your data",
                placeholder="What is the average sales amount?"
            )

            if st.button("Generate Answer"):

                if not api_key:
                    st.warning("Please enter your OpenRouter API Key.")

                elif not user_input:
                    st.warning("Please enter a question.")

                else:

                    try:

                        client = OpenAI(
                            api_key=api_key,
                            base_url="https://openrouter.ai/api/v1"
                        )

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

                        with st.spinner("Analyzing data..."):

                            response = client.chat.completions.create(
                                model="openai/gpt-latest",
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

        # ==================================================
        # TAB 2 - VISUALIZATION
        # ==================================================

        with tab2:

            st.subheader("📊 Generate AI Charts")

            viz_input = st.text_input(
                "Describe the chart you want",
                placeholder="Show a bar chart of sales by category"
            )

            if st.button("Generate Visualization"):

                if not api_key:
                    st.warning("Please enter your OpenRouter API Key.")

                elif not viz_input:
                    st.warning("Please describe a visualization.")

                else:

                    try:

                        client = OpenAI(
                            api_key=api_key,
                            base_url="https://openrouter.ai/api/v1"
                        )

                        prompt = f"""
Generate ONLY executable Python code.

DataFrame name is df.

User Request:
{viz_input}

Rules:
1. Use matplotlib and seaborn.
2. Create figure using:

fig, ax = plt.subplots(figsize=(10,6))

3. Store chart in variable fig.
4. No markdown.
5. No explanation.
6. No plt.show().
"""

                        with st.spinner("Creating chart..."):

                            response = client.chat.completions.create(
                                model="openai/gpt-latest",
                                messages=[
                                    {
                                        "role": "user",
                                        "content": prompt
                                    }
                                ],
                                temperature=0
                            )

                            code = response.choices[0].message.content

                            code = code.replace(
                                "```python", ""
                            ).replace(
                                "```", ""
                            )

                            local_vars = {
                                "df": df,
                                "pd": pd,
                                "plt": plt,
                                "sns": sns
                            }

                            exec(code, {}, local_vars)

                            fig = local_vars.get("fig")

                            if fig:
                                st.pyplot(fig)
                            else:
                                st.warning(
                                    "No chart was generated."
                                )

                            with st.expander(
                                "View Generated Python Code"
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
