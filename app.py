import gradio as gr
import pandas as pd
import os
import google.generativeai as genai
from datetime import datetime

# Load API Key
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY environment variable not set")
genai.configure(api_key=google_api_key)

# Use the latest model
MODEL_NAME = "gemini-1.5-flash"  # or "gemini-1.5-pro" for more complex analysis

def analyze_bank_statement(file):
    try:
        # Read file based on extension
        if file.name.endswith(".csv"):
            df = pd.read_csv(file.name)
        elif file.name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file.name)
        else:
            return "❌ Unsupported file format. Please upload CSV or Excel."

        # Clean column names - CORRECTED VERSION
        df.columns = (
            df.columns
            .astype(str)  # Ensure all columns are strings
            .str.lower()  # Convert to lowercase
            .str.strip()  # Remove whitespace
            .str.replace('[^a-z0-9]', '_', regex=True)  # Replace special chars
        )

        # Find relevant columns with flexible matching
        amount_col = next(
            (col for col in df.columns 
             if any(x in col for x in ['amount', 'amt', 'value', 'balance', 'debit', 'credit'])),
            None
        )
        
        date_col = next(
            (col for col in df.columns 
             if any(x in col for x in ['date', 'time', 'posted'])),
            None
        )
        
        desc_col = next(
            (col for col in df.columns 
             if any(x in col for x in ['desc', 'particular', 'transaction', 'details', 'narration'])),
            None
        )

        if not amount_col:
            return f"❌ Could not find amount column. Available columns: {list(df.columns)}"

        # Prepare data - limit to 500 rows for efficiency
        cols_to_keep = [c for c in [date_col, desc_col, amount_col] if c is not None]
        df = df[cols_to_keep].dropna().head(500)
        
        # Format transactions nicely
        transactions = []
        for _, row in df.iterrows():
            parts = []
            if date_col and pd.notna(row[date_col]):
                try:
                    date_str = str(row[date_col])
                    if isinstance(row[date_col], (datetime, pd.Timestamp)):
                        date_str = row[date_col].strftime('%Y-%m-%d')
                    parts.append(f"Date: {date_str}")
                except:
                    pass
            if desc_col and pd.notna(row[desc_col]):
                parts.append(f"Description: {str(row[desc_col])}")
            if amount_col and pd.notna(row[amount_col]):
                parts.append(f"Amount: {float(row[amount_col]):.2f}")
            transactions.append(", ".join(parts))
        
        statement_text = "\n".join(transactions)

        # Generate analysis
        model = genai.GenerativeModel(MODEL_NAME)
        prompt = f"""You are a professional financial advisor analyzing bank statements. Provide:
1. Spending categories and patterns (essential vs discretionary)
2. Monthly expense estimates and potential savings
3. Investment recommendations based on cash flow
4. Any unusual transactions worth reviewing
Format your response with clear headings and bullet points. Be specific and actionable.
Recent Transactions:
{statement_text}
"""
        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        return f"❌ Error: {str(e)}"

# Gradio Interface
with gr.Blocks(title="💰 AI Financial Advisor") as app:
    gr.Markdown("# 💰 AI Financial Advisor")
    gr.Markdown("Upload your bank statement (CSV/Excel) for personalized financial analysis")
    
    with gr.Row():
        file_input = gr.File(label="Bank Statement", file_types=[".csv", ".xlsx", ".xls"])
        submit_btn = gr.Button("Analyze", variant="primary")
    
    output = gr.Textbox(label="Financial Analysis", interactive=False)
    
    submit_btn.click(
        fn=analyze_bank_statement,
        inputs=file_input,
        outputs=output
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
