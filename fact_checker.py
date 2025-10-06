import streamlit as st
import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="Fact Checker", page_icon="🔍", layout="wide")

st.title("🔍 Fact Checker")
st.write("Enter text below to fact-check it for misleading, questionable, or incomplete information.")

# Text input area
text_input = st.text_area(
    "Text to fact-check:",
    height=300,
    placeholder="Paste your text here..."
)

def highlight_text(original_text, issues):
    """Highlight problematic sections in the text"""
    if not issues:
        return original_text

    # Sort issues by position in text (longest first to avoid substring issues)
    sorted_issues = sorted(issues, key=lambda x: len(x.get('excerpt', '')), reverse=True)

    highlighted = original_text
    replacements = []

    for i, issue in enumerate(sorted_issues):
        excerpt = issue.get('excerpt', '')
        issue_type = issue.get('type', 'questionable')
        explanation = issue.get('issue', 'No explanation')

        # Color code by type
        colors = {
            'misleading': '#ffcccc',  # light red
            'questionable': '#fff4cc',  # light yellow
            'incomplete': '#cce5ff'  # light blue
        }
        color = colors.get(issue_type, '#fff4cc')

        if excerpt and excerpt in highlighted:
            # Create a unique placeholder
            placeholder = f"___HIGHLIGHT_{i}___"
            replacements.append((placeholder, excerpt, color, i, explanation))
            highlighted = highlighted.replace(excerpt, placeholder, 1)

    # Replace placeholders with HTML
    for placeholder, excerpt, color, idx, explanation in replacements:
        # Escape quotes in explanation for HTML attribute
        escaped_explanation = explanation.replace('"', '&quot;').replace("'", '&#39;')

        html_highlight = f'''<mark
            id="highlight-{idx}"
            class="fact-issue"
            data-issue-id="issue-{idx}"
            style="background-color: {color}; color: #000; padding: 2px 4px; border-radius: 3px; cursor: pointer; border: 2px solid transparent;"
            onmouseover="this.style.border='2px solid #333'; this.style.fontWeight='bold';"
            onmouseout="this.style.border='2px solid transparent'; this.style.fontWeight='normal';"
            onclick="document.getElementById('issue-{idx}').scrollIntoView({{behavior: 'smooth', block: 'center'}}); document.getElementById('issue-{idx}').style.boxShadow='0 0 15px rgba(0,123,255,0.5)'; setTimeout(() => document.getElementById('issue-{idx}').style.boxShadow='none', 2000);"
            title="{escaped_explanation[:200]}{'...' if len(escaped_explanation) > 200 else ''}">{excerpt}</mark>'''
        highlighted = highlighted.replace(placeholder, html_highlight)

    return highlighted

# Fact-check button
if st.button("Fact Check", type="primary"):
    if not text_input.strip():
        st.error("Please enter some text to fact-check.")
    elif not os.getenv("OPENAI_API_KEY"):
        st.error("OpenAI API key not found. Please add OPENAI_API_KEY to your .env file.")
    else:
        with st.spinner("Performing deep research and fact-checking..."):
            try:
                # Prepare the prompt
                prompt = f"""Run a deep research to fact check this text. Identify any misleading information, questionable statements, or missing important information that would confuse the reader.

IMPORTANT: Respond in the SAME LANGUAGE as the input text. If the text is in Bulgarian, respond in Bulgarian. If it's in English, respond in English.

Return your analysis as a JSON object with the following structure:
{{
  "issues": [
    {{
      "excerpt": "exact text from the original that has an issue",
      "issue": "explanation of what is wrong, misleading, or missing",
      "type": "misleading" | "questionable" | "incomplete"
    }}
  ]
}}

If no issues are found, return: {{"issues": []}}

Text to fact-check:
{text_input}"""

                # Call OpenAI API
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a professional fact-checker. Analyze texts thoroughly and identify misleading information, questionable statements, and missing context. Always respond in the same language as the input text."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    response_format={"type": "json_object"}
                )

                # Parse results
                result_text = response.choices[0].message.content

                # Try to parse JSON
                try:
                    result_json = json.loads(result_text)
                    # Handle different possible JSON structures
                    if isinstance(result_json, dict):
                        issues = result_json.get('issues', result_json.get('findings', []))
                        if not isinstance(issues, list):
                            issues = [result_json]
                    else:
                        issues = result_json
                except json.JSONDecodeError:
                    st.error("Failed to parse fact-check results. Please try again.")
                    st.text(result_text)
                    issues = []

                st.success("Fact-check complete!")

                if issues:
                    # Create two columns
                    col1, col2 = st.columns([1, 1])

                    with col1:
                        st.markdown("### Original Text (Highlighted)")
                        st.markdown('<div style="background-color: transparent; color: inherit;">', unsafe_allow_html=True)
                        highlighted_text = highlight_text(text_input, issues)
                        st.markdown(highlighted_text, unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)

                        # Legend
                        st.markdown("""
                        <div style="margin-top: 20px; padding: 10px; background-color: #f0f0f0; border-radius: 5px; color: #000;">
                            <b>Legend:</b><br>
                            <mark style="background-color: #ffcccc; padding: 2px 4px; color: #000;">Misleading</mark>
                            <mark style="background-color: #fff4cc; padding: 2px 4px; color: #000;">Questionable</mark>
                            <mark style="background-color: #cce5ff; padding: 2px 4px; color: #000;">Incomplete</mark>
                            <br><br>
                            <small>💡 Hover over highlights to see explanations, click to jump to details →</small>
                        </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        st.markdown("### Issues Found")
                        for i, issue in enumerate(issues):
                            issue_type = issue.get('type', 'questionable').title()
                            excerpt = issue.get('excerpt', 'N/A')
                            explanation = issue.get('issue', 'No explanation provided')

                            # Color for issue type badge
                            type_colors = {
                                'Misleading': '🔴',
                                'Questionable': '🟡',
                                'Incomplete': '🔵'
                            }
                            icon = type_colors.get(issue_type, '⚪')

                            st.markdown(f"""
                            <div id="issue-{i}" style="padding: 10px; margin-bottom: 15px; border-left: 3px solid #ccc; background-color: #f9f9f9; color: #000; transition: box-shadow 0.3s;">
                                <b>{icon} Issue #{i+1}: {issue_type}</b><br>
                                <i style="color: #666;">"{excerpt[:100]}{'...' if len(excerpt) > 100 else ''}"</i><br><br>
                                <div style="color: #000;">{explanation}</div>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.success("✅ No issues found! The text appears to be accurate and complete.")

            except Exception as e:
                st.error(f"Error: {str(e)}")
