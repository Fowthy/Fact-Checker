import streamlit as st
import streamlit.components.v1 as components
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

# Sidebar for model configuration
with st.sidebar:
    st.header("Model Configuration")

    # Model selection
    model_choice = st.selectbox(
        "Select Model",
        ["gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-5-search-api", "gpt-4o"],
        index=0,
        help="GPT-5 models with reasoning capabilities. Use gpt-5-search-api for web search integration."
    )

    # Reasoning effort (for GPT-5 models)
    if model_choice.startswith("gpt-5"):
        reasoning_effort = st.selectbox(
            "Reasoning Effort",
            ["high", "medium", "low", "minimal"],
            index=0,
            help="Higher reasoning effort = more thorough analysis but slower response"
        )

        # Verbosity
        verbosity = st.selectbox(
            "Response Verbosity",
            ["medium", "high", "low"],
            index=0,
            help="Controls how detailed the explanations are"
        )

        # Web search option (for search-enabled models)
        if model_choice == "gpt-5-search-api":
            enable_web_search = st.checkbox(
                "Enable Web Search",
                value=True,
                help="Allow the model to search the web for fact-checking"
            )
        else:
            enable_web_search = False
    else:
        reasoning_effort = None
        verbosity = None
        enable_web_search = False

    # Streaming option
    enable_streaming = st.checkbox(
        "Enable Streaming",
        value=True,
        help="Show live responses as the model generates them"
    )

    st.divider()
    st.caption(f"Model: {model_choice}")
    if reasoning_effort:
        st.caption(f"Reasoning: {reasoning_effort}")
    if verbosity:
        st.caption(f"Verbosity: {verbosity}")
    st.caption(f"Streaming: {'On' if enable_streaming else 'Off'}")

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
            class="fact-issue fact-issue-{idx}"
            data-issue-id="issue-{idx}"
            style="background-color: {color}; color: #000; padding: 2px 4px; border-radius: 3px; cursor: help; border: 2px solid transparent; transition: all 0.2s ease;"
            title="Description ⬇\n\n{escaped_explanation}">{excerpt}</mark>'''
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
                prompt = f"""
                
                Run a deep research to fact check this text. Identify any misleading information, questionable statements, or missing important information that would confuse the reader.

                IMPORTANT: Respond in English.

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
                {text_input}
"""

                # Define JSON schema for structured output
                json_schema = {
                    "type": "object",
                    "properties": {
                        "issues": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "excerpt": {"type": "string"},
                                    "issue": {"type": "string"},
                                    "type": {"type": "string", "enum": ["misleading", "questionable", "incomplete"]}
                                },
                                "required": ["excerpt", "issue", "type"],
                                "additionalProperties": False
                            }
                        }
                    },
                    "required": ["issues"],
                    "additionalProperties": False
                }

                # Placeholder for streaming output
                status_placeholder = st.empty()
                streaming_placeholder = st.empty()

                # Call OpenAI API - different endpoints for GPT-5 vs GPT-4
                if model_choice.startswith("gpt-5"):
                    # Use Responses API for GPT-5 models
                    api_params = {
                        "model": model_choice,
                        "input": [
                            {"role": "system", "content": "You are a professional fact-checker. Analyze texts thoroughly and identify misleading information, questionable statements, and missing context. Always respond in English."},
                            {"role": "user", "content": prompt}
                        ],
                        "stream": enable_streaming
                    }

                    # Add reasoning effort parameter with summary
                    if reasoning_effort:
                        api_params["reasoning"] = {
                            "effort": reasoning_effort,
                            "summary": "auto"  # Request reasoning summary
                        }

                    # Add verbosity parameter and JSON format
                    text_config = {}
                    if verbosity:
                        text_config["verbosity"] = verbosity

                    # Add JSON schema for structured output
                    # NOTE: Structured outputs (json_schema) don't stream incrementally
                    # They return the complete JSON at the end, so streaming won't show progressive updates
                    if not enable_streaming:
                        text_config["format"] = {
                            "type": "json_schema",
                            "name": "fact_check_results",
                            "schema": json_schema,
                            "strict": True
                        }

                    if text_config:
                        api_params["text"] = text_config

                    # Add web search tool if enabled
                    if enable_web_search:
                        api_params["tools"] = [{"type": "web_search"}]

                    # Make API call using responses endpoint
                    response = client.responses.create(**api_params)

                    if enable_streaming:
                        # Process streaming response (event-based)
                        result_text = ""
                        reasoning_text = ""
                        reasoning_summary = ""

                        status_placeholder.info("🤔 Model is thinking and analyzing...")

                        for event in response:
                            # Handle different event types
                            event_type = getattr(event, 'type', None)

                            # Text delta events contain the actual content
                            if event_type == 'response.output_item.text.delta':
                                if hasattr(event, 'delta'):
                                    result_text += event.delta
                                    streaming_placeholder.markdown(f"**Live Response:**\n```json\n{result_text}\n```")
                                elif hasattr(event, 'text'):
                                    result_text += event.text
                                    streaming_placeholder.markdown(f"**Live Response:**\n```json\n{result_text}\n```")

                            # Done event contains the final item
                            elif event_type == 'response.output_item.done':
                                if hasattr(event, 'item'):
                                    item = event.item
                                    item_type = getattr(item, 'type', None)

                                    # Extract reasoning summary
                                    if item_type == 'reasoning' and hasattr(item, 'summary') and item.summary:
                                        for summary_item in item.summary:
                                            if hasattr(summary_item, 'text'):
                                                reasoning_summary += summary_item.text

                                    # Extract text from message/output_text items
                                    elif item_type in ['message', 'output_text']:
                                        if hasattr(item, 'content') and item.content and not result_text:
                                            for content_item in item.content:
                                                if hasattr(content_item, 'text'):
                                                    result_text += content_item.text
                                        elif hasattr(item, 'text') and not result_text:
                                            result_text = item.text

                            # Completed event contains the final response
                            elif event_type == 'response.completed':
                                if hasattr(event, 'response'):
                                    resp = event.response
                                    if hasattr(resp, 'output') and resp.output:
                                        for output_item in resp.output:
                                            item_type = getattr(output_item, 'type', None)

                                            # Extract reasoning summary
                                            if item_type == 'reasoning':
                                                if hasattr(output_item, 'summary') and output_item.summary:
                                                    for summary_item in output_item.summary:
                                                        if hasattr(summary_item, 'text'):
                                                            reasoning_summary += summary_item.text

                                            # Extract message text
                                            elif item_type in ['message', 'output_text']:
                                                if hasattr(output_item, 'content') and output_item.content and not result_text:
                                                    for content_item in output_item.content:
                                                        if hasattr(content_item, 'text'):
                                                            result_text += content_item.text
                                                elif hasattr(output_item, 'text') and not result_text:
                                                    result_text = output_item.text

                            # Reasoning delta events
                            if event_type == 'response.output_item.reasoning.delta':
                                if hasattr(event, 'delta'):
                                    reasoning_text += event.delta
                                    if reasoning_text:
                                        status_placeholder.info(f"🧠 Reasoning: {reasoning_text[:200]}...")

                        status_placeholder.empty()
                        streaming_placeholder.empty()
                    else:
                        # Non-streaming: wait for complete response
                        result_text = ""
                        reasoning_summary = ""

                        # Extract both reasoning and message from response
                        if hasattr(response, 'output') and response.output:
                            for output_item in response.output:
                                item_type = getattr(output_item, 'type', None)

                                # Extract reasoning summary
                                if item_type == 'reasoning':
                                    if hasattr(output_item, 'summary') and output_item.summary:
                                        for summary_item in output_item.summary:
                                            if hasattr(summary_item, 'text'):
                                                reasoning_summary += summary_item.text

                                # Extract message text (the actual response)
                                elif item_type == 'message' or item_type == 'output_text':
                                    # Try different ways to get the text
                                    if hasattr(output_item, 'content') and output_item.content:
                                        # Content is a list of content items
                                        for content_item in output_item.content:
                                            if hasattr(content_item, 'text'):
                                                result_text += content_item.text
                                    elif hasattr(output_item, 'text'):
                                        result_text += output_item.text

                                # Fallback: if no type specified, try to extract text
                                elif not result_text:
                                    if hasattr(output_item, 'text'):
                                        result_text = output_item.text
                                    elif hasattr(output_item, 'content'):
                                        if isinstance(output_item.content, str):
                                            result_text = output_item.content
                                        elif hasattr(output_item.content, 'text'):
                                            result_text = output_item.content.text

                else:
                    # Use Chat Completions API for GPT-4 models
                    response = client.chat.completions.create(
                        model=model_choice,
                        messages=[
                            {"role": "system", "content": "You are a professional fact-checker. Analyze texts thoroughly and identify misleading information, questionable statements, and missing context. Always respond in English."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.3,
                        response_format={"type": "json_object"},
                        stream=enable_streaming
                    )

                    if enable_streaming:
                        # Process streaming response for GPT-4
                        result_text = ""
                        status_placeholder.info("🤔 Analyzing text...")

                        for chunk in response:
                            if chunk.choices[0].delta.content:
                                result_text += chunk.choices[0].delta.content
                                streaming_placeholder.markdown(f"**Live Response:**\n```json\n{result_text}\n```")

                        status_placeholder.empty()
                        streaming_placeholder.empty()
                    else:
                        # Non-streaming: get complete response
                        result_text = response.choices[0].message.content

                # Parse results
                try:
                    # Ensure result_text is a string
                    if isinstance(result_text, list):
                        result_text = result_text[0] if result_text else ""

                    # Extract text from ResponseOutputText or similar objects
                    if not isinstance(result_text, str):
                        if hasattr(result_text, 'text'):
                            result_text = result_text.text
                        elif hasattr(result_text, 'content'):
                            result_text = result_text.content
                        else:
                            result_text = str(result_text) if result_text else ""

                    if not result_text or result_text.strip() == "":
                        st.error("No response received from the API. Please check your API key and model availability.")
                        issues = []
                    else:
                        result_json = json.loads(result_text)
                        # Handle different possible JSON structures
                        if isinstance(result_json, dict):
                            issues = result_json.get('issues', result_json.get('findings', []))
                            if not isinstance(issues, list):
                                issues = [result_json]
                        else:
                            issues = result_json
                except json.JSONDecodeError as e:
                    st.error(f"Failed to parse fact-check results. Error: {str(e)}")
                    st.text(result_text)
                    issues = []

                st.success("Fact-check complete!")

                # Display reasoning summary if available
                if model_choice.startswith("gpt-5") and 'reasoning_summary' in locals() and reasoning_summary:
                    with st.expander("🧠 Reasoning Summary", expanded=False):
                        st.markdown(reasoning_summary)

                # Display raw API response at the bottom
                with st.expander("📋 Raw API Response (Debug)", expanded=False):
                    try:
                        # Show the actual raw response from OpenAI
                        if 'response' in locals():
                            st.write("**Full Response Object:**")
                            # Convert to dict for display
                            if hasattr(response, 'model_dump'):
                                st.json(response.model_dump())
                            elif hasattr(response, 'dict'):
                                st.json(response.dict())
                            else:
                                st.write(response)

                        st.write("**Extracted Values:**")
                        st.json({
                            "model": model_choice,
                            "reasoning_summary_length": len(reasoning_summary) if 'reasoning_summary' in locals() and reasoning_summary else 0,
                            "result_text_length": len(result_text) if result_text else 0,
                            "parsed_issues_count": len(issues) if issues else 0
                        })
                    except Exception as e:
                        st.error(f"Error displaying response: {e}")

                if issues:
                    # Original Text (full width)
                    st.markdown("### Original Text (Highlighted)")

                    highlighted_text = highlight_text(text_input, issues)

                    # Use components.html with large height and no scrolling
                    html_content = f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                    <style>
                    body {{
                        font-family: "Source Sans Pro", sans-serif;
                        font-size: 16px;
                        line-height: 1.6;
                        color: white;
                        margin: 0;
                        padding: 10px;
                        overflow: hidden;
                    }}
                    .fact-issue {{
                        transition: all 0.2s ease;
                    }}
                    .fact-issue:hover {{
                        border: 2px solid #333 !important;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
                    }}
                    </style>
                    </head>
                    <body>
                        {highlighted_text}
                    </body>
                    </html>
                    """
                    # Use very large height to accommodate any text
                    components.html(html_content, height=1000, scrolling=True)

                    # Legend
                    st.markdown("""
                    <div style="margin-top: 20px; margin-bottom: 30px; padding: 10px; background-color: #f0f0f0; border-radius: 5px; color: #000;">
                        <b>Legend:</b><br>
                        <mark style="background-color: #ffcccc; padding: 2px 4px; color: #000;">Misleading</mark>
                        <mark style="background-color: #fff4cc; padding: 2px 4px; color: #000;">Questionable</mark>
                        <mark style="background-color: #cce5ff; padding: 2px 4px; color: #000;">Incomplete</mark>
                        <br><br>
                        <small>💡 Hover over highlights to see full explanations. See details below ⬇</small>
                    </div>
                    """, unsafe_allow_html=True)

                    # Issues section (full width below)
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
                import traceback
                with st.expander("Show error details"):
                    st.code(traceback.format_exc())
