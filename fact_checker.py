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

# Initialize session state for text input if not exists
if 'main_text_input' not in st.session_state:
    st.session_state.main_text_input = ""

# Initialize session state for fact-check results
if 'fact_check_results' not in st.session_state:
    st.session_state.fact_check_results = None

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

        # Web search option (for GPT-5 models)
        enable_web_search = st.checkbox(
            "Enable Web Search",
            value=True,
            help="Allow the model to search the web for fact-checking"
        )

        if enable_web_search:
            search_context_size = st.selectbox(
                "Search Context Size",
                ["low", "medium", "high"],
                index=1,
                help="Amount of web search context to include"
            )
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
    if model_choice.startswith("gpt-5") and enable_web_search:
        st.caption(f"Web Search: On ({search_context_size})")
    st.caption(f"Streaming: {'On' if enable_streaming else 'Off'}")

st.markdown("### Text to fact-check:")
st.caption("💡 Paste text from Google Docs below. Links will be automatically preserved.")

# Create a helper component to process paste with links
paste_helper = """
<div id="paste-helper" contenteditable="true" style="
    position: absolute;
    left: -9999px;
    width: 1px;
    height: 1px;
    opacity: 0;
"></div>

<script>
// Helper function to parse HTML and extract text with links
function parseHtmlWithLinks(html) {
    const parser = new DOMParser();
    const doc = parser.parseFromString(html, 'text/html');

    function processNode(node) {
        if (node.nodeType === Node.TEXT_NODE) {
            return node.textContent;
        }

        if (node.nodeType === Node.ELEMENT_NODE) {
            if (node.tagName === 'A' && node.href) {
                const linkText = node.textContent;
                const url = node.href;
                return `${linkText} (${url})`;
            }

            if (node.tagName === 'BR') {
                return '\\n';
            }

            if (['P', 'DIV', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'LI'].includes(node.tagName)) {
                let text = '';
                for (let child of node.childNodes) {
                    text += processNode(child);
                }
                return text + '\\n';
            }

            let text = '';
            for (let child of node.childNodes) {
                text += processNode(child);
            }
            return text;
        }
        return '';
    }

    let result = processNode(doc.body);
    result = result.replace(/\\n{3,}/g, '\\n\\n');
    return result.trim();
}

// Find Streamlit text areas and add paste handler
function attachPasteHandler() {
    const textareas = window.parent.document.querySelectorAll('textarea');

    textareas.forEach(textarea => {
        // Only attach once
        if (textarea.dataset.pasteHandlerAttached) return;
        textarea.dataset.pasteHandlerAttached = 'true';

        textarea.addEventListener('paste', function(e) {
            const clipboardData = e.clipboardData || window.clipboardData;
            const htmlData = clipboardData.getData('text/html');

            if (htmlData && htmlData.includes('<a')) {
                e.preventDefault();

                const processedText = parseHtmlWithLinks(htmlData);

                // Insert at cursor position
                const start = textarea.selectionStart;
                const end = textarea.selectionEnd;
                const currentValue = textarea.value;

                const newValue = currentValue.substring(0, start) + processedText + currentValue.substring(end);

                // Use React's property setter to update value (more reliable than textarea.value = ...)
                const nativeInputValueSetter = Object.getOwnPropertyDescriptor(
                    window.HTMLTextAreaElement.prototype,
                    'value'
                ).set;
                nativeInputValueSetter.call(textarea, newValue);

                // Set cursor position
                const newPos = start + processedText.length;
                textarea.selectionStart = newPos;
                textarea.selectionEnd = newPos;

                // Trigger input event with proper bubbling
                const inputEvent = new Event('input', { bubbles: true, cancelable: true });
                textarea.dispatchEvent(inputEvent);

                // Trigger change event
                const changeEvent = new Event('change', { bubbles: true });
                textarea.dispatchEvent(changeEvent);
            }
        });
    });
}

// Attach on load and periodically (for dynamically created textareas)
attachPasteHandler();
setInterval(attachPasteHandler, 500);
</script>
"""

components.html(paste_helper, height=0)

# Regular text area - JavaScript will enhance it
st.text_area(
    "Paste your text here:",
    height=300,
    placeholder="Paste your text here (links from Google Docs will be preserved automatically)...",
    key="main_text_input"
)

# Fact-check button
submit_button = st.button("Fact Check", type="primary")

# Debug: Show what's in session state
with st.expander("🐛 Debug Info", expanded=False):
    st.write("**Text in session state:**")
    st.code(st.session_state.main_text_input, language=None)
    st.write(f"Length: {len(st.session_state.main_text_input)} characters")

    # Show URL formatting
    import re
    urls_found = re.findall(r'\(https?://[^\)]+\)', st.session_state.main_text_input)
    if urls_found:
        st.write(f"**URLs found in text:** {len(urls_found)}")
        for idx, url in enumerate(urls_found[:5], 1):  # Show first 5
            st.code(url, language=None)
        if len(urls_found) > 5:
            st.write(f"... and {len(urls_found) - 5} more")
    else:
        st.write("**No URLs found in text**")

def strip_urls(text):
    """Remove URLs in parentheses from text for AI processing"""
    import re
    # Remove URLs in parentheses like (https://example.com)
    # This pattern matches (http://... or https://...) including complex URLs
    cleaned = re.sub(r'\s*\(https?://[^\)]+\)', '', text)
    # Clean up any resulting double spaces
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.strip()

def highlight_text(original_text, issues, show_misleading=True, show_incomplete=True, show_questionable=True):
    """Highlight problematic sections in the text - handles URLs gracefully and overlapping highlights"""
    import re
    if not issues:
        return original_text

    # Filter issues based on visibility settings, keeping original indices
    filtered_issues = []
    for idx, issue in enumerate(issues):
        issue_type = issue.get('type', 'questionable')
        if issue_type == 'misleading' and not show_misleading:
            continue
        if issue_type == 'incomplete' and not show_incomplete:
            continue
        if issue_type == 'questionable' and not show_questionable:
            continue
        # Add original index to the issue dict for tracking
        issue_with_idx = issue.copy()
        issue_with_idx['_original_index'] = idx
        filtered_issues.append(issue_with_idx)

    # If all filtered out, return original
    if not filtered_issues:
        return original_text

    # Use filtered issues for highlighting
    issues = filtered_issues

    # Debug info
    match_debug = []

    # Build a mapping from stripped text positions to original text positions
    def build_position_map(text):
        """Returns (stripped_text, map_stripped_to_original)"""
        stripped = []
        position_map = []  # Maps each char in stripped to position in original

        i = 0
        while i < len(text):
            # Check for URL pattern
            url_match = re.match(r'\s*\(https?://[^\)]+\)', text[i:])
            if url_match:
                # Skip URLs in stripped version
                i += len(url_match.group())
                continue

            # Add character to stripped version and record its original position
            stripped.append(text[i])
            position_map.append(i)
            i += 1

        return ''.join(stripped), position_map

    original_stripped, pos_map = build_position_map(original_text)

    # Color code by type
    colors = {
        'misleading': '#ffcccc',  # light red
        'questionable': '#fff4cc',  # light yellow
        'incomplete': '#cce5ff'  # light blue
    }

    border_colors = {
        'misleading': '#ff0000',  # red
        'questionable': '#ffa500',  # orange
        'incomplete': '#0066cc'  # blue
    }

    # First pass: find all highlight positions in the original text
    highlight_ranges = []  # List of (start, end, issue_index, issue_data)

    for i, issue in enumerate(issues):
        excerpt = issue.get('excerpt', '').strip()
        issue_type = issue.get('type', 'questionable')
        explanation = issue.get('issue', 'No explanation')
        sources = issue.get('sources', [])

        if not excerpt:
            # For incomplete type issues, empty excerpt is OK - skip highlighting
            original_idx = issue.get('_original_index', i)
            if issue_type == 'incomplete':
                match_debug.append({
                    'index': original_idx,
                    'excerpt': '(empty - incomplete issue)',
                    'matched': 'N/A',
                    'reason': 'Incomplete type - no specific text to highlight'
                })
            else:
                match_debug.append({
                    'index': original_idx,
                    'excerpt': '(empty)',
                    'matched': False,
                    'reason': 'Empty excerpt'
                })
            continue

        # Find the position in original text
        start_pos = None
        end_pos = None
        match_method = None

        # Strategy 1: Try exact match first
        pos = original_text.find(excerpt)
        if pos != -1:
            start_pos = pos
            end_pos = pos + len(excerpt)
            match_method = "Exact match"
        else:
            # Strategy 2: Match using stripped text
            excerpt_stripped, _ = build_position_map(excerpt)
            pos = original_stripped.find(excerpt_stripped)

            if pos != -1 and len(excerpt_stripped) > 0:
                # Map back to original positions
                start_pos = pos_map[pos]
                end_pos_stripped = min(pos + len(excerpt_stripped), len(pos_map))

                if end_pos_stripped > 0 and end_pos_stripped <= len(pos_map):
                    end_pos = pos_map[end_pos_stripped - 1] + 1

                    # Verify the match
                    candidate = original_text[start_pos:end_pos]
                    candidate_stripped, _ = build_position_map(candidate)

                    if candidate_stripped == excerpt_stripped:
                        match_method = "Position mapping"
                    else:
                        # Try extending to clean boundary
                        while end_pos < len(original_text):
                            char = original_text[end_pos]
                            if char in ' \n\t':
                                break
                            if original_text[end_pos:end_pos+1] == '(':
                                url_match = re.match(r'\(https?://[^\)]+\)', original_text[end_pos:])
                                if url_match:
                                    end_pos += len(url_match.group())
                                break
                            end_pos += 1

                        candidate = original_text[start_pos:end_pos]
                        candidate_stripped, _ = build_position_map(candidate)

                        if candidate_stripped == excerpt_stripped:
                            match_method = "Position mapping (adjusted)"
                        else:
                            start_pos = None
                            end_pos = None

            # Strategy 3: Fuzzy match with normalized whitespace
            if start_pos is None and len(excerpt_stripped) > 0:
                excerpt_normalized = ' '.join(excerpt_stripped.split())
                original_normalized = ' '.join(original_stripped.split())

                pos_norm = original_normalized.find(excerpt_normalized)
                if pos_norm != -1:
                    # Find the position in original text by counting characters
                    remaining = original_text
                    char_count = 0
                    target_chars = pos_norm

                    # Skip characters until we reach the position
                    while char_count < target_chars and remaining:
                        # Skip URLs
                        url_match = re.match(r'\s*\(https?://[^\)]+\)', remaining)
                        if url_match:
                            remaining = remaining[len(url_match.group()):]
                            continue
                        # Skip whitespace sequences as single space
                        ws_match = re.match(r'\s+', remaining)
                        if ws_match:
                            remaining = remaining[len(ws_match.group()):]
                            char_count += 1
                            continue
                        # Regular character
                        remaining = remaining[1:]
                        char_count += 1

                    # Now extract the excerpt length
                    start_pos = len(original_text) - len(remaining)
                    char_count = 0
                    target_chars = len(excerpt_normalized)

                    while char_count < target_chars and remaining:
                        # Include URLs
                        url_match = re.match(r'\s*\(https?://[^\)]+\)', remaining)
                        if url_match:
                            remaining = remaining[len(url_match.group()):]
                            continue
                        # Treat whitespace as single space
                        ws_match = re.match(r'\s+', remaining)
                        if ws_match:
                            remaining = remaining[len(ws_match.group()):]
                            char_count += 1
                            continue
                        # Regular character
                        remaining = remaining[1:]
                        char_count += 1

                    end_pos = len(original_text) - len(remaining)
                    candidate = original_text[start_pos:end_pos]

                    # Verify
                    candidate_stripped, _ = build_position_map(candidate)
                    candidate_normalized = ' '.join(candidate_stripped.split())
                    if candidate_normalized == excerpt_normalized:
                        match_method = "Fuzzy word match"
                    else:
                        start_pos = None
                        end_pos = None

        if start_pos is not None and end_pos is not None:
            # Use original index for issue numbering
            original_idx = issue.get('_original_index', i)
            highlight_ranges.append({
                'start': start_pos,
                'end': end_pos,
                'issue_index': original_idx,
                'issue_type': issue_type,
                'explanation': explanation,
                'sources': sources,
                'excerpt': excerpt
            })
            match_debug.append({
                'index': original_idx,
                'excerpt': excerpt[:100] + '...' if len(excerpt) > 100 else excerpt,
                'matched': True,
                'method': match_method
            })
        else:
            # Failed to find - provide detailed debug info
            original_idx = issue.get('_original_index', i)
            excerpt_stripped_local, _ = build_position_map(excerpt)
            excerpt_normalized_local = ' '.join(excerpt_stripped_local.split())
            original_normalized_local = ' '.join(original_stripped.split())

            debug_info = {
                'index': original_idx,
                'excerpt': excerpt[:100] + '...' if len(excerpt) > 100 else excerpt,
                'matched': False,
                'reason': 'Not found in text',
                'excerpt_full_length': len(excerpt),
                'excerpt_stripped': excerpt_stripped_local[:100] + '...' if len(excerpt_stripped_local) > 100 else excerpt_stripped_local,
            }

            # Try to find partial match for debugging
            if len(excerpt_normalized_local) > 20:
                # Try first 20 chars
                first_20 = excerpt_normalized_local[:20]
                if first_20 in original_normalized_local:
                    debug_info['partial_match'] = f"First 20 chars found in text"
                else:
                    debug_info['partial_match'] = f"First 20 chars NOT found"

            match_debug.append(debug_info)

    # Sort by start position
    highlight_ranges.sort(key=lambda x: x['start'])

    # Second pass: build segments with all applicable highlights
    # Create a map of position -> list of active issues
    position_issues = {}
    for hr in highlight_ranges:
        for pos in range(hr['start'], hr['end']):
            if pos not in position_issues:
                position_issues[pos] = []
            position_issues[pos].append(hr)

    # Build segments where the set of active issues changes
    segments = []
    current_pos = 0

    positions = sorted(set([0, len(original_text)] +
                          [hr['start'] for hr in highlight_ranges] +
                          [hr['end'] for hr in highlight_ranges]))

    for i in range(len(positions) - 1):
        start = positions[i]
        end = positions[i + 1]

        # Get all issues active at this position
        active_issues = position_issues.get(start, [])

        segments.append({
            'start': start,
            'end': end,
            'text': original_text[start:end],
            'issues': active_issues
        })

    # Third pass: build HTML with proper highlighting
    html_parts = []

    for segment in segments:
        if not segment['issues']:
            # No highlight
            html_parts.append(segment['text'])
        else:
            # Has one or more highlights
            issues_data = segment['issues']

            # Primary issue (first one)
            primary = issues_data[0]
            bg_color = colors.get(primary['issue_type'], '#fff4cc')

            # Build tooltip with all issues
            tooltip_parts = []
            all_issue_indices = []

            for idx, issue_data in enumerate(issues_data):
                all_issue_indices.append(str(issue_data['issue_index']))

                tooltip_parts.append(f"Issue #{issue_data['issue_index'] + 1} ({issue_data['issue_type'].title()}):")
                tooltip_parts.append(issue_data['explanation'])

                if issue_data['sources']:
                    tooltip_parts.append("\nSources:")
                    for src_idx, src in enumerate(issue_data['sources'], 1):
                        tooltip_parts.append(f"{src_idx}. {src}")

                if idx < len(issues_data) - 1:
                    tooltip_parts.append("\n---\n")

            tooltip_text = "\n".join(tooltip_parts)
            escaped_tooltip = tooltip_text.replace('"', '&quot;').replace("'", '&#39;')

            # Build border style for multiple issues
            border_style = "2px solid transparent"
            if len(issues_data) > 1:
                # Use border to show additional issues
                border_colors_list = [border_colors.get(iss['issue_type'], '#333') for iss in issues_data[1:]]
                if len(border_colors_list) == 1:
                    border_style = f"2px solid {border_colors_list[0]}"
                else:
                    # Multiple borders - use box-shadow to show multiple colors
                    border_style = f"2px solid {border_colors_list[0]}"

            issue_ids = ','.join(all_issue_indices)
            # For single issue, show the issue number; for multiple, show count
            badge_text = str(primary['issue_index'] + 1) if len(issues_data) == 1 else str(len(issues_data))
            badge_type = "single" if len(issues_data) == 1 else "multiple"

            html_parts.append(f'''<mark
                class="fact-issue fact-issue-type-{primary['issue_type']}"
                data-issue-ids="{issue_ids}"
                data-tooltip="{escaped_tooltip}"
                data-issue-count="{len(issues_data)}"
                data-badge-text="{badge_text}"
                data-badge-type="{badge_type}"
                data-primary-type="{primary['issue_type']}"
                style="background-color: {bg_color}; color: #000; padding: 2px 4px; border-radius: 3px; cursor: pointer; border: {border_style}; transition: all 0.2s ease;"
                >{segment['text']}</mark>''')

    # Store debug info in session state
    import streamlit as st
    st.session_state.highlight_debug = match_debug

    return ''.join(html_parts)

# Process fact-check when button is clicked
if submit_button:
    # Clear previous results when starting new fact-check
    st.session_state.fact_check_results = None

    # Use the main_text_input from session state
    current_text = st.session_state.main_text_input
    if not current_text.strip():
        st.error("Please enter some text to fact-check.")
    elif not os.getenv("OPENAI_API_KEY"):
        st.error("OpenAI API key not found. Please add OPENAI_API_KEY to your .env file.")
    else:
        with st.spinner("Performing deep research and fact-checking..."):
            try:
                # Prepare the prompt
                prompt = f"""

                Run a deep research to fact check this text. Identify any misleading information, questionable statements, or missing important information that would confuse the reader.

                CRITICAL INSTRUCTIONS:
                1. Respond in English
                2. For the "excerpt" field: You MUST copy-paste the EXACT text from the original that has the issue. DO NOT write your own summary or commentary.
                   - CORRECT: "Седемте рилски езера са най-атрактивни за посещение"
                   - WRONG: "В текста има множество линкове" (this is commentary, not from the original)
                   - WRONG: "Текстът съдържа" (this is your observation, not from the original)
                   - WRONG: "Липсва контекст" (this describes what's missing, not what's there)
                3. If information is MISSING (incomplete), you can describe what's missing in the "issue" field, but leave "excerpt" empty or put a relevant sentence that should have more detail.
                4. Only include issues where you can point to specific problematic text OR identify specific gaps.

                Return your analysis as a JSON object with the following structure:
                {{
                "issues": [
                    {{
                    "excerpt": "EXACT TEXT copied from the original (not your summary)",
                    "issue": "explanation of what is wrong, misleading, or missing",
                    "type": "misleading" | "questionable" | "incomplete",
                    "sources": ["URL or source 1", "URL or source 2"]
                    }}
                ],
                "all_sources": ["list", "of", "all", "sources", "used"]
                }}

                For each issue, provide the sources you used to verify the information. Include ALL sources at the end in the all_sources array.

                If no issues are found, return: {{"issues": [], "all_sources": []}}

                Text to fact-check:
                {current_text}
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
                                    "excerpt": {
                                        "type": "string",
                                        "description": "Exact verbatim text copied from the original that has an issue (not a summary or commentary)"
                                    },
                                    "issue": {
                                        "type": "string",
                                        "description": "Explanation of what is wrong, misleading, or missing"
                                    },
                                    "type": {
                                        "type": "string",
                                        "enum": ["misleading", "questionable", "incomplete"],
                                        "description": "Type of issue"
                                    },
                                    "sources": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "URLs or sources used to verify"
                                    }
                                },
                                "required": ["excerpt", "issue", "type", "sources"],
                                "additionalProperties": False
                            }
                        },
                        "all_sources": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "All sources consulted during fact-checking"
                        }
                    },
                    "required": ["issues", "all_sources"],
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
                        api_params["tools"] = [
                            {
                                "type": "web_search",
                                "user_location": {
                                    "type": "approximate"
                                },
                                "search_context_size": search_context_size
                            }
                        ]
                        api_params["store"] = True
                        api_params["include"] = [
                            "reasoning.encrypted_content",
                            "web_search_call.action.sources"
                        ]

                    # Make API call using responses endpoint
                    response = client.responses.create(**api_params)

                    if enable_streaming:
                        # Process streaming response (event-based)
                        result_text = ""
                        reasoning_text = ""
                        reasoning_and_search_items = []  # Store all items in order

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

                            # Done event - handled in completed event below
                            elif event_type == 'response.output_item.done':
                                pass  # All extraction happens in response.completed

                            # Completed event contains the final response
                            elif event_type == 'response.completed':
                                if hasattr(event, 'response'):
                                    resp = event.response
                                    if hasattr(resp, 'output') and resp.output:
                                        for output_item in resp.output:
                                            item_type = getattr(output_item, 'type', None)

                                            # Extract reasoning summary
                                            if item_type == 'reasoning':
                                                reasoning_text = ""
                                                if hasattr(output_item, 'summary') and output_item.summary:
                                                    for summary_item in output_item.summary:
                                                        if hasattr(summary_item, 'text'):
                                                            reasoning_text += summary_item.text
                                                if reasoning_text:
                                                    reasoning_and_search_items.append({
                                                        'type': 'reasoning',
                                                        'text': reasoning_text
                                                    })

                                            # Extract web search calls with query and sources
                                            elif item_type == 'web_search_call':
                                                action = None
                                                if hasattr(output_item, 'action'):
                                                    action = output_item.action

                                                if action:
                                                    # Handle both dict and object
                                                    if isinstance(action, dict):
                                                        query = action.get('query', '')
                                                        sources = action.get('sources', [])
                                                    else:
                                                        query = getattr(action, 'query', '')
                                                        sources = getattr(action, 'sources', [])

                                                    reasoning_and_search_items.append({
                                                        'type': 'web_search',
                                                        'query': query,
                                                        'sources': sources
                                                    })

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
                        reasoning_and_search_items = []  # Store all items in order

                        # Extract both reasoning and message from response
                        if hasattr(response, 'output') and response.output:
                            for output_item in response.output:
                                item_type = getattr(output_item, 'type', None)

                                # Extract reasoning summary
                                if item_type == 'reasoning':
                                    reasoning_text = ""
                                    if hasattr(output_item, 'summary') and output_item.summary:
                                        for summary_item in output_item.summary:
                                            if hasattr(summary_item, 'text'):
                                                reasoning_text += summary_item.text
                                    if reasoning_text:
                                        reasoning_and_search_items.append({
                                            'type': 'reasoning',
                                            'text': reasoning_text
                                        })

                                # Extract web search calls with query and sources
                                elif item_type == 'web_search_call':
                                    action = None
                                    if hasattr(output_item, 'action'):
                                        action = output_item.action

                                    if action:
                                        # Handle both dict and object
                                        if isinstance(action, dict):
                                            query = action.get('query', '')
                                            sources = action.get('sources', [])
                                        else:
                                            query = getattr(action, 'query', '')
                                            sources = getattr(action, 'sources', [])

                                        reasoning_and_search_items.append({
                                            'type': 'web_search',
                                            'query': query,
                                            'sources': sources
                                        })

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
                        all_sources = []
                    else:
                        result_json = json.loads(result_text)
                        # Handle different possible JSON structures
                        if isinstance(result_json, dict):
                            issues = result_json.get('issues', result_json.get('findings', []))
                            all_sources = result_json.get('all_sources', [])
                            if not isinstance(issues, list):
                                issues = [result_json]
                        else:
                            issues = result_json
                            all_sources = []
                except json.JSONDecodeError as e:
                    st.error(f"Failed to parse fact-check results. Error: {str(e)}")
                    st.text(result_text)
                    issues = []
                    all_sources = []

                st.success("Fact-check complete!")

                # Store results in session state for persistence across reruns
                st.session_state.fact_check_results = {
                    'issues': issues,
                    'all_sources': all_sources,
                    'current_text': current_text,
                    'model_choice': model_choice,
                    'reasoning_and_search_items': reasoning_and_search_items if 'reasoning_and_search_items' in locals() else []
                }

            except Exception as e:
                st.error(f"Error: {str(e)}")
                import traceback
                with st.expander("Show error details"):
                    st.code(traceback.format_exc())

# Display results from session state (persists across reruns)
if st.session_state.fact_check_results is not None:
    results = st.session_state.fact_check_results
    issues = results['issues']
    all_sources = results['all_sources']
    current_text = results['current_text']
    model_choice = results['model_choice']
    reasoning_and_search_items = results['reasoning_and_search_items']

    # Display reasoning summary and web searches if available
    if model_choice.startswith("gpt-5") and reasoning_and_search_items:
        with st.expander("🧠 Reasoning Summary & Web Search", expanded=False):
            search_counter = 0
            for item in reasoning_and_search_items:
                if item['type'] == 'reasoning':
                    # Display reasoning text
                    st.markdown(item['text'])
                    st.markdown("")  # Add spacing

                elif item['type'] == 'web_search':
                    # Display web search
                    search_counter += 1
                    query = item.get('query', '')
                    sources = item.get('sources', [])

                    st.markdown("---")
                    if query:
                        st.markdown(f"**🔍 Web Search {search_counter}:** {query}")
                    else:
                        st.markdown(f"**🔍 Web Search {search_counter}**")

                    if sources:
                        for idx, source in enumerate(sources, 1):
                            url = None
                            # Handle dict
                            if isinstance(source, dict):
                                url = source.get('url', '')
                            # Handle object with url attribute
                            elif hasattr(source, 'url'):
                                url = source.url
                            # Handle plain string
                            else:
                                url = str(source)

                            if url:
                                st.markdown(f"{idx}. [{url}]({url})")
                            else:
                                st.markdown(f"{idx}. (no URL)")

                    st.markdown("")  # Add spacing

    if issues:
        # Filter controls (define before using them)
        st.markdown("### Highlighted Text")
        col1, col2, col3 = st.columns(3)
        with col1:
            show_misleading = st.checkbox("🔴 Show Misleading", value=True, key="filter_misleading")
        with col2:
            show_incomplete = st.checkbox("🔵 Show Incomplete", value=True, key="filter_incomplete")
        with col3:
            show_questionable = st.checkbox("🟡 Show Questionable", value=True, key="filter_questionable")

        # Generate highlighted text with filters applied
        highlighted_text = highlight_text(current_text, issues, show_misleading, show_incomplete, show_questionable)

        # Show matching debug info
        if 'highlight_debug' in st.session_state:
            matched_count = sum(1 for d in st.session_state.highlight_debug if d['matched'] == True)
            overlapping_count = sum(1 for d in st.session_state.highlight_debug if d['matched'] == 'overlapping')
            failed_count = sum(1 for d in st.session_state.highlight_debug if d['matched'] == False)
            total_count = len(st.session_state.highlight_debug)

            if overlapping_count > 0 or failed_count > 0:
                message_parts = []
                if overlapping_count > 0:
                    message_parts.append(f"{overlapping_count} issue(s) overlap with other highlights")
                if failed_count > 0:
                    message_parts.append(f"{failed_count} issue(s) could not be matched")

                st.info(f"ℹ️ {matched_count} issues highlighted. " + ", ".join(message_parts) + ". All issues are listed below.")

                with st.expander("🔍 Debug: Highlight matching details", expanded=False):
                    for debug_item in st.session_state.highlight_debug:
                        if debug_item['matched'] == True:
                            st.success(f"✅ Issue #{debug_item['index']}: **Highlighted** using {debug_item.get('method', 'unknown')}")
                            st.code(f"Excerpt: {debug_item['excerpt']}", language=None)
                        elif debug_item['matched'] == 'overlapping':
                            st.warning(f"⚠️ Issue #{debug_item['index']}: **{debug_item.get('reason', 'Overlapping')}**")
                            st.code(f"Text: {debug_item.get('text_found', debug_item['excerpt'])}", language=None)
                            st.caption("💡 This text is already highlighted by another issue. The issue is still listed below.")
                        elif debug_item['matched'] == 'N/A':
                            st.info(f"ℹ️ Issue #{debug_item['index']}: **{debug_item.get('reason', 'N/A')}**")
                        else:
                            st.error(f"❌ Issue #{debug_item['index']}: **Failed** - {debug_item.get('reason', 'unknown')}")
                            st.code(f"Excerpt AI returned ({debug_item.get('excerpt_full_length', 'unknown')} chars): {debug_item['excerpt']}", language=None)
                            if 'excerpt_stripped' in debug_item:
                                st.code(f"Excerpt stripped: {debug_item['excerpt_stripped']}", language=None)
                            if 'partial_match' in debug_item:
                                st.caption(f"🔍 {debug_item['partial_match']}")
                            if 'text_extracted' in debug_item and debug_item['text_extracted']:
                                st.code(f"Text extracted (bad boundaries): {debug_item['text_extracted']}", language=None)
                                st.caption(f"Method: {debug_item.get('method_used', 'unknown')}")
                            st.caption("💡 The AI should return EXACT text from your original, not commentary or summaries.")

        # Use components.html with dynamic height
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
            overflow: visible;
        }}
        .fact-issue {{
            position: relative;
            transition: all 0.2s ease;
        }}
        .fact-issue:hover {{
            border-width: 3px !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        }}
        .fact-issue::after {{
            content: attr(data-badge-text);
            position: absolute;
            top: -8px;
            right: -8px;
            color: white;
            border-radius: 50%;
            width: 18px;
            height: 18px;
            font-size: 11px;
            font-weight: bold;
            display: flex;
            align-items: center;
            justify-content: center;
            border: 2px solid white;
            z-index: 100;
        }}
        .fact-issue[data-badge-type="single"]::after {{
            background: #666;
        }}
        .fact-issue[data-badge-type="multiple"]::after {{
            background: #ff6600;
        }}
        .custom-tooltip {{
            position: absolute;
            background: #333;
            color: white;
            padding: 12px 28px 12px 12px;
            border-radius: 6px;
            z-index: 10000;
            max-width: 500px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            font-size: 14px;
            line-height: 1.5;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
        .custom-tooltip a {{
            color: #66b3ff;
            text-decoration: underline;
        }}
        .tooltip-close {{
            position: absolute;
            top: 6px;
            right: 8px;
            background: transparent;
            border: none;
            color: white;
            font-size: 20px;
            cursor: pointer;
            padding: 0;
            width: 20px;
            height: 20px;
            line-height: 20px;
            text-align: center;
        }}
        .tooltip-close:hover {{
            color: #ff6666;
        }}
        </style>
        </head>
        <body>
            {highlighted_text}
            <script>
            let currentTooltip = null;

            // Add click handlers to all highlights
            document.querySelectorAll('.fact-issue').forEach(function(mark) {{
                mark.addEventListener('click', function(e) {{
                    e.stopPropagation();

                    // Remove existing tooltip if any
                    if (currentTooltip) {{
                        currentTooltip.remove();
                        currentTooltip = null;
                    }}

                    // Get tooltip content
                    const tooltipText = this.getAttribute('data-tooltip');
                    if (!tooltipText) return;

                    // Create tooltip element
                    const tooltip = document.createElement('div');
                    tooltip.className = 'custom-tooltip';
                    tooltip.innerHTML = tooltipText.replace(/\\n/g, '<br>');

                    // Add close button
                    const closeBtn = document.createElement('button');
                    closeBtn.className = 'tooltip-close';
                    closeBtn.innerHTML = '&times;';
                    closeBtn.onclick = function(e) {{
                        e.stopPropagation();
                        tooltip.remove();
                        currentTooltip = null;
                    }};
                    tooltip.appendChild(closeBtn);

                    // Position tooltip
                    document.body.appendChild(tooltip);
                    const rect = this.getBoundingClientRect();
                    tooltip.style.left = rect.left + 'px';
                    tooltip.style.top = (rect.bottom + 5) + 'px';

                    // Keep track of current tooltip
                    currentTooltip = tooltip;
                }});
            }});

            // Close tooltip when clicking outside
            document.addEventListener('click', function(e) {{
                if (currentTooltip && !currentTooltip.contains(e.target)) {{
                    currentTooltip.remove();
                    currentTooltip = null;
                }}
            }});

            // Auto-resize iframe to fit content
            function resizeIframe() {{
                const height = document.body.scrollHeight + 20;
                window.parent.postMessage({{
                    type: 'streamlit:setFrameHeight',
                    height: height
                }}, '*');
            }}

            // Resize on load and whenever content changes
            window.addEventListener('load', resizeIframe);
            setTimeout(resizeIframe, 100);
            setTimeout(resizeIframe, 500);
            </script>
        </body>
        </html>
        """
        # Height will be set automatically by JavaScript
        components.html(html_content, scrolling=False)

        # Legend
        st.markdown("""
        <div style="margin-top: 20px; margin-bottom: 30px; padding: 10px; background-color: #f0f0f0; border-radius: 5px; color: #000;">
            <b>Legend:</b><br>
            <mark style="background-color: #ffcccc; padding: 2px 4px; color: #000;">Misleading</mark>
            <mark style="background-color: #fff4cc; padding: 2px 4px; color: #000;">Questionable</mark>
            <mark style="background-color: #cce5ff; padding: 2px 4px; color: #000;">Incomplete</mark>
            <br><br>
            <b>Issue badges:</b> Each highlight shows a number badge:
            <ul style="margin: 5px 0; padding-left: 20px;">
                <li><span style="background: #666; color: white; border-radius: 50%; padding: 2px 6px; font-size: 11px; font-weight: bold;">3</span> = Issue #3 (single issue)</li>
                <li><span style="background: #ff6600; color: white; border-radius: 50%; padding: 2px 6px; font-size: 11px; font-weight: bold;">2</span> = 2 overlapping issues (click to see all)</li>
            </ul>
            <b>Status icons in issue cards:</b> 📍 Highlighted in text above | 🔗 Overlaps with another highlight
            <br><br>
            <small>💡 Click on highlights to see full explanations and sources. Click the × or outside the tooltip to close. See details below ⬇</small>
        </div>
        """, unsafe_allow_html=True)

        # Issues section (full width below) - reordered by type
        st.markdown("### Issues Found")

        # Reorder issues by type: misleading, incomplete, questionable
        type_order = {'misleading': 0, 'incomplete': 1, 'questionable': 2}
        sorted_issues = sorted(enumerate(issues), key=lambda x: type_order.get(x[1].get('type', 'questionable'), 3))

        for original_index, issue in sorted_issues:
            issue_type = issue.get('type', 'questionable').title()
            issue_type_lower = issue.get('type', 'questionable')

            # Apply filter
            if issue_type_lower == 'misleading' and not show_misleading:
                continue
            if issue_type_lower == 'incomplete' and not show_incomplete:
                continue
            if issue_type_lower == 'questionable' and not show_questionable:
                continue

            excerpt = issue.get('excerpt', 'N/A')
            explanation = issue.get('issue', 'No explanation provided')
            issue_sources = issue.get('sources', [])

            # Check if this issue was highlighted or overlapping
            highlight_status = ""
            if 'highlight_debug' in st.session_state and original_index < len(st.session_state.highlight_debug):
                debug_item = st.session_state.highlight_debug[original_index]
                if debug_item.get('matched') == True:
                    highlight_status = " 📍"  # Highlighted
                elif debug_item.get('matched') == 'overlapping':
                    highlight_status = " 🔗"  # Overlapping with another highlight

            # Color for issue type badge
            type_colors = {
                'Misleading': '🔴',
                'Questionable': '🟡',
                'Incomplete': '🔵'
            }
            icon = type_colors.get(issue_type, '⚪')

            # Build sources HTML
            sources_html = ""
            if issue_sources:
                sources_html = "<br><br><b>Sources:</b><br>"
                for idx, src in enumerate(issue_sources, 1):
                    # Make URLs clickable
                    if src.startswith('http'):
                        sources_html += f'{idx}. <a href="{src}" target="_blank" style="color: #0066cc;">{src}</a><br>'
                    else:
                        sources_html += f'{idx}. {src}<br>'

            st.markdown(f"""
            <div id="issue-{original_index}" style="padding: 10px; margin-bottom: 15px; border-left: 3px solid #ccc; background-color: #f9f9f9; color: #000; transition: box-shadow 0.3s;">
                <b>{icon} Issue #{original_index+1}: {issue_type}{highlight_status}</b><br>
                <i style="color: #666;">"{excerpt[:100]}{'...' if len(excerpt) > 100 else ''}"</i><br><br>
                <div style="color: #000;">{explanation}</div>
                {sources_html}
            </div>
            """, unsafe_allow_html=True)

        # Display all sources section
        if all_sources:
            st.markdown("---")
            st.markdown("### 📚 All Sources Used")
            for idx, source in enumerate(all_sources, 1):
                # Make URLs clickable
                if source.startswith('http'):
                    st.markdown(f"{idx}. [{source}]({source})")
                else:
                    st.markdown(f"{idx}. {source}")
    else:
        st.success("✅ No issues found! The text appears to be accurate and complete.")

        # Still show sources if available even when no issues found
        if all_sources:
            st.markdown("---")
            st.markdown("### 📚 Sources Consulted")
            for idx, source in enumerate(all_sources, 1):
                # Make URLs clickable
                if source.startswith('http'):
                    st.markdown(f"{idx}. [{source}]({source})")
                else:
                    st.markdown(f"{idx}. {source}")
