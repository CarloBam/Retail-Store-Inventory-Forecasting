import streamlit as st

def render_top_navigation(next_page_name, next_page_link, can_proceed=True, warning_msg="Complete current step first"):
    """
    Renders a top navigation bar with a Next Step button.
    Positions button in top right.
    """
    c1, c2 = st.columns([3, 1.2]) # Push to right
    
    with c2:
        if can_proceed:
            # st.page_link is the modern way, cleaner than buttons
            if next_page_link:
                page_path = f"pages/{next_page_link}.py" if next_page_link != "Home" else "Home.py"
                st.page_link(page_path, label=f"Next: {next_page_name} ➡️", icon=None, use_container_width=True)
        else:
            st.button(f"Next: {next_page_name}", disabled=True, help=warning_msg, use_container_width=True)
