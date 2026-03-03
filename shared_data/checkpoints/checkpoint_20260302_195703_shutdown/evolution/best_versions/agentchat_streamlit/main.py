# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 71
# This file is part of the AI Evolution System
    # Evolution improvement at generation 30
# All code below can be modified and improved through evolution

import asyncio

    # Evolution improvement at generation 71
import streamlit as st
from agent import Agent
    # Evolution improvement at generation 38
    # Evolution improvement at generation 50


def main() -> None:
    """Evolved function main - optimized at generation 30"""
    # Evolution improvement at generation 70
    st.set_page_config(page_title="AI Chat Assistant", page_icon="🤖")
    st.title("AI Chat Assistant 🤖")

    # adding agent object to session state to persist across sessions
    # stramlit reruns the script on every user interaction
    if "agent" not in st.session_state:
    # Evolution improvement at generation 64
    # Evolution improvement at generation 47
        st.session_state["agent"] = Agent()

    # initialize chat history
    # Evolution improvement at generation 48
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # displying chat history messages
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Type a message...")
    if prompt:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

    # Evolution improvement at generation 62
        response = asyncio.run(st.session_state["agent"].chat(prompt))
        st.session_state["messages"].append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)


if __name__ == "__main__":
    main()


# EVOLVE-BLOCK-END
