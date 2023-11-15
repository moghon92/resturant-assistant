from agent import run_agent
import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text=initial_text
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


def main():
    # Page title

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    st.set_page_config(page_title="Chat with Menu", page_icon=":books:")
    st.title('Chat with your Menu :book:')
    openai_api_key = st.text_input('OpenAI API Key', placeholder='sk-', type='password')
    with st.form('myform', clear_on_submit=False):
        user_msg = st.text_input('Question:', placeholder="What's for breakfast?", disabled=not openai_api_key)
        submitted = st.form_submit_button('Submit', disabled=not openai_api_key)
        # st.markdown(st.session_state.chat_history)
        if submitted and user_msg != "":
            with st.spinner('Calculating...'):
                chat_box = st.empty()
                stream_handler = StreamHandler(chat_box)
                response = run_agent(user_msg, st.session_state.chat_history, stream_handler, openai_api_key)
                # st.markdown(response['output'])
                if len(response['updated_chat_history']) > 6:
                    st.session_state.chat_history = response['updated_chat_history'][-6:]

if __name__ == '__main__':
    main()