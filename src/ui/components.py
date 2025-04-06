import streamlit as st
import time
import markdown
from typing import List, Dict, Any, Optional, Callable
from ..utils.caching import SessionState

class UIComponents:
    """
    Static UI components for the Streamlit app.
    All methods are static to minimize object creation.
    """
    
    @staticmethod
    def render_header() -> None:
        """
        Render the app header with title and description.
        """
        with SessionState.get("containers", {}).get("title", st.container()):
            st.title("Multi-Agent Research Chat")
            st.markdown(
                "This application uses multiple specialized AI agents to research and answer your questions."
            )
    
    @staticmethod
    def render_file_upload() -> List[Any]:
        """
        Render the file upload component with state preservation.
        
        Returns:
            List of uploaded files
        """
        # Preserve file uploader state across reruns
        SessionState.preserve_file_uploader_state()
        
        # Use the preserved key for file uploader
        file_uploader_key = SessionState.get("file_uploader_key", f"uploader_{int(time.time())}")
        
        st.subheader("Upload Documents")
        st.markdown("Upload markdown files to add to the knowledge base.")
        
        uploaded_files = st.file_uploader(
            "Choose markdown files",
            type=["md", "txt"],
            accept_multiple_files=True,
            key=file_uploader_key
        )
        
        return uploaded_files
    
    @staticmethod
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def format_markdown(text: str) -> str:
        """
        Format markdown text to HTML for better display.
        
        Args:
            text: Markdown text
            
        Returns:
            HTML formatted text
        """
        return markdown.markdown(text)
    
    @staticmethod
    def render_chat_history() -> None:
        """
        Render the chat history from session state.
        """
        chat_history = SessionState.get("chat_history", [])
        
        for i, (role, content) in enumerate(chat_history):
            with st.chat_message(role):
                if role == "assistant":
                    # Format the assistant's markdown responses
                    st.markdown(UIComponents.format_markdown(content))
                else:
                    st.markdown(content)
    
    @staticmethod
    def render_chat_input(on_submit: Callable[[str], None]) -> None:
        """
        Render the chat input and handle submission.
        
        Args:
            on_submit: Callback function for message submission
        """
        # Create a container for the chat input to avoid form reloading
        chat_input_container = st.container()
        
        with chat_input_container:
            user_input = st.chat_input("Ask a question...")
            
            if user_input:
                on_submit(user_input)
    
    @staticmethod
    def render_processing_status(is_processing: bool = False) -> None:
        """
        Render the processing status indicator.
        
        Args:
            is_processing: Whether the system is currently processing
        """
        status_container = SessionState.get("containers", {}).get("status", st.empty())
        
        if is_processing:
            with status_container:
                st.info("Processing your request... This may take a few moments.")
        else:
            with status_container:
                status_container.empty()
    
    @staticmethod
    def render_file_processing_progress(total: int, current: int, description: str = "Processing files") -> None:
        """
        Render a progress bar for file processing.
        
        Args:
            total: Total number of files/steps
            current: Current progress
            description: Description of the progress
        """
        progress_container = st.empty()
        
        if total > 0:
            progress = current / total
            progress_container.progress(progress, text=f"{description}: {current}/{total}")
            
            if current >= total:
                time.sleep(0.5)  # Brief pause to show completion
                progress_container.empty()
    
    @staticmethod
    def render_sidebar_controls(
        on_clear_history: Callable[[], None],
        on_regenerate: Optional[Callable[[], None]] = None
    ) -> None:
        """
        Render the sidebar controls for the app.
        
        Args:
            on_clear_history: Callback for clearing chat history
            on_regenerate: Optional callback for regenerating the last response
        """
        with st.sidebar:
            st.header("Controls")
            
            # Clear chat history button
            if st.button("Clear Chat History", use_container_width=True):
                on_clear_history()
            
            # Regenerate last response button
            if on_regenerate and st.button("Regenerate Last Response", use_container_width=True):
                on_regenerate()
            
            st.divider()
            
            # About section
            st.header("About")
            st.markdown(
                """
                This application uses a multi-agent approach:
                
                1. **Analyst Agent**: Researches and synthesizes information
                2. **Reviewer Agent**: Checks and refines the analyst's work
                
                This design ensures accurate and comprehensive answers.
                """
            )
    
    @staticmethod
    def render_error(error_message: str) -> None:
        """
        Render an error message.
        
        Args:
            error_message: The error message to display
        """
        with SessionState.get("containers", {}).get("error_recovery", st.container()):
            st.error(f"Error: {error_message}")
            
            # Add retry button for recoverable errors
            if st.button("Retry"):
                st.rerun()


class Callbacks:
    """
    Static callback handlers for UI interactions.
    """
    
    @staticmethod
    def handle_send_message(
        message: str,
        process_message_func: Callable[[str, List], Any]
    ) -> None:
        """
        Handle sending a message in the chat.
        
        Args:
            message: The user message
            process_message_func: Function to process the message
        """
        # Add user message to chat history
        chat_history = SessionState.get("chat_history", [])
        chat_history.append(("user", message))
        SessionState.set("chat_history", chat_history)
        
        # Display the updated chat history
        UIComponents.render_chat_history()
        
        # Show processing status
        UIComponents.render_processing_status(True)
        
        try:
            # Process the message with the provided function
            result = process_message_func(message, chat_history)
            
            # Add assistant response to chat history
            if result:
                chat_history.append(("assistant", result.raw))
                SessionState.set("chat_history", chat_history)
        except Exception as e:
            # Handle errors
            UIComponents.render_error(str(e))
        finally:
            # Clear processing status
            UIComponents.render_processing_status(False)
    
    @staticmethod
    def handle_clear_history() -> None:
        """
        Handle clearing the chat history.
        """
        SessionState.clear_chat_history()
        st.rerun()
    
    @staticmethod
    def handle_regenerate_last_response(
        process_message_func: Callable[[str, List], Any]
    ) -> None:
        """
        Handle regenerating the last response.
        
        Args:
            process_message_func: Function to process the message
        """
        chat_history = SessionState.get("chat_history", [])
        
        # Check if there's a user message to regenerate a response for
        user_messages = [(i, msg) for i, (role, msg) in enumerate(chat_history) if role == "user"]
        
        if user_messages:
            # Get the last user message
            last_user_idx, last_user_msg = user_messages[-1]
            
            # Remove the last assistant response if it exists
            if last_user_idx < len(chat_history) - 1 and chat_history[last_user_idx + 1][0] == "assistant":
                chat_history.pop(last_user_idx + 1)
            
            # Update the session state
            SessionState.set("chat_history", chat_history)
            
            # Process the last user message again
            Callbacks.handle_send_message(last_user_msg, process_message_func) 