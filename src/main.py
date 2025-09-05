import streamlit as st
import pandas as pd
import io

try:
    from generate import generate_multiple_batches

    generate_available = True
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Make sure generate.py is in the same directory")
    generate_available = False

try:
    from refine import model

    refine_available = True
except ImportError:
    refine_available = False

try:
    from system_prompts.dataset_config import dataset_config_prompt

    config_available = True
except ImportError:
    config_available = False

try:
    from logger import logger

    logger_available = True
except ImportError:
    logger_available = False


def main():
    st.set_page_config(page_title="Dataset Generator", page_icon="📊", layout="wide")

    st.title("📊 Dataset Generator")
    st.markdown("Generate synthetic datasets using AI models")

    if not generate_available:
        st.error(
            "Cannot proceed without the generation module. Please check your imports."
        )
        return

    with st.sidebar:
        st.header("Configuration")
        st.info("Make sure your environment variables are set:")
        st.code(
            """
        openrouter_api_key=your_key_here
        deep_seek_api=your_key_here
        """
        )

    # Main interface
    st.subheader("Enter Your Prompt")
    user_prompt = st.text_area(
        "Dataset Generation Prompt",
        placeholder="Enter your prompt here... (e.g., 'Generate a dataset about customer demographics for an e-commerce store')",
        height=100,
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Generate Dataset")
        if st.button("🚀 Generate Dataset", type="primary", use_container_width=True):
            if not user_prompt.strip():
                st.error("Please enter a prompt first!")
            else:
                with st.spinner("Generating dataset..."):
                    result = generate_multiple_batches(user_prompt)
                    if result:
                        st.session_state.generated_csv = result
                        st.session_state.show_results = True

    with col2:
        st.subheader("Actions")
        if (
            hasattr(st.session_state, "generated_csv")
            and st.session_state.generated_csv
        ):
            # Download button
            st.download_button(
                label="💾 Download CSV",
                data=st.session_state.generated_csv,
                file_name="generated_dataset.csv",
                mime="text/csv",
                use_container_width=True,
            )

            # Clear results button
            if st.button("🗑️ Clear Results", use_container_width=True):
                if "generated_csv" in st.session_state:
                    del st.session_state.generated_csv
                if "show_results" in st.session_state:
                    del st.session_state.show_results
                st.rerun()

    # Display results
    if hasattr(st.session_state, "show_results") and st.session_state.show_results:
        st.subheader("Generated Dataset")

        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["📊 Table View", "📝 Raw CSV", "📈 Statistics"])

        with tab1:
            try:
                # Convert CSV string to DataFrame
                csv_data = io.StringIO(st.session_state.generated_csv)
                df = pd.read_csv(csv_data)

                # Display the dataframe
                st.dataframe(df, use_container_width=True, height=400)

                # Show basic info
                st.subheader("Dataset Info")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Rows", len(df))
                with col2:
                    st.metric("Total Columns", len(df.columns))
                with col3:
                    st.metric(
                        "Data Size", f"{len(st.session_state.generated_csv)} chars"
                    )

            except Exception as e:
                st.error(f"Error displaying table: {e}")
                st.text("Raw CSV data:")
                st.text(st.session_state.generated_csv)

        with tab2:
            st.text_area(
                "Raw CSV Data",
                value=st.session_state.generated_csv,
                height=400,
                disabled=True,
            )

        with tab3:
            try:
                csv_data = io.StringIO(st.session_state.generated_csv)
                df = pd.read_csv(csv_data)

                st.subheader("Column Information")
                for col in df.columns:
                    with st.expander(f"Column: {col}"):
                        if df[col].dtype in ["int64", "float64"]:
                            col_info = df[col].describe()
                        else:
                            col_info = df[col].value_counts().head()
                        st.write(col_info)

            except Exception as e:
                st.error(f"Error generating statistics: {e}")


if __name__ == "__main__":
    main()
