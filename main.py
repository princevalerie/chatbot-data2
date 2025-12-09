import streamlit as st
import pandas as pd
import pandasql as ps
import google.generativeai as genai
from google import genai
from google.genai import types
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import re
import traceback
import warnings
import os
warnings.filterwarnings('ignore')

# Environment variable support
try:
    from dotenv import load_dotenv
    load_dotenv()
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

# Database connectivity imports
import psycopg2
import mysql.connector
from sqlalchemy import create_engine, text
import urllib.parse

# Additional libraries for enhanced visualizations
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Data Chatbot - Files & Databases",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for chatbot UI
st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #f0f2f6;
        margin-left: 20%;
    }
    .bot-message {
        background-color: #e8f4f8;
        margin-right: 20%;
    }
    .sql-code {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 0.5rem 0;
        font-family: 'Courier New', monospace;
    }
    .stDataFrame {
    # API Key configuration section
    st.subheader("🔑 Gemini API Configuration")
    
    # Show environment API key status if detected
    if st.session_state.env_api_key:
        if st.session_state.env_api_valid:
            st.success(f"✅ API Key loaded from {st.session_state.env_api_source}")
            st.info("🔒 API Key input is hidden (using environment variable)")
            
            # Option to override environment API key
            with st.expander("🔄 Override Environment API Key", expanded=False):
                manual_override = st.checkbox("Use manual API key instead", key="manual_override")
                
                if manual_override:
                    api_key = st.text_input(
                        "Manual Gemini API Key",
                        type="password",
                        value="",
                        help="Enter your Google Gemini API key to override environment variable"
                    )
                    
                    if api_key and api_key != st.session_state.api_key:
                        st.session_state.api_key = api_key
                        st.info("🔄 Using manual API key")
                else:
                    # Use environment API key
                    if st.session_state.api_key != st.session_state.env_api_key:
                        st.session_state.api_key = st.session_state.env_api_key
        else:
            st.error(f"❌ Invalid API Key found in {st.session_state.env_api_source}")
            st.warning("⚠️ Please enter a valid API key manually")
            
            # Show manual input for invalid environment key
            api_key = st.text_input(
                "Gemini API Key",
                type="password",
                value="",
                help="Enter your Google Gemini API key (environment key is invalid)"
            )
            
            if api_key != st.session_state.api_key:
                st.session_state.api_key = api_key
    else:
        # No environment API key found, show manual input
        st.info("📝 No .env file detected. Please enter API key manually.")
        
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            value=st.session_state.api_key,
            help="Enter your Google Gemini API key"
        )
        
        if api_key != st.session_state.api_key:
            st.session_state.api_key = api_key
    
    # Show .env setup instructions if needed
    if not st.session_state.env_api_valid:
        with st.expander("💡 .env File Setup", expanded=False):
            st.markdown("""
            **To use .env file for API key:**
            
            1. Create a `.env` file in your project root
            2. Add this line to the file:
               ```
               GEMINI_API_KEY=your_actual_api_key_here
               ```
            3. Restart the application
            4. API key input will be automatically hidden
            
            **Requirements:**
            - Install python-dotenv: `pip install python-dotenv`
            - Keep .env file in the same directory as app.py
            """)
    
    st.divider()
    
    # Auto visualization toggle
    st.subheader("🎨 Visualization Settings")
    auto_viz = st.toggle(
        "Auto Visualization",
        value=st.session_state.auto_visualization,
        help="If enabled, visualizations will be generated automatically. If disabled, you can manually request visualizations."
    )
    
    if auto_viz != st.session_state.auto_visualization:
        st.session_state.auto_visualization = auto_viz
    
    st.divider()
    
    # Data source tabs
    tab1, tab2 = st.tabs(["📁 Files", "🗄️ Databases"])
    
    with tab1:
        # File upload
        st.subheader("📁 Upload Data Files")
        uploaded_files = st.file_uploader(
            "Drag and drop files here",
            type=['csv', 'xlsx', 'xls', 'txt'],
            accept_multiple_files=True,
            help="Upload CSV, Excel (xlsx/xls), or TXT files to query"
        )
        
        # Process uploaded files
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in st.session_state.dataframes:
                    try:
                        # Read file using the appropriate method
                        df, file_ext = read_uploaded_file(uploaded_file)
                        
                        # Create table name (remove extension and clean name)
                        table_name = os.path.splitext(uploaded_file.name)[0]
                        table_name = table_name.replace(' ', '_').replace('-', '_')
                        # Ensure table name is valid SQL identifier
                        table_name = re.sub(r'[^\w]', '_', table_name)
                        if table_name and not (table_name[0].isalpha() or table_name[0] == '_'):
                            table_name = f"table_{table_name}"
                        
                        # Store dataframe
                        st.session_state.dataframes[table_name] = df
                        
                        # Show table preview only
                        st.write(f"**{uploaded_file.name}** ({len(df)} rows, {len(df.columns)} columns)")
                        st.dataframe(df.head(3), use_container_width=True)
                            
                    except Exception as e:
                        st.error(f"❌ Error loading {uploaded_file.name}: {str(e)}")
                        # Show specific troubleshooting tips
                        if uploaded_file.name.endswith('.txt'):
                            st.info("💡 TXT file tips: Make sure data is separated by commas, tabs, semicolons, or pipes")
                        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                            st.info("💡 Excel troubleshooting:")
                            st.info("   • Make sure the file is not corrupted")
                            st.info("   • Ensure data starts from row 1 or has clear headers")
                            st.info("   • Try saving as CSV if problems persist")
                            st.info("   • Check if the file has multiple sheets with data")
                        elif uploaded_file.name.endswith('.csv'):
                            st.info("💡 CSV tips: Check file encoding and delimiter format")
    
    with tab2:
        # Database connection
        st.subheader("🗄️ Database Connection")
        
        # Database type selection
        db_type = st.selectbox(
            "Database Type",
            ["PostgreSQL", "MySQL"],
            help="Select your database type"
        )
        
        # Connection form
        with st.expander("➕ Add New Connection", expanded=len(st.session_state.db_connections) == 0):
            with st.form("db_connection_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    conn_name = st.text_input(
                        "Connection Name",
                        placeholder="my_database",
                        help="Give this connection a name"
                    )
                    host = st.text_input(
                        "Host",
                        value="localhost",
                        help="Database server host"
                    )
                    database = st.text_input(
                        "Database Name",
                        placeholder="database_name",
                        help="Name of the database"
                    )
                
                with col2:
                    port = st.number_input(
                        "Port",
                        value=5432 if db_type == "PostgreSQL" else 3306,
                        help="Database port number"
                    )
                    username = st.text_input(
                        "Username",
                        placeholder="username",
                        help="Database username"
                    )
                    password = st.text_input(
                        "Password",
                        type="password",
                        help="Database password"
                    )
                
                submitted = st.form_submit_button("🔗 Connect to Database")
                
                if submitted and conn_name and host and database and username and password:
                    with st.spinner(f"Connecting to {db_type} database..."):
                        engine = create_db_connection(db_type, host, port, database, username, password)
                        
                        if engine:
                            # Store connection
                            st.session_state.db_connections[conn_name] = (engine, db_type)
                            st.session_state.table_schemas[conn_name] = {}
                            st.session_state.full_table_info[conn_name] = {}
                            
                            # Get available tables
                            tables = get_table_list(engine, db_type)
                            
                            if tables:
                                st.success(f"✅ Connected to {conn_name}! Found {len(tables)} tables.")
                                
                                # Load sample data for each table
                                for table_name in tables:
                                    try:
                                        sample_df, total_rows, column_types = get_table_sample_and_schema(engine, table_name)
                                        
                                        if sample_df is not None:
                                            # Store sample data with connection prefix
                                            table_key = f"{conn_name}_{table_name}"
                                            st.session_state.dataframes[table_key] = sample_df
                                            st.session_state.table_schemas[conn_name][table_name] = column_types
                                            st.session_state.full_table_info[conn_name][table_name] = {
                                                'total_rows': total_rows,
                                                'sample_rows': len(sample_df)
                                            }
                                    except Exception as e:
                                        st.warning(f"Could not load table {table_name}: {str(e)}")
                                
                                st.rerun()
                            else:
                                st.warning("No tables found in the database")
                        else:
                            st.error(f"❌ Failed to connect to {conn_name}")
        
        # Show connected databases and tables
        if st.session_state.db_connections:
            st.subheader("📊 Connected Databases")
            
            for conn_name, (engine, db_type) in st.session_state.db_connections.items():
                with st.expander(f"🗄️ {conn_name} ({db_type})", expanded=False):
                    if conn_name in st.session_state.table_schemas:
                        for table_name, column_types in st.session_state.table_schemas[conn_name].items():
                            table_key = f"{conn_name}_{table_name}"
                            if table_key in st.session_state.dataframes:
                                sample_df = st.session_state.dataframes[table_key]
                                total_rows = st.session_state.full_table_info[conn_name][table_name]['total_rows']
                                
                                st.write(f"**{table_name}** (Sample: {len(sample_df)} rows, Total: {total_rows:,} rows)")
                                st.dataframe(sample_df, use_container_width=True)
                    
                    # Disconnect button
                    if st.button(f"🔌 Disconnect {conn_name}", key=f"disconnect_{conn_name}"):
                        # Remove connection and related data
                        del st.session_state.db_connections[conn_name]
                        if conn_name in st.session_state.table_schemas:
                            # Remove dataframes for this connection
                            tables_to_remove = []
                            for key in st.session_state.dataframes.keys():
                                if key.startswith(f"{conn_name}_"):
                                    tables_to_remove.append(key)
                            for key in tables_to_remove:
                                del st.session_state.dataframes[key]
                            
                            del st.session_state.table_schemas[conn_name]
                            del st.session_state.full_table_info[conn_name]
                        
                        st.rerun()
    
    st.divider()
    
    # Clear data button
    if st.button("🗑️ Clear All Data"):
        st.session_state.dataframes = {}
        st.session_state.chat_history = []
        st.session_state.visualizations = {}
        st.session_state.edited_queries = {}
        st.session_state.query_results = {}
        st.session_state.db_connections = {}
        st.session_state.table_schemas = {}
        st.session_state.full_table_info = {}
        st.rerun()

# Main content
st.title("🤖 Data Chatbot - Files & Databases")
st.markdown("Ask questions about your uploaded files and database tables in natural language!")

# Check if API key and data are available
if not st.session_state.api_key:
    st.warning("⚠️ Please enter your Gemini API key in the sidebar to start chatting.")
elif not st.session_state.dataframes:
    st.info("📊 Please upload data files or connect to a database in the sidebar to begin analysis.")
else:
    # Configure Gemini
    model = configure_gemini(st.session_state.api_key)
    
    if model:
        # Chat interface
        st.subheader("💬 Chat with your Data")
        
        # Display chat history
        for i, (user_msg, bot_response, sql_query, result_df) in enumerate(st.session_state.chat_history):
            # User message
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>You:</strong> {user_msg}
            </div>
            """, unsafe_allow_html=True)
            
            # Bot response
            st.markdown(f"""
            <div class="chat-message bot-message">
                <strong>🤖 Bot:</strong> {bot_response}
            </div>
            """, unsafe_allow_html=True)
            
            # SQL query
            if sql_query:
                st.markdown("**Generated SQL Query:**")
                
                if st.session_state.auto_visualization:
                    # Auto mode - show SQL as code block only
                    st.code(sql_query, language='sql')
                else:
                    # Manual mode - allow editing SQL
                    query_id = f"query_{i}"
                    
                    # Check if there's an edited version
                    current_sql = st.session_state.edited_queries.get(query_id, sql_query)
                    
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        edited_sql = st.text_area(
                            f"Edit SQL Query (Query {i+1})",
                            value=current_sql,
                            height=100,
                            key=f"sql_edit_{i}"
                        )
                    
                    with col2:
                        st.write("")  # Spacing
                        st.write("")  # Spacing
                        if st.button("🔄 Re-execute", key=f"reexec_{i}"):
                            if edited_sql.strip():
                                # Store edited query
                                st.session_state.edited_queries[query_id] = edited_sql
                                
                                # Execute edited query
                                with st.spinner("⚡ Re-executing query..."):
                                    new_result = execute_sql_query(edited_sql)
                                    if new_result is not None:
                                        # Update results
                                        st.session_state.query_results[query_id] = new_result
                                        st.success("✅ Query re-executed successfully!")
                                        st.rerun()
                                    else:
                                        st.error("❌ Error executing edited query")
                            else:
                                st.warning("⚠️ Please enter a valid SQL query")
                    
                    # Show original query if different from edited
                    if current_sql != sql_query:
                        with st.expander("📝 Original Generated Query"):
                            st.code(sql_query, language='sql')
            
            # Results
            query_id = f"query_{i}"
            
            # Use updated result if available, otherwise use original
            display_df = st.session_state.query_results.get(query_id, result_df)
            
            if display_df is not None and not display_df.empty:
                st.markdown("**Query Results:**")
                
                # Show row count and execution info
                if query_id in st.session_state.query_results:
                    st.info(f"📊 Showing {len(display_df)} rows (from re-executed query)")
                else:
                    st.info(f"📊 Showing {len(display_df)} rows")
                
                st.dataframe(display_df, use_container_width=True)
                
                # Visualization logic based on auto_visualization setting                
                if st.session_state.auto_visualization:
                    # Auto visualization - show immediately
                    if query_id not in st.session_state.visualizations:
                        with st.spinner("🎨 Creating visualization..."):
                            fig = create_visualization(display_df, user_msg, model)
                            if fig:
                                st.session_state.visualizations[query_id] = fig
                            else:
                                st.session_state.visualizations[query_id] = None
                    
                    # Display stored visualization
                    if st.session_state.visualizations.get(query_id):
                        st.markdown("**Visualization:**")
                        st.pyplot(st.session_state.visualizations[query_id])
                else:
                    # Manual visualization - show button
                    col1, col2 = st.columns([3, 1])
                    with col2:
                        if query_id not in st.session_state.visualizations:
                            if st.button(f"📊 Visualize", key=f"viz_btn_{i}"):
                                with st.spinner("🎨 Creating visualization..."):
                                    # Use current query for context
                                    current_query = st.session_state.edited_queries.get(query_id, user_msg)
                                    fig = create_visualization(display_df, current_query, model)
                                    if fig:
                                        st.session_state.visualizations[query_id] = fig
                                        st.rerun()
                                    else:
                                        st.session_state.visualizations[query_id] = None
                                        st.warning("Could not generate visualization for this query.")
                        else:
                            if st.button(f"🗑️ Clear Chart", key=f"clear_btn_{i}"):
                                del st.session_state.visualizations[query_id]
                                st.rerun()
                        
                        # Re-visualize button if query was re-executed
                        if query_id in st.session_state.query_results and query_id in st.session_state.visualizations:
                            if st.button(f"🔄 Re-visualize", key=f"reviz_btn_{i}"):
                                with st.spinner("🎨 Re-creating visualization..."):
                                    current_query = st.session_state.edited_queries.get(query_id, user_msg)
                                    fig = create_visualization(display_df, current_query, model)
                                    if fig:
                                        st.session_state.visualizations[query_id] = fig
                                        st.rerun()
                                    else:
                                        st.warning("Could not generate visualization for this query.")
                    
                    # Display stored visualization if exists
                    if st.session_state.visualizations.get(query_id):
                        st.markdown("**Visualization:**")
                        if query_id in st.session_state.query_results:
                            st.caption("📊 Visualization based on re-executed query")
                        st.pyplot(st.session_state.visualizations[query_id])
            
            st.divider()
        
        # Chat input
        user_query = st.chat_input("Ask a question about your data...")
        
        if user_query:
            # Generate SQL query
            with st.spinner("🧠 Generating SQL query..."):
                sql_query = generate_sql_query(user_query, model)
            
            if sql_query:
                # Execute SQL query
                with st.spinner("⚡ Executing query..."):
                    result_df = execute_sql_query(sql_query)
                
                if result_df is not None:
                    bot_response = f"I found {len(result_df)} rows matching your query."
                    
                    # Add to chat history
                    st.session_state.chat_history.append((
                        user_query, 
                        bot_response, 
                        sql_query, 
                        result_df
                    ))
                    
                    # If auto visualization is enabled, create visualization for new query
                    if st.session_state.auto_visualization:
                        query_id = f"query_{len(st.session_state.chat_history) - 1}"
                        with st.spinner("🎨 Creating visualization..."):
                            fig = create_visualization(result_df, user_query, model)
                            if fig:
                                st.session_state.visualizations[query_id] = fig
                            else:
                                st.session_state.visualizations[query_id] = None
                    
                    # Rerun to display new message
                    st.rerun()
                else:
                    st.error("❌ Failed to execute the SQL query. Please try rephrasing your question.")
            else:
                st.error("❌ Failed to generate SQL query. Please try rephrasing your question.")

# Footer
st.markdown("---")

# Only show tips if no chat history exists (chatting hasn't started yet)
if not st.session_state.chat_history:
    st.markdown("**💡 Tips:**")
    st.markdown("""
    - Ask questions like: "Show me the average sales by region"
    - Try: "Filter customers with age > 30" 
    - Or: "Count orders by product category"
    - Use natural language - the AI will convert it to SQL!

    **🎨 Enhanced Visualizations:**
    - **WordCloud**: Ask "show word frequency" or "create wordcloud" for text analysis
    - **Standard Charts**: Bar, line, scatter, histogram, heatmap, pie charts
    - **Auto-detection**: AI automatically selects the best visualization type

    **🔧 Manual Mode Features:**
    - Edit generated SQL queries for fine-tuning
    - Re-execute queries with your modifications
    - Use table names exactly as shown in sidebar
    - SQL supports: SELECT, WHERE, GROUP BY, ORDER BY, LIMIT, etc.

    **📂 Supported File Formats:**
    - **CSV**: Comma-separated values with headers
    - **Excel**: .xlsx and .xls files with intelligent sheet and header detection
    - **TXT**: Tab, comma, semicolon, or pipe-separated files
    - **Auto-detection**: Smart format detection for all file types
    
    **🗄️ Supported Databases:**
    - **PostgreSQL**: Full support for PostgreSQL databases
    - **MySQL**: Full support for MySQL databases
    - **Auto-detection**: Intelligent table and schema detection
    - **Sample Preview**: Shows first 10 rows, queries use full dataset
    
    **🔧 Combined Features:**
    - **Hybrid Queries**: Work with both uploaded files and database tables
    - **Multiple Connections**: Connect to multiple databases simultaneously
    - **Full SQL Support**: Use standard PostgreSQL/MySQL SQL syntax for databases
    - **PandasSQL**: Use pandasql syntax for file-based queries
    - **Security**: Connection credentials are session-based

    **📝 Query Tips:**
    - Use exact table names as shown in sidebar
    - For wordclouds, query text columns (descriptions, comments, etc.)
    - Try aggregate queries: "average sales by category"
    - Use filtering: "customers where age > 30"
    - Combine data sources: query both files and database tables
    
    **🔑 API Key Management:**
    - **Environment Variables**: Create `.env` file with `GEMINI_API_KEY=your_key`
    - **Auto-detection**: API key input is automatically hidden when .env is detected
    - **Override Option**: Manually override environment key if needed
    - **Security**: Environment variables are more secure than manual input
    - **Setup**: Install `python-dotenv` and restart the application
    """) 
