import streamlit as st
import pandas as pd
import pandasql as ps
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
        border: 1px solid #e0e0e0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

def detect_env_api_key():
    """Detect and validate Gemini API key from environment variables"""
    api_key = None
    source = None
    
    # Try to get API key from environment
    if DOTENV_AVAILABLE:
        api_key = os.getenv('GEMINI_API_KEY')
        if api_key:
            source = ".env file"
    
    # If not found in .env, try system environment
    if not api_key:
        api_key = os.environ.get('GEMINI_API_KEY')
        if api_key:
            source = "system environment"
    
    # Validate the API key by testing it
    if api_key:
        try:
            client = genai.Client(api_key=api_key)
            # Test with a simple API call
            return api_key, source, True
        except Exception as e:
            return api_key, source, False
    
    return None, None, False

def configure_gemini(api_key):
    """Configure Gemini API and return client"""
    try:
        client = genai.Client(api_key=api_key)
        return client
    except Exception as e:
        st.error(f"Error configuring Gemini API: {str(e)}")
        return None

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'dataframes' not in st.session_state:
    st.session_state.dataframes = {}
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""
if 'auto_visualization' not in st.session_state:
    st.session_state.auto_visualization = True
if 'visualizations' not in st.session_state:
    st.session_state.visualizations = {}
if 'edited_queries' not in st.session_state:
    st.session_state.edited_queries = {}
if 'query_results' not in st.session_state:
    st.session_state.query_results = {}
if 'db_connections' not in st.session_state:
    st.session_state.db_connections = {}
if 'table_schemas' not in st.session_state:
    st.session_state.table_schemas = {}
if 'full_table_info' not in st.session_state:
    st.session_state.full_table_info = {}
if 'data_source_type' not in st.session_state:
    st.session_state.data_source_type = "files"  # "files" or "database"
if 'env_api_key' not in st.session_state:
    st.session_state.env_api_key = None
if 'env_api_source' not in st.session_state:
    st.session_state.env_api_source = None
if 'env_api_valid' not in st.session_state:
    st.session_state.env_api_valid = False

# Detect environment API key on startup
if not st.session_state.env_api_key:
    env_key, env_source, env_valid = detect_env_api_key()
    if env_key:
        st.session_state.env_api_key = env_key
        st.session_state.env_api_source = env_source
        st.session_state.env_api_valid = env_valid
        if env_valid:
            st.session_state.api_key = env_key

def create_db_connection(db_type, host, port, database, username, password):
    """Create database connection"""
    try:
        if db_type == "PostgreSQL":
            # Create PostgreSQL connection
            connection_string = f"postgresql://{username}:{urllib.parse.quote_plus(password)}@{host}:{port}/{database}"
            engine = create_engine(connection_string)
            
            # Test connection
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            return engine
            
        elif db_type == "MySQL":
            # Create MySQL connection
            connection_string = f"mysql+mysqlconnector://{username}:{urllib.parse.quote_plus(password)}@{host}:{port}/{database}"
            engine = create_engine(connection_string)
            
            # Test connection
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            return engine
            
    except Exception as e:
        st.error(f"Error connecting to {db_type}: {str(e)}")
        return None

def get_table_list(engine, db_type):
    """Get list of tables from database"""
    try:
        if db_type == "PostgreSQL":
            query = """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND (table_type = 'BASE TABLE' OR table_type = 'VIEW')
            ORDER BY table_name;
            """
        elif db_type == "MySQL":
            query = """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = DATABASE()
            ORDER BY table_name
            """
        
        with engine.connect() as conn:
            result = conn.execute(text(query))
            tables = [row[0] for row in result]
        
        return tables
    except Exception as e:
        st.error(f"Error getting table list: {str(e)}")
        return []

def get_table_sample_and_schema(engine, table_name, sample_size=10):
    """Get table sample (first 10 rows) and schema info"""
    try:
        # Get sample data
        sample_query = f"SELECT * FROM {table_name} LIMIT {sample_size}"
        sample_df = pd.read_sql(sample_query, engine)
        
        # Get row count
        count_query = f"SELECT COUNT(*) as total_rows FROM {table_name}"
        with engine.connect() as conn:
            result = conn.execute(text(count_query))
            total_rows = result.fetchone()[0]
        
        # Get column info
        info_query = f"SELECT * FROM {table_name} WHERE 1=0"  # Get structure only
        info_df = pd.read_sql(info_query, engine)
        
        return sample_df, total_rows, info_df.dtypes.to_dict()
        
    except Exception as e:
        st.error(f"Error getting table data for {table_name}: {str(e)}")
        return None, 0, {}

def read_uploaded_file(uploaded_file):
    """Read uploaded file based on its extension"""
    try:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        
        if file_extension == '.csv':
            # Read CSV file
            df = pd.read_csv(uploaded_file)
            
        elif file_extension in ['.xlsx', '.xls']:
            # Read Excel file with enhanced error handling
            try:
                # First, get all sheet names to check what's available
                excel_file = pd.ExcelFile(uploaded_file)
                sheet_names = excel_file.sheet_names
                
                # Try to read the first sheet
                df = None
                for sheet_name in sheet_names:
                    try:
                        # Read with different header options
                        temp_df = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=0)
                        
                        # Check if this sheet has meaningful data
                        if not temp_df.empty and len(temp_df.columns) > 0:
                            # Remove completely empty rows and columns
                            temp_df = temp_df.dropna(how='all').dropna(axis=1, how='all')
                            
                            # Check if we still have data after cleaning
                            if not temp_df.empty and len(temp_df.columns) > 0:
                                df = temp_df
                                break
                    except Exception:
                        continue
                
                # If first attempt failed, try reading without header and detect it manually
                if df is None or df.empty:
                    for sheet_name in sheet_names:
                        try:
                            # Read without assuming header position
                            temp_df = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=None)
                            
                            if not temp_df.empty:
                                # Try to find the header row (look for row with most non-null values)
                                header_row = 0
                                max_non_null = 0
                                
                                for i in range(min(5, len(temp_df))):  # Check first 5 rows
                                    non_null_count = temp_df.iloc[i].count()
                                    if non_null_count > max_non_null:
                                        max_non_null = non_null_count
                                        header_row = i
                                
                                # Re-read with detected header
                                df = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=header_row)
                                df = df.dropna(how='all').dropna(axis=1, how='all')
                                
                                if not df.empty and len(df.columns) > 0:
                                    break
                        except Exception:
                            continue
                
                if df is None or df.empty:
                    raise ValueError("Could not read any meaningful data from Excel file")
                
            except Exception as e:
                raise ValueError(f"Error reading Excel file: {str(e)}")
            
        elif file_extension == '.txt':
            # Read TXT file - try different delimiters
            try:
                # First try tab-separated
                df = pd.read_csv(uploaded_file, sep='\t')
                # Check if it looks reasonable (more than 1 column)
                if len(df.columns) == 1:
                    # Try comma-separated
                    uploaded_file.seek(0)  # Reset file pointer
                    df = pd.read_csv(uploaded_file, sep=',')
                    if len(df.columns) == 1:
                        # Try semicolon-separated
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, sep=';')
                        if len(df.columns) == 1:
                            # Try pipe-separated
                            uploaded_file.seek(0)
                            df = pd.read_csv(uploaded_file, sep='|')
            except:
                # Fallback to comma-separated
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        # Enhanced data cleaning for all file types
        if df.empty:
            raise ValueError("File is empty after processing")
        
        # Handle column names properly
        # First, ensure all column names are strings
        df.columns = df.columns.astype(str)
        
        # Clean column names (remove special characters, spaces)
        df.columns = df.columns.str.strip()
        
        # Replace problematic characters with underscores
        df.columns = df.columns.str.replace(r'[^\w\s]', '_', regex=True)
        df.columns = df.columns.str.replace(r'\s+', '_', regex=True)  # Replace multiple spaces with single underscore
        df.columns = df.columns.str.replace(r'_+', '_', regex=True)   # Replace multiple underscores with single
        df.columns = df.columns.str.strip('_')  # Remove leading/trailing underscores
        
        # Handle duplicate column names
        if df.columns.duplicated().any():
            cols = []
            for i, col in enumerate(df.columns):
                if col in cols:
                    counter = 1
                    new_col = f"{col}_{counter}"
                    while new_col in cols:
                        counter += 1
                        new_col = f"{col}_{counter}"
                    cols.append(new_col)
                else:
                    cols.append(col)
            df.columns = cols
        
        # Ensure column names are valid SQL identifiers
        final_cols = []
        for col in df.columns:
            # Ensure column starts with letter or underscore
            if col and not (col[0].isalpha() or col[0] == '_'):
                col = f"col_{col}"
            
            # Ensure column is not empty
            if not col or col.strip() == '':
                col = f"column_{len(final_cols)}"
            
            final_cols.append(col)
        
        df.columns = final_cols
        
        # Convert problematic data types for SQL compatibility
        for col in df.columns:
            # Handle mixed types in columns
            if df[col].dtype == 'object':
                # Try to convert to numeric if possible
                try:
                    # Check if the column contains mostly numbers
                    numeric_series = pd.to_numeric(df[col], errors='coerce')
                    if numeric_series.notna().sum() > len(df) * 0.8:  # If 80% can be converted to numbers
                        df[col] = numeric_series
                except:
                    pass
        
        # Final validation
        if len(df.columns) == 0:
            raise ValueError("No valid columns found in file")
        
        return df, file_extension
        
    except Exception as e:
        raise Exception(f"Error reading file: {str(e)}")

def generate_sql_query(user_query, client):
    """Generate SQL query from natural language using Gemini"""
    try:
        # Create context about available tables from both files and databases
        schema_context = "Available tables and their schemas:\n"
        
        # Add uploaded files
        for table_name, df in st.session_state.dataframes.items():
            # Skip database tables (they have connection prefix)
            if not any(table_name.startswith(f"{conn_name}_") for conn_name in st.session_state.db_connections.keys()):
                schema_context += f"\nFile Table: {table_name}\n"
                schema_context += f"Columns: {', '.join(df.columns.tolist())}\n"
                schema_context += f"Sample data (first 2 rows):\n{df.head(2).to_string()}\n"
        
        # Add database tables
        for conn_name, (engine, db_type) in st.session_state.db_connections.items():
            if conn_name in st.session_state.table_schemas:
                schema_context += f"\nDatabase: {conn_name} ({db_type})\n"
                for table_name, table_info in st.session_state.table_schemas[conn_name].items():
                    sample_df = st.session_state.dataframes.get(f"{conn_name}_{table_name}")
                    if sample_df is not None:
                        total_rows = st.session_state.full_table_info[conn_name][table_name]['total_rows']
                        schema_context += f"\nDatabase Table: {table_name} (Total rows: {total_rows})\n"
                        schema_context += f"Columns: {', '.join(sample_df.columns.tolist())}\n"
                        schema_context += f"Sample data (first 2 rows):\n{sample_df.head(2).to_string()}\n"
        
        # Determine if we should use pandasql or direct database query
        has_files = any(not table_name.startswith(f"{conn_name}_") 
                       for table_name in st.session_state.dataframes.keys() 
                       for conn_name in st.session_state.db_connections.keys())
        has_database = bool(st.session_state.db_connections)
        
        sql_type = "pandasql" if has_files or not has_database else "standard SQL"
        
        prompt = f"""
        You are a SQL expert. Based on the user's natural language query and the available table schemas, 
        generate a SQL query using {sql_type} syntax.
        
        {schema_context}
        
        User Query: {user_query}
        
        Rules:
        1. Use table names exactly as provided in the schema
        2. Generate only the SQL query, no explanations
        3. Use {'pandasql' if sql_type == 'pandasql' else 'standard SQL'} syntax
        4. If the query involves aggregation, use appropriate GROUP BY clauses
        5. For filtering, use WHERE clauses
        6. For database tables, the query will be executed on the full dataset (not just the sample shown)
        7. For file tables, the query uses the complete uploaded data
        8. Return only the SQL query without any markdown formatting or extra text
        
        SQL Query:
        """
        
        response = client.models.generate_content(
            model="gemini-3.0-preview",
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)],
                ),
            ],
        )
        sql_query = response.text.strip()
        
        # Clean up the response to get only SQL
        sql_query = re.sub(r'```sql|```|SQL Query:|Query:', '', sql_query).strip()
        
        return sql_query
    except Exception as e:
        st.error(f"Error generating SQL query: {str(e)}")
        return None

def execute_sql_query(sql_query):
    """Execute SQL query using either pandasql for files or database connection"""
    try:
        # Check if query involves database tables
        query_lower = sql_query.lower()
        
        # Find if any database tables are referenced
        db_tables_used = []
        for conn_name, (engine, db_type) in st.session_state.db_connections.items():
            if conn_name in st.session_state.table_schemas:
                for table_name in st.session_state.table_schemas[conn_name].keys():
                    if table_name.lower() in query_lower:
                        db_tables_used.append((conn_name, table_name, engine))
        
        # If database tables are involved, use database execution
        if db_tables_used:
            # Use the first database connection found
            engine = db_tables_used[0][2]
            result = pd.read_sql(sql_query, engine)
            return result
        else:
            # Use pandasql for file-based queries
            # Create a local namespace with file dataframes only (exclude database tables)
            local_ns = {}
            for table_name, df in st.session_state.dataframes.items():
                # Only include tables that are not from database connections
                if not any(table_name.startswith(f"{conn_name}_") for conn_name in st.session_state.db_connections.keys()):
                    local_ns[table_name] = df
            
            if not local_ns:
                st.error("No data tables available for query execution")
                return None
            
            result = ps.sqldf(sql_query, local_ns)
            return result
        
    except Exception as e:
        st.error(f"Error executing SQL query: {str(e)}")
        return None

def create_visualization(df, query, client):
    """Create visualizations using Gemini to generate matplotlib/seaborn code with support for wordcloud"""
    try:
        if df is None or df.empty or len(df) == 0:
            return None
        
        # Analyze data structure
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Enhanced analysis for special visualization types
        text_cols = []
        
        # Detect text columns suitable for wordcloud
        for col in categorical_cols:
            if df[col].dtype == 'object':
                # Check if column contains longer text (potential for wordcloud)
                avg_length = df[col].astype(str).str.len().mean()
                if avg_length > 20:  # Arbitrary threshold for text content
                    text_cols.append(col)
        
        # Check available libraries
        available_libs = []
        if WORDCLOUD_AVAILABLE:
            available_libs.append("WordCloud")
        
        # Create enhanced data summary for Gemini
        data_info = f"""
        DataFrame shape: {df.shape}
        Columns info:
        - Numeric columns: {numeric_cols}
        - Categorical columns: {categorical_cols} 
        - DateTime columns: {datetime_cols}
        - Text columns (for wordcloud): {text_cols}
        
        Available visualization libraries: {', '.join(available_libs)}
        
        Sample data:
        {df.head(3).to_string()}
        
        Data types:
        {df.dtypes.to_string()}
        """
        
        # Enhanced prompt with wordcloud support
        wordcloud_instructions = ""
        if WORDCLOUD_AVAILABLE and text_cols:
            wordcloud_instructions = f"""
        
        WordCloud Instructions (if appropriate):
        - For text analysis queries, use WordCloud from wordcloud library
        - Combine text from relevant columns using: ' '.join(df[column].dropna().astype(str))
        - Use: WordCloud(width=800, height=400, background_color='white').generate(text)
        - Display with: plt.imshow(wordcloud, interpolation='bilinear')
        - Remove axes: plt.axis('off')
        Text columns available: {text_cols}
        """

        
        prompt = f"""
        You are a data visualization expert. Based on the user's query and the resulting data, 
        generate Python code to create the most appropriate visualization.
        
        User Query: "{query}"
        
        Data Information:
        {data_info}
        {wordcloud_instructions}
        
        Instructions:
        1. Analyze the query intent and data structure to determine the best visualization type
        2. Standard charts: bar chart, line chart, scatter plot, histogram, box plot, heatmap, pie chart
        3. Special visualizations:
           - WordCloud: For text analysis, word frequency, content analysis
        4. Choose the most appropriate visualization based on:
           - Query keywords (e.g., "word", "text", "frequency" → WordCloud)
           - Data structure and column names
        5. Generate clean Python code using the variable 'df' for the dataframe
        6. For matplotlib/seaborn: Set figure size with plt.figure(figsize=(10, 6))
        7. Add meaningful title, labels, and styling
        8. Use seaborn style: sns.set_style("whitegrid") for standard plots
        9. Return ONLY the Python code, no explanations or markdown
        10. Ensure the code handles potential data issues (missing values, etc.)
        11. For matplotlib plots: Use plt.tight_layout() and plt.show() at the end
        
        Standard matplotlib example:
        ```
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        sns.set_style("whitegrid")
        plt.figure(figsize=(10, 6))
        # Your visualization code here
        plt.title("Your Title")
        plt.tight_layout()
        plt.show()
        ```
        
        WordCloud example:
        ```
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
        
        text = ' '.join(df['text_column'].dropna().astype(str))
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud')
        plt.tight_layout()
        plt.show()
        ```
        
        Generate the visualization code:
        """
        
        response = client.models.generate_content(
            model="gemini-2.5-flash-preview-05-20",
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)],
                ),
            ],
        )
        viz_code = response.text.strip()
        
        # Clean up the response
        viz_code = re.sub(r'```python|```', '', viz_code).strip()
        
        # Handle matplotlib/seaborn visualizations
        plt.clf()  # Clear any previous plots
        
        # Create a safe execution environment
        exec_globals = {
            'df': df,
            'plt': plt,
            'sns': sns,
            'pd': pd
        }
        
        # Add wordcloud if available
        if WORDCLOUD_AVAILABLE:
            exec_globals['WordCloud'] = WordCloud

        exec(viz_code, exec_globals)
        # Return the current figure
        return plt.gcf()
        
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        st.error(f"Generated code: {viz_code if 'viz_code' in locals() else 'No code generated'}")
        return None

# Sidebar
with st.sidebar:
    st.title("🔧 Configuration")
    
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
