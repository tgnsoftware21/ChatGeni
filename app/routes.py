from flask import Blueprint, request, jsonify, current_app
import pandas as pd
import logging
from functools import lru_cache
from pathlib import Path
import filelock
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from .openai_service import generate_sql_query
from .db import query_database, get_db_connection
import os, json


route_app = Blueprint("route_app", __name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add schema caching
@lru_cache(maxsize=32)
def get_cached_schema(metadata_path):
    lock_file = Path(metadata_path).with_suffix('.lock')
    with filelock.FileLock(lock_file):
        return pd.read_excel(metadata_path)

def filter_schema_intelligently(schema_df, user_input, relevant_columns):
    """
    Enhanced schema filtering that handles multiple tables and column-based filtering
    """
    try:
        filtered_schema_df = pd.DataFrame()
        user_input_lower = user_input.lower()
        
        # Extract keywords from user input
        keywords = [word for word in user_input_lower.split() if len(word) > 2]
        logger.debug(f"Keywords: {keywords}")
        # Define table mappings with more comprehensive keyword matching
        table_mappings = {
            'workorder': ['order', 'work', 'workorder', 'job', 'task', 'maintenance', 'repair', 'ticket'],
            'property': ['property', 'properties', 'client', 'building', 'site', 'location', 'address', 'premise','loan'],
            'schedule': ['schedule', 'scheduled', 'auto triggering','auto cancellation','coordiator assignment'],
            'asset': ['asset', 'equipment', 'machine', 'device', 'tool', 'inventory'],
            'user': ['user', 'person', 'employee', 'worker', 'staff', 'technician', 'contact'],
            'invoice': ['invoice', 'bill', 'payment', 'cost', 'price', 'amount', 'charge', 'expense'],
            'contract': ['contract', 'agreement', 'service', 'vendor', 'supplier', 'provider']
        }
        
        # Step 1: Find relevant tables based on keywords
        relevant_tables = set()
        
        for table_type, search_keywords in table_mappings.items():
            if any(keyword in user_input_lower for keyword in search_keywords):
                # Find tables that match this type (case insensitive)
                logger.debug(f"Search Key: {table_type} , {table_type}")
                table_pattern = f'(?i).*{table_type}.*|.*VW.*{table_type}.*'
                matching_tables = schema_df[
                    schema_df['Table_Name'].str.contains(table_type, case=False, na=False, regex=False)
                ]['Table_Name'].unique()
                relevant_tables.update(matching_tables)
        
        # Step 2: Add column-based filtering
        relevant_columns_by_name = set()
        
        # Search for columns that match user input keywords
        if relevant_tables:
            table_filtered_df = schema_df[schema_df['Table_Name'].isin(relevant_tables)]
            for keyword in keywords:
                logger.debug(f"Search Key for column in filtered tables: {keyword}")
                matching_columns = table_filtered_df[
                    table_filtered_df['columnname'].str.contains(keyword, case=False, na=False, regex=False) |
                    (table_filtered_df['Description'].notna() & 
                        table_filtered_df['Description'].str.contains(keyword, case=False, na=False, regex=False))
                ]
                logger.debug(f"Matching columns in filtered tables: {len(matching_columns)}")
                if not matching_columns.empty:
                    relevant_columns_by_name.update(matching_columns['Table_Name'].unique())
    
        
        # Step 3: Combine table-based and column-based results
        all_relevant_tables = relevant_tables.union(relevant_columns_by_name)
        
        # Step 4: Filter schema based on identified tables
        if all_relevant_tables:
            table_filter = schema_df['Table_Name'].isin(all_relevant_tables)
            filtered_schema_df = schema_df[table_filter][relevant_columns].copy()
        
        # Step 5: If still no matches, try broader search
        if filtered_schema_df.empty and keywords:
            # Create OR condition for all keywords
            keyword_pattern = '|'.join(keywords)
            text_search_filter = (
                schema_df['Table_Name'].str.contains(keyword_pattern, case=False, na=False, regex=True) |
                schema_df['columnname'].str.contains(keyword_pattern, case=False, na=False, regex=True) |
                (schema_df['Description'].notna() & 
                 schema_df['Description'].str.contains(keyword_pattern, case=False, na=False, regex=True))
            )
            filtered_schema_df = schema_df[text_search_filter][relevant_columns].copy()
        
        # Step 6: Always include primary keys from relevant tables
        if not filtered_schema_df.empty:
            relevant_table_names = filtered_schema_df['Table_Name'].unique()
            pk_filter = (
                schema_df['Table_Name'].isin(relevant_table_names) &
                schema_df['CONSTRAINT_TYPE'].notna() &
                schema_df['CONSTRAINT_TYPE'].str.contains('PRIMARY KEY', case=False, na=False)
            )
            pk_schema = schema_df[pk_filter][relevant_columns]
            
            # Combine and remove duplicates
            if not pk_schema.empty:
                filtered_schema_df = pd.concat([filtered_schema_df, pk_schema]).drop_duplicates()
        
        # Step 7: Add foreign key relationships if multiple tables are involved
        if not filtered_schema_df.empty and len(filtered_schema_df['Table_Name'].unique()) > 1:
            relevant_table_names = filtered_schema_df['Table_Name'].unique()
            fk_filter = (
                schema_df['Table_Name'].isin(relevant_table_names) &
                schema_df['CONSTRAINT_TYPE'].notna() &
                schema_df['CONSTRAINT_TYPE'].str.contains('FOREIGN KEY', case=False, na=False)
            )
            fk_schema = schema_df[fk_filter][relevant_columns]
            
            if not fk_schema.empty:
                filtered_schema_df = pd.concat([filtered_schema_df, fk_schema]).drop_duplicates()
        
        # Step 8: Limit results to prevent overwhelming the LLM
        max_rows = 50
        if len(filtered_schema_df) > max_rows:
            # Prioritize: Primary Keys > Foreign Keys > Matching columns > Others
            priority_df_list = []
            
            # Primary keys first
            pk_mask = (filtered_schema_df['CONSTRAINT_TYPE'].notna() & 
                      filtered_schema_df['CONSTRAINT_TYPE'].str.contains('PRIMARY KEY', case=False, na=False))
            if pk_mask.any():
                priority_df_list.append(filtered_schema_df[pk_mask])
            
            # Foreign keys second
            fk_mask = (filtered_schema_df['CONSTRAINT_TYPE'].notna() & 
                      filtered_schema_df['CONSTRAINT_TYPE'].str.contains('FOREIGN KEY', case=False, na=False))
            if fk_mask.any():
                priority_df_list.append(filtered_schema_df[fk_mask])
            
            # Columns matching user input third
            remaining_df = filtered_schema_df[~(pk_mask | fk_mask)]
            for keyword in keywords:
                keyword_mask = (
                    remaining_df['columnname'].str.contains(keyword, case=False, na=False, regex=False) |
                    (remaining_df['Description'].notna() & 
                     remaining_df['Description'].str.contains(keyword, case=False, na=False, regex=False))
                )
                if keyword_mask.any():
                    priority_df_list.append(remaining_df[keyword_mask])
            
            # Add remaining rows if we haven't reached the limit
            if priority_df_list:
                filtered_schema_df = pd.concat(priority_df_list).drop_duplicates().head(max_rows)
            else:
                filtered_schema_df = filtered_schema_df.head(max_rows)
        
        # Step 9: Fallback if still empty
        if filtered_schema_df.empty:
            logger.warning("No matching schema found, using minimal schema with primary keys")
            # Get primary keys from all tables as fallback
            pk_fallback = schema_df[
                schema_df['CONSTRAINT_TYPE'].notna() &
                schema_df['CONSTRAINT_TYPE'].str.contains('PRIMARY KEY', case=False, na=False)
            ][relevant_columns].head(15)
            
            if not pk_fallback.empty:
                filtered_schema_df = pk_fallback
            else:
                # Last resort - just take first few rows
                filtered_schema_df = schema_df[relevant_columns].head(10)
        
        return filtered_schema_df
        
    except Exception as filter_error:
        logger.error(f"Error in enhanced schema filtering: {filter_error}")
        # Fallback to basic filtering
        return schema_df[relevant_columns].head(20)

def format_schema_for_llm(filtered_schema_df):
    """
    Format the filtered schema for optimal LLM consumption
    """
    try:
        schema_info = []
        
        # Group by table for better organization
        if not filtered_schema_df.empty:
            grouped_df = filtered_schema_df.groupby('Table_Name')
            
            for table_name, table_group in grouped_df:
                schema_info.append(f"\n--- {table_name} ---")
                
                for _, row in table_group.iterrows():
                    info = f"{row['columnname']}"
                    
                    # Add constraint information
                    if pd.notna(row['CONSTRAINT_TYPE']):
                        if 'PRIMARY KEY' in str(row['CONSTRAINT_TYPE']):
                            info += "(PK)"
                        elif 'FOREIGN KEY' in str(row['CONSTRAINT_TYPE']):
                            info += "(FK)"
                    
                    # Add description if available and concise
                    if pd.notna(row['Description']) and len(str(row['Description'])) < 50:
                        info += f" - {row['Description']}"
                    
                    schema_info.append(info)
        
        schema_description = "\n".join(schema_info)
        
        # Add summary information
        table_count = len(filtered_schema_df['Table_Name'].unique()) if not filtered_schema_df.empty else 0
        column_count = len(filtered_schema_df)
        
        summary = f"Schema Summary: {table_count} tables, {column_count} columns\n"
        schema_description = summary + schema_description
        
        return schema_description
        
    except Exception as e:
        logger.error(f"Error formatting schema: {e}")
        return "Error formatting schema information"


@route_app.route('/ChatGenie/v1/oldchat', methods=['POST'])
def chat():
    logger.debug("Received request at /chat endpoint")
    data = request.get_json()
    logger.debug(f"Request data: {data}")
    user_input = data.get('message')
    UserId = data.get('UserId')
    print("UserId", UserId)
    db_name_input = data.get('db_name')
    db_name = db_name_input.upper() if db_name_input else None
    
    if not user_input:
        logger.warning("No message provided in the request")
        return jsonify({'error': 'No message provided'}), 400

    if not db_name:
        logger.warning("No database name provided in the request")
        return jsonify({'error': 'No database name provided'}), 400

    # Read the database schema from an Excel file
    try:
        tenant_base_config = json.loads(os.getenv('DATABASE_CONNECTIONS_JSON').replace("'", '"'))
        file_config = tenant_base_config[db_name]
        logger.debug(f"File config: {file_config}")
        # Use cached schema reading
        schema_df = get_cached_schema(file_config["metadatapath"])
        logger.info(f"Successfully read schema with {schema_df}")
        logger.debug(f"Successfully read schema with {len(schema_df)} rows")

        # Enhanced schema filtering
        try:
            # Define relevant columns to keep
            relevant_columns = ['Table_Name', 'columnname', 'CONSTRAINT_TYPE', 'Description']
            
            # Use the enhanced filtering function
            filtered_schema_df = filter_schema_intelligently(schema_df, user_input, relevant_columns)
            logger.debug(f"Filtered schema to {len(filtered_schema_df)} rows from {len(filtered_schema_df['Table_Name'].unique())} tables")

            # Format schema for LLM consumption
            schema_description = format_schema_for_llm(filtered_schema_df)

        except Exception as filter_error:
            logger.error(f"Error in schema filtering: {filter_error}")
            return jsonify({'error': 'Error filtering schema data'}), 500
            
    except Exception as e:
        logger.error(f"Error in schema processing: {str(e)}")
        return jsonify({'error': 'Error processing schema data'}), 500

    # Generate SQL query with optimized prompt
    prompt = f"""You are an expert SQL developer. Generate an accurate and optimized Azure SQL query for the following user request.

User request: {user_input}

Database schema information:
{schema_description}

Important guidelines:
- Use proper JOIN conditions when multiple tables are involved
- Include primary key and foreign key relationships
- Use appropriate WHERE clauses for filtering
- Ensure column names match exactly as shown in schema
- Return only the SQL query without explanations

SQL:"""
    
    logger.debug(f"Generated Prompt: {prompt}")
    sql_query = generate_sql_query(prompt)
    logger.debug(f"Generated SQL query: {sql_query}")

    # Check database connection
    try:
        connection = get_db_connection(db_name)
        print(db_name)
        connection.close()
        logger.debug("Database connection successful")
    except OperationalError as e:
        logger.error(f"Database connection error: {e}")
        return jsonify({'response': "Database connection error. Please try again later."}), 500
    except Exception as e:
        logger.error(f"Unexpected error during database connection check: {e}")
        return jsonify({'response': "An unexpected error occurred. Please try again later."}), 500

    try:
        result = query_database(db_name, sql_query)
        logger.debug(f"Query result: {result}")
    except SQLAlchemyError as e:
        logger.error(f"SQLAlchemy error: {e}")
        return jsonify({'response': "Database query error. Please provide valid queries."}), 400
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return jsonify({'response': "An unexpected error occurred. Please try again later."}), 500

    if result:
        logger.info("Query successful, returning result")
        return jsonify({'response': result})
    else:
        logger.info("No data found for the given query")
        return jsonify({'response': 'No data found for the given query'}), 404