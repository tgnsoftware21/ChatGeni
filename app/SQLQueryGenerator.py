from sqlite3 import OperationalError
from flask import Blueprint, jsonify, request
import pandas as pd
import openai
from openai import OpenAI
import os
from typing import Dict, Any, Optional
import json
from datetime import datetime
import logging
from sqlalchemy.exc import SQLAlchemyError
from app.db import get_db_connection, query_database

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SQLQueryGenerator:
    """
    Dynamic SQL Query Generator using OpenAI and Excel Schema
    """
    
    def __init__(self, openai_api_key: str = None, model: str = "gpt-4", 
                 azure_endpoint: str = None, api_version: str = "2024-02-01",
                 deployment_name: str = None):
        """
        Initialize the SQL Query Generator
        
        Args:
            openai_api_key: OpenAI API key (if None, will try to get from environment)
            model: OpenAI model to use (default: gpt-4)
            azure_endpoint: Azure OpenAI endpoint (if using Azure OpenAI)
            api_version: Azure API version
            deployment_name: Azure OpenAI deployment name (required for Azure, overrides model)
        """
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY') or os.getenv('AZURE_OPENAI_KEY')
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY or AZURE_OPENAI_KEY environment variable or pass it directly.")
        
        self.azure_endpoint = azure_endpoint or os.getenv('AZURE_OPENAI_ENDPOINT')
        self.api_version = api_version
        self.is_azure = bool(self.azure_endpoint)
        
        # For Azure OpenAI, use deployment name if provided, otherwise use model name
        if self.is_azure:
            self.deployment_name = deployment_name or os.getenv('AZURE_OPENAI_DEPLOYMENT') or model
            self.model = model  # Keep original model name for reference
        else:
            self.model = model
            self.deployment_name = None
        
        # Initialize the appropriate client
        if self.is_azure:
            from openai import AzureOpenAI
            self.client = AzureOpenAI(
                api_key=self.openai_api_key,
                azure_endpoint=self.azure_endpoint,
                api_version=self.api_version
            )
            logger.info(f"Initialized Azure OpenAI client with endpoint: {self.azure_endpoint}")
            logger.info(f"Using deployment: {self.deployment_name}")
        else:
            self.client = OpenAI(api_key=self.openai_api_key)
            logger.info("Initialized OpenAI client")
        
        self.schema_cache = {}
        
        # Different prompt templates for different model types
        self.chat_prompt_template = """You are an expert SQL developer. Generate an accurate and optimized Azure SQL query for the following user request.

User request: {user_input}

Database schema information: {schema_description}

Important guidelines:
- Use proper JOIN conditions when multiple tables are involved
- Include primary key and foreign key relationships
- Use appropriate WHERE clauses for filtering
- Ensure column names match exactly as shown in schema
- Return only the SQL query without explanations

SQL:"""
        
        self.completion_prompt_template = """You are an expert SQL developer. Generate an accurate and optimized Azure SQL query for the following user request.

User request: {user_input}

Database schema information: {schema_description}

Important guidelines:
- Use proper JOIN conditions when multiple tables are involved
- Include primary key and foreign key relationships
- Use appropriate WHERE clauses for filtering
- Ensure column names match exactly as shown in schema
- Return only the SQL query without explanations

SQL:"""
    
    def read_schema_from_excel(self, excel_file_path: str, sheet_name: str = 'Sheet1') -> Dict[str, Any]:
        """
        Read database schema from Excel file
        
        Args:
            excel_file_path: Path to the Excel file containing schema
            sheet_name: Name of the sheet to read (default: 'Sheet1')
            
        Returns:
            Dictionary containing schema information
        """
        try:
            # Read Excel file
            df = pd.read_excel(excel_file_path, sheet_name=sheet_name)
            
            # Assuming standard schema format with columns: Table_Schema, Table_Name, columnname, CONSTRAINT_TYPE, Reference, Description
            schema_info = {
                'tables': {},
                'relationships': [],
                'raw_data': df.to_dict('records')
            }
            
            # Process schema data
            for _, row in df.iterrows():
                # Safely get values and handle NaN
                table_name = str(row.get('Table_Name', '')).strip() if pd.notna(row.get('Table_Name')) else ''
                column_name = str(row.get('columnname', '')).strip() if pd.notna(row.get('columnname')) else ''
                constraint_type = row.get('CONSTRAINT_TYPE', '')
                reference = row.get('Reference', '')
                description = row.get('Description', '')
                
                if table_name and column_name:
                    # Initialize table if not exists
                    if table_name not in schema_info['tables']:
                        schema_info['tables'][table_name] = {
                            'columns': [],
                            'primary_keys': [],
                            'foreign_keys': [],
                            'unique_keys': []
                        }
                    
                    # Clean and convert values to strings, handling NaN values
                    clean_constraint = str(constraint_type).strip() if pd.notna(constraint_type) else ''
                    clean_reference = str(reference).strip() if pd.notna(reference) else ''
                    clean_description = str(description).strip() if pd.notna(description) else ''
                    
                    # Add column information
                    column_info = {
                        'name': column_name,
                        'constraint': clean_constraint,
                        'reference': clean_reference,
                        'description': clean_description
                    }
                    schema_info['tables'][table_name]['columns'].append(column_info)
                    
                    # Categorize constraints (now safe to use .lower() after converting to string)
                    if clean_constraint and 'primary' in clean_constraint.lower():
                        schema_info['tables'][table_name]['primary_keys'].append(column_name)
                    elif clean_constraint and 'foreign' in clean_constraint.lower():
                        schema_info['tables'][table_name]['foreign_keys'].append({
                            'column': column_name,
                            'reference': clean_reference
                        })
                    elif clean_constraint and 'unique' in clean_constraint.lower():
                        schema_info['tables'][table_name]['unique_keys'].append(column_name)
            
            # Cache the schema
            self.schema_cache[excel_file_path] = schema_info
            logger.info(f"Schema loaded successfully from {excel_file_path}")
            logger.info(f"Found {len(schema_info['tables'])} tables")
            
            return schema_info
            
        except Exception as e:
            logger.error(f"Error reading schema from Excel: {str(e)}")
            raise
    
    def format_schema_for_prompt(self, schema_info: Dict[str, Any]) -> str:
        """
        Format schema information for the OpenAI prompt
        
        Args:
            schema_info: Schema information dictionary
            
        Returns:
            Formatted schema description string
        """
        formatted_schema = []
        
        for table_name, table_info in schema_info['tables'].items():
            formatted_schema.append(f"\nTable: {table_name}")
            
            # Add columns
            formatted_schema.append("Columns:")
            for col in table_info['columns']:
                constraint_text = f" ({col['constraint']})" if col['constraint'] and col['constraint'] != 'nan' else ""
                description_text = f" - {col['description']}" if col['description'] and col['description'] != 'nan' else ""
                formatted_schema.append(f"  - {col['name']}{constraint_text}{description_text}")
            
            # Add primary keys
            if table_info['primary_keys']:
                formatted_schema.append(f"Primary Keys: {', '.join(table_info['primary_keys'])}")
            
            # Add foreign keys
            if table_info['foreign_keys']:
                fk_list = [f"{fk['column']} -> {fk['reference']}" for fk in table_info['foreign_keys']]
                formatted_schema.append(f"Foreign Keys: {', '.join(fk_list)}")
            
            # Add unique keys
            if table_info['unique_keys']:
                formatted_schema.append(f"Unique Keys: {', '.join(table_info['unique_keys'])}")
        
        return '\n'.join(formatted_schema)
    
    def generate_sql_query(self, user_input: str, excel_schema_path: str, sheet_name: str = 'Sheet1') -> Dict[str, Any]:
        """
        Generate SQL query based on user input and Excel schema
        
        Args:
            user_input: User's request for SQL query
            excel_schema_path: Path to Excel file containing database schema
            sheet_name: Sheet name in Excel file
            
        Returns:
            Dictionary containing generated SQL and metadata
        """
        try:
            # Read schema from Excel
            schema_info = self.read_schema_from_excel(excel_schema_path, sheet_name)
            
            # Format schema for prompt
            schema_description = self.format_schema_for_prompt(schema_info)
            
            # Create the prompt
            if self._is_chat_model():
                prompt = self.chat_prompt_template.format(
                    user_input=user_input,
                    schema_description=schema_description
                )
            else:
                prompt = self.completion_prompt_template.format(
                    user_input=user_input,
                    schema_description=schema_description
                )
            
            logger.info(f"Generating SQL query for: {user_input}")
            logger.info(f"Using model: {self.model} (Chat API: {self._is_chat_model()})")
            
            # Call OpenAI API
            sql_query, tokens_used = self._call_openai_api(prompt)
            
            # Clean up the SQL query (remove any extra formatting)
            if sql_query.lower().startswith('sql:'):
                sql_query = sql_query[4:].strip()
            
            # Remove markdown code blocks if present
            if sql_query.startswith('```'):
                lines = sql_query.split('\n')
                sql_query = '\n'.join(lines[1:-1]) if len(lines) > 2 else sql_query
            
            result = {
                'success': True,
                'sql_query': sql_query,
                'user_input': user_input,
                'schema_tables': list(schema_info['tables'].keys()),
                'timestamp': datetime.now().isoformat(),
                'model_used': self.model,
                'tokens_used': tokens_used,
                'api_type': 'Azure OpenAI' if self.is_azure else 'OpenAI',
                'chat_api': self._is_chat_model()
            }
            
            logger.info("SQL query generated successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error generating SQL query: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'user_input': user_input,
                'timestamp': datetime.now().isoformat()
            }
            
    def _is_chat_model(self) -> bool:
        """
        Determine if the model supports chat completions API
        
        Returns:
            True if model supports chat completions, False otherwise
        """
        chat_models = [
            'gpt-4', 'gpt-4-turbo', 'gpt-4-32k', 'gpt-4-0613', 'gpt-4-32k-0613',
            'gpt-4-1106-preview', 'gpt-4-0125-preview', 'gpt-4-turbo-preview',
            'gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'gpt-3.5-turbo-0613', 
            'gpt-3.5-turbo-16k-0613', 'gpt-3.5-turbo-1106', 'gpt-3.5-turbo-0125',
            'gpt-35-turbo'  # Azure naming convention
        ]
        
        # Check if model name contains any chat model identifier
        model_lower = self.model.lower()
        return any(chat_model in model_lower for chat_model in chat_models)
    
    def _call_openai_api(self, prompt: str) -> tuple:
        """
        Call the appropriate OpenAI API based on model type
        
        Args:
            prompt: The formatted prompt
            
        Returns:
            Tuple of (response_text, usage_info)
        """
        try:
            # Use deployment name for Azure, model name for regular OpenAI
            model_to_use = self.deployment_name if self.is_azure else self.model
            print(f"Using model: {model_to_use} for API call")
            if self._is_chat_model():
                # Use chat completions API
                response = self.client.chat.completions.create(
                    model=model_to_use,
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are an expert SQL developer specializing in Azure SQL Database queries."
                        },
                        {
                            "role": "user", 
                            "content": prompt
                        }
                    ],
                    max_tokens=1000,
                    temperature=0.1
                )
                return response.choices[0].message.content.strip(), response.usage.total_tokens
            else:
                # Use legacy completions API for models like gpt-3.5-turbo-instruct
                response = self.client.completions.create(
                    model=model_to_use,
                    prompt=prompt,
                    max_tokens=1000,
                    temperature=0.1,
                    stop=None
                )
                return response.choices[0].text.strip(), response.usage.total_tokens
                
        except Exception as e:
            logger.error(f"API call failed: {str(e)}")
            # Provide helpful error message for deployment issues
            if "DeploymentNotFound" in str(e):
                logger.error(f"Deployment '{model_to_use}' not found. Available deployments in your Azure OpenAI resource:")
                logger.error("1. Check Azure Portal -> Your OpenAI Resource -> Model deployments")
                logger.error("2. Use the exact deployment name, not the model name")
                logger.error("3. Common deployment names: 'gpt-35-turbo', 'gpt-4', 'text-davinci-003'")
            raise
            
            # Clean up the SQL query (remove any extra formatting)
            if sql_query.lower().startswith('sql:'):
                sql_query = sql_query[4:].strip()
            
            # Remove markdown code blocks if present
            if sql_query.startswith('```'):
                lines = sql_query.split('\n')
                sql_query = '\n'.join(lines[1:-1]) if len(lines) > 2 else sql_query
            
            result = {
                'success': True,
                'sql_query': sql_query,
                'user_input': user_input,
                'schema_tables': list(schema_info['tables'].keys()),
                'timestamp': datetime.now().isoformat(),
                'model_used': self.model,
                'tokens_used': tokens_used,
                'api_type': 'Azure OpenAI' if self.is_azure else 'OpenAI',
                'chat_api': self._is_chat_model()
            }
            
            logger.info("SQL query generated successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error generating SQL query: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'user_input': user_input,
                'timestamp': datetime.now().isoformat()
            }
    
    def generate_multiple_queries(self, user_inputs: list, excel_schema_path: str, sheet_name: str = 'Sheet1') -> list:
        """
        Generate multiple SQL queries in batch
        
        Args:
            user_inputs: List of user input strings
            excel_schema_path: Path to Excel file containing database schema
            sheet_name: Sheet name in Excel file
            
        Returns:
            List of query results
        """
        results = []
        
        for user_input in user_inputs:
            result = self.generate_sql_query(user_input, excel_schema_path, sheet_name)
            results.append(result)
        
        return results
    
    def validate_sql_syntax(self, sql_query: str) -> Dict[str, Any]:
        """
        Basic SQL syntax validation
        
        Args:
            sql_query: SQL query to validate
            
        Returns:
            Validation result dictionary
        """
        try:
            # Basic syntax checks
            sql_lower = sql_query.lower().strip()
            
            issues = []
            
            # Check for common issues
            if not sql_lower.startswith(('select', 'with', 'insert', 'update', 'delete')):
                issues.append("Query should start with a valid SQL statement (SELECT, INSERT, etc.)")
            
            # Check for balanced parentheses
            if sql_query.count('(') != sql_query.count(')'):
                issues.append("Unbalanced parentheses")
            
            # Check for basic SQL keywords
            keywords = ['select', 'from', 'where', 'group by', 'order by', 'having']
            found_keywords = [kw for kw in keywords if kw in sql_lower]
            
            return {
                'is_valid': len(issues) == 0,
                'issues': issues,
                'found_keywords': found_keywords,
                'query_type': sql_lower.split()[0].upper() if sql_lower else 'UNKNOWN'
            }
            
        except Exception as e:
            return {
                'is_valid': False,
                'issues': [f"Validation error: {str(e)}"],
                'found_keywords': [],
                'query_type': 'UNKNOWN'
            }

# API-like wrapper class
class SQLGeneratorAPI:
    """
    API wrapper for the SQL Generator
    """
    
    def __init__(self, openai_api_key: str = None, model: str = "gpt-4", 
                 azure_endpoint: str = None, api_version: str = "2024-02-01",
                 deployment_name: str = None):
        self.generator = SQLQueryGenerator(openai_api_key, model, azure_endpoint, api_version, deployment_name)
    
    def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process API request for SQL generation
        
        Args:
            request_data: Dictionary containing:
                - user_input: User's SQL request
                - excel_schema_path: Path to schema Excel file
                - sheet_name: Optional sheet name
                - validate: Optional boolean to validate syntax
                
        Returns:
            API response dictionary
        """
        try:
            user_input = request_data.get('message')
            sheet_name = request_data.get('sheet_name', 'Sheet1')
            validate = request_data.get('validate', True)
            tenant_base_config = json.loads(os.getenv('DATABASE_CONNECTIONS_JSON').replace("'", '"'))
            file_config = tenant_base_config[request_data.get('db_name').upper()]
            excel_schema_path = file_config.get('metadatapath')
            
            if not user_input:
                return {
                    'success': False,
                    'error': 'user_input is required',
                    'timestamp': datetime.now().isoformat()
                }
            
            if not excel_schema_path:
                return {
                    'success': False,
                    'error': 'excel_schema_path is required',
                    'timestamp': datetime.now().isoformat()
                }
            
            # Generate SQL query
            result = self.generator.generate_sql_query(user_input, excel_schema_path, sheet_name)
            
            # Add validation if requested
            if validate and result['success']:
                validation_result = self.generator.validate_sql_syntax(result['sql_query'])
                result['validation'] = validation_result
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

# Example usage and testing functions
def demo_sql_generator():
    """
    Demonstrate the SQL Generator functionality
    """
    # Configuration for Azure OpenAI
    azure_config = {
        'openai_api_key': os.getenv('AZURE_OPENAI_KEY'),  # Your Azure OpenAI key
        'model': os.getenv('AZURE_MODEL'),  # Use a chat model for Azure
        'azure_endpoint': os.getenv('AZURE_OPENAI_ENDPOINT'),  # Your Azure endpoint
        'api_version': os.getenv('AZURE_API_VERSION')
    }
    
    # Configuration for regular OpenAI (alternative)
    openai_config = {
        'openai_api_key': os.getenv('OPENAI_API_KEY'),
        'model': os.getenv('AZURE_MODEL')
    }
    
    # Use Azure config if available, otherwise use OpenAI
    if azure_config['azure_endpoint'] and azure_config['openai_api_key']:
        print("Using Azure OpenAI...")
        generator = SQLQueryGenerator(**azure_config)
    elif openai_config['openai_api_key']:
        print("Using OpenAI...")
        generator = SQLQueryGenerator(**openai_config)
    else:
        print("Error: No API keys found. Set AZURE_OPENAI_KEY + AZURE_OPENAI_ENDPOINT or OPENAI_API_KEY")
        return
    
    # Example schema path (replace with your actual path)
    excel_path = "GenPi_Schema V1.xlsx"
    
    # Example queries
    example_queries = [
        "provide me property detail with property number, Client Code, Occupancy Status",
        "Number of work orders issued for Client 'PFC' in July",
        "Show all properties in New York with vacant occupancy status",
        "Get work order details for property number P12345",
        "Count of active properties by client code"
    ]
    
    print("=== SQL GENERATOR DEMO ===\n")
    
    for i, query in enumerate(example_queries, 1):
        print(f"{i}. User Input: {query}")
        
        result = generator.generate_sql_query(query, excel_path)
        
        if result['success']:
            print(f"Generated SQL:")
            print(result['sql_query'])
            print(f"Tables used: {result['schema_tables']}")
            print(f"Tokens used: {result['tokens_used']}")
            print(f"API Type: {result['api_type']}")
            print(f"Using Chat API: {result['chat_api']}")
            
            # Validate syntax
            validation = generator.validate_sql_syntax(result['sql_query'])
            print(f"Syntax valid: {validation['is_valid']}")
            if validation['issues']:
                print(f"Issues: {validation['issues']}")
        else:
            print(f"Error: {result['error']}")
        
        print("-" * 80)

def api_example():
    """
    Example of using the API wrapper
    """
    # Configuration for Azure OpenAI
    if os.getenv('AZURE_OPENAI_ENDPOINT'):
        api = SQLGeneratorAPI(
            openai_api_key=os.getenv('AZURE_OPENAI_KEY'),
            model=os.getenv('AZURE_MODEL'),  # Use chat model instead of instruct
            azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
            api_version=os.getenv('AZURE_API_VERSION')
        )
    else:
        api = SQLGeneratorAPI(model=os.getenv('AZURE_MODEL'))
    
    # Example API request
    request = {
        'user_input': 'provide me property detail with property number, Client Code, Occupancy Status',
        'excel_schema_path': 'GenPi_Schema V1.xlsx',
        'sheet_name': 'Sheet1',
        'validate': True
    }
    
    # Process request
    response = api.process_request(request)
    
    print("=== API RESPONSE ===")
    print(json.dumps(response, indent=2))

def complete_example():
    """
    Complete example showing all configuration options and usage patterns
    """
    print("=== COMPLETE CONFIGURATION EXAMPLES ===\n")
    
    # Example 1: Azure OpenAI with chat model (recommended)
    try:
        azure_chat_generator = SQLQueryGenerator(
            openai_api_key=os.getenv('AZURE_OPENAI_KEY'),
            model=os.getenv('AZURE_MODEL'),  # Chat model
            azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
            api_version=os.getenv('AZURE_API_VERSION')
        )
        print("✓ Azure OpenAI with chat model configured successfully")
    except Exception as e:
        print(f"✗ Azure OpenAI with chat model failed: {str(e)}")
    
    # Example 2: Azure OpenAI with instruct model
    try:
        azure_instruct_generator = SQLQueryGenerator(
            openai_api_key=os.getenv('AZURE_OPENAI_KEY'),
            model='gpt-35-turbo-instruct',  # Instruct model
            azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
            api_version='2024-02-01'
        )
        print("✓ Azure OpenAI with instruct model configured successfully")
    except Exception as e:
        print(f"✗ Azure OpenAI with instruct model failed: {str(e)}")
    
    # Example 3: Regular OpenAI
    try:
        openai_generator = SQLQueryGenerator(
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            model=os.getenv('AZURE_MODEL')
        )
        print("✓ Regular OpenAI configured successfully")
    except Exception as e:
        print(f"✗ Regular OpenAI failed: {str(e)}")
    
    print("\n=== TESTING SQL GENERATION ===")
    
    # Test with the first available generator
    test_generator = None
    if os.getenv('AZURE_OPENAI_ENDPOINT') and os.getenv('AZURE_OPENAI_KEY'):
        test_generator = SQLQueryGenerator(
            openai_api_key=os.getenv('AZURE_OPENAI_KEY'),
            model=os.getenv('AZURE_MODEL'),
            azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT')
        )
        print("Using Azure OpenAI for testing...")
    elif os.getenv('OPENAI_API_KEY'):
        test_generator = SQLQueryGenerator(
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            model=os.getenv('AZURE_MODEL')
        )
        print("Using OpenAI for testing...")
    
    if test_generator:
        test_query = "provide me property detail with property number, Client Code, Occupancy Status"
        result = test_generator.generate_sql_query(test_query, "GenPi_Schema V1.xlsx")
        
        if result['success']:
            print(f"\n✓ Test Query Generated Successfully:")
            print(f"Input: {result['user_input']}")
            print(f"SQL: {result['sql_query']}")
            print(f"Model: {result['model_used']}")
            print(f"API Type: {result['api_type']}")
        else:
            print(f"\n✗ Test Query Failed: {result['error']}")
    else:
        print("No API keys configured for testing")

def environment_setup_guide():
    """
    Print environment setup guide
    """
    print("=== ENVIRONMENT SETUP GUIDE ===\n")
    
    print("Option 1: Azure OpenAI Setup")
    print("export AZURE_OPENAI_KEY='your-azure-openai-api-key'")
    print("export AZURE_OPENAI_ENDPOINT='https://your-resource-name.openai.azure.com/'")
    print("export AZURE_OPENAI_DEPLOYMENT='your-deployment-name'  # IMPORTANT: Use exact deployment name")
    print("# Common deployment names: gpt-35-turbo, gpt-4, text-davinci-003")
    print()
    
    print("How to find your deployment name:")
    print("1. Go to Azure Portal")
    print("2. Navigate to your OpenAI resource")
    print("3. Click on 'Model deployments'")
    print("4. Use the 'Deployment name' (not the model name)")
    print()
    
    print("Option 2: Regular OpenAI Setup") 
    print("export OPENAI_API_KEY='your-openai-api-key'")
    print("# Models: gpt-4, gpt-3.5-turbo")
    print()
    
    print("Python Installation:")
    print("pip install pandas openpyxl openai python-dotenv")
    print()
    
    print("Usage Example:")
    print("""
from sql_generator import SQLQueryGenerator

# For Azure OpenAI (with deployment name)
generator = SQLQueryGenerator(
    model='gpt-35-turbo',  # Model type for API selection
    azure_endpoint='https://your-resource.openai.azure.com/',
    deployment_name='your-exact-deployment-name'  # Critical!
)

# Generate SQL
result = generator.generate_sql_query(
    "show me all properties for client PFC",
    "GenPi_Schema V1.xlsx"
)

if result['success']:
    print(result['sql_query'])
else:
    print(result['error'])
""")

# if __name__ == "__main__":
#     # Environment variables setup examples:
#     # For Azure OpenAI:
#     # os.environ['AZURE_OPENAI_KEY'] = 'your-azure-openai-key'
#     # os.environ['AZURE_OPENAI_ENDPOINT'] = 'https://your-resource.openai.azure.com/'
    
#     # For regular OpenAI:
#     # os.environ['OPENAI_API_KEY'] = 'your-openai-api-key'
    
#     try:
#         # Show environment setup guide
#         environment_setup_guide()
        
#         # Run complete examples
#         complete_example()
        
#         # Run demo
#         demo_sql_generator()
        
#         # Run API example
#         api_example()
        
#     except Exception as e:
#         print(f"Error: Make sure to set your API keys and endpoints")
#         print(f"For Azure OpenAI: AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT")
#         print(f"For OpenAI: OPENAI_API_KEY")
#         print(f"Error details: {str(e)}")

# Additional utility functions
def quick_test(user_input: str, excel_path: str = "GenPi_Schema V1.xlsx", deployment_name: str = None):
    """
    Quick test function for immediate SQL generation
    
    Args:
        user_input: SQL request in natural language
        excel_path: Path to Excel schema file
        deployment_name: Azure OpenAI deployment name (if not set in environment)
    """
    try:
        # Try Azure OpenAI first, then regular OpenAI
        if os.getenv('AZURE_OPENAI_ENDPOINT'):
            actual_deployment = deployment_name or os.getenv('AZURE_OPENAI_DEPLOYMENT')
            if not actual_deployment:
                print("ERROR: Azure OpenAI deployment name is required!")
                print("Either pass deployment_name parameter or set AZURE_OPENAI_DEPLOYMENT environment variable")
                print("Find your deployment name in Azure Portal -> OpenAI Resource -> Model deployments")
                return
            
            generator = SQLQueryGenerator(
                model=os.getenv('AZURE_MODEL'),  # Model type for API selection
                azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
                deployment_name=actual_deployment
            )
            print(f"Using Azure OpenAI deployment: {actual_deployment}")
        else:
            generator = SQLQueryGenerator(model=os.getenv('AZURE_MODEL'))
            print("Using OpenAI")
        
        result = generator.generate_sql_query(user_input, excel_path)
        
        if result['success']:
            print("Generated SQL Query:")
            print("=" * 50)
            print(result['sql_query'])
            print("=" * 50)
            print(f"Model: {result['model_used']}")
            print(f"API: {result['api_type']}")
            print(f"Tokens: {result['tokens_used']}")
        else:
            print(f"Error: {result['error']}")
    
    except Exception as e:
        print(f"Quick test failed: {str(e)}")

# Example usage:
# quick_test("show me property details with client code PFC", deployment_name="your-deployment-name")
#Flask API Implementation (uncomment to use):

generate_sql = Blueprint("generate_sql", __name__)

@generate_sql.route('/chatgenie/v1/chat', methods=['POST'])
def generate_sql_endpoint():
    try:
        data = request.json
        
        # Initialize generator based on configuration
        if os.getenv('AZURE_OPENAI_ENDPOINT'):
            api = SQLGeneratorAPI(
                model=data.get('model', os.getenv('AZURE_MODEL')),
                azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT')
            )
        else:
            api = SQLGeneratorAPI(model=data.get('model', os.getenv('OPENAI_MODEL')))
        
        response = api.process_request(data)
        
        try:
            print(f"Checking database connection...")
            connection = get_db_connection(data.get('db_name').upper())
            print(f"Checking database connection...{connection}")
            connection.close()
            logger.debug("Database connection successful")
        except OperationalError as e:
            logger.error(f"Database connection error: {e}")
            return jsonify({'response': "Database connection error. Please try again later."}), 500
        except Exception as e:
            logger.error(f"Unexpected error during database connection check: {e}")
            return jsonify({'response': "An unexpected error occurred. Please try again later."}), 500

        try:
            print(f"Executing SQL query: {response}")
            result = query_database('DEV', response['sql_query'])
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
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'user_input': data.get('message', ''),
            'timestamp': datetime.now().isoformat()
        }), 500

@generate_sql.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

"""

# Flask API endpoint example (optional)
from flask import Blueprint, Flask, request, jsonify

generate_sql = Blueprint("generate_sql", __name__)

@generate_sql.route('/', methods=['POST'])
def generate_sql_endpoint():
    try:
        request_data = request.json
        response = api.process_request(request_data)
        return jsonify(response)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

"""