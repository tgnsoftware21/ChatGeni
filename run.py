#!/usr/bin/env python3
import os
import sys
import logging
from typing import Optional
from flask import Flask
from logging.handlers import RotatingFileHandler
from pathlib import Path
from flask_cors import CORS

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app


def setup_file_logging(app: Flask) -> None:
    """Configure file-based logging for the application."""
    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(exist_ok=True)
    
    file_handler = RotatingFileHandler(
        log_dir / "application.log",
        maxBytes=1024 * 1024,
        backupCount=10
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.DEBUG)  # Changed to DEBUG
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.DEBUG)
    app.logger.info('Application startup')

def get_ssl_context() -> Optional[tuple]:
    """Get SSL context for HTTPS support if certificates are configured."""
    cert_path = os.getenv('SSL_CERT_PATH')
    key_path = os.getenv('SSL_KEY_PATH')
    
    if cert_path and key_path and os.path.exists(cert_path) and os.path.exists(key_path):
        return (cert_path, key_path)
    return None

def main() -> None:
    """Main application entry point."""
    # # Debug: Print all environment variables
    # app.logger.debug("--- Environment Variables ---")
    # for key, value in os.environ.items():
    #     app.logger.debug(f"{key}: {value}")
    
    # Create Flask application instance
    app = create_app()
    
    try:
        
        CORS(app, origins=["https://localhost:5001"]) 
        # Initialize logging
        setup_file_logging(app)
        
        # Get server configuration
        host = os.getenv('HOST', '0.0.0.0')
        
        # Verbose port parsing with extensive logging
        port_env = os.getenv('PORT')
        app.logger.debug(f"Raw PORT value: {port_env}")
        app.logger.debug(f"Type of PORT value: {type(port_env)}")
        
        # Robust port parsing
        try:
            if port_env is None:
                app.logger.warning("PORT is None, using default")
                port = 8082
            elif port_env.strip() == '':
                app.logger.warning("PORT is empty string, using default")
                port = 8082
            else:
                port = int(port_env)
                app.logger.info(f"Successfully parsed port: {port}")
        except ValueError as ve:
            app.logger.error(f"ValueError parsing port: {ve}")
            app.logger.error(f"Invalid PORT value: {port_env}. Using default port 8082.")
            port = 8082
        
        debug = os.getenv('FLASK_ENV', 'production').lower() == 'development'
        
        # Configure SSL if available
        ssl_context = get_ssl_context()
        
        # Log server startup details
        app.logger.info(f"Starting server on host {host}, port {port}")
        app.logger.info(f"Debug mode: {debug}")
        
        # Start the server
        app.run(
            host=host,
            port=port,
            debug=debug,
            ssl_context=ssl_context
        )
        #app.run(ssl_context=('localhost.crt', 'localhost.key'),host='0.0.0.0',port=8082,debug=True)

    except Exception as error:
        app.logger.error(f"Catastrophic error: {error}", exc_info=True)
        print(f"Failed to start server: {error}")
        sys.exit(1)

if __name__ == '__main__':
    main()