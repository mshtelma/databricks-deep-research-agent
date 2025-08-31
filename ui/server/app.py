"""FastAPI application for Databricks App Template."""

import logging
import os
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from server.routers import router


# Load environment variables from .env.local if it exists
def load_env_file(filepath: str) -> None:
  """Load environment variables from a file."""
  if Path(filepath).exists():
    with open(filepath) as f:
      for line in f:
        line = line.strip()
        if line and not line.startswith('#'):
          key, _, value = line.partition('=')
          if key and value:
            os.environ[key] = value


# Load .env files
load_env_file('.env')
load_env_file('.env.local')


# Configure logging
def setup_logging():
  """Set up logging configuration."""
  log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
  
  # Create handlers
  handlers = [
    logging.StreamHandler(),  # Console output (goes to watch log)
  ]
  
  # Add file handlers if /tmp exists
  if os.path.exists('/tmp'):
    handlers.extend([
      logging.FileHandler('/tmp/databricks-app.log'),  # Dedicated app log
      logging.FileHandler('/tmp/databricks-app-watch.log')  # Also log to watch log
    ])
  
  # Configure root logger
  logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=handlers,
    force=True  # Override any existing configuration
  )
  
  # Configure specific loggers
  logger = logging.getLogger(__name__)
  logger.info(f"Logging configured at {log_level} level - output to console and files")
  
  # Databricks SDK can be noisy, set to WARNING
  logging.getLogger('databricks').setLevel(logging.WARNING)
  logging.getLogger('httpx').setLevel(logging.WARNING)
  
  return logger


# Set up logging
logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
  """Manage application lifespan."""
  logger.info("Starting Databricks App API")
  yield
  logger.info("Shutting down Databricks App API")


app = FastAPI(
  title='Databricks App API',
  description='Modern FastAPI application template for Databricks Apps with React frontend',
  version='0.1.0',
  lifespan=lifespan,
)

# Exception handling middleware
@app.middleware("http")
async def catch_exceptions(request: Request, call_next):
  """Catch and log all unhandled exceptions."""
  try:
    response = await call_next(request)
    return response
  except Exception as e:
    logger.error(f"Unhandled exception for {request.method} {request.url}: {str(e)}")
    logger.error(f"Exception details: {traceback.format_exc()}")
    return JSONResponse(
      status_code=500,
      content={
        "error": "Internal server error",
        "detail": str(e) if os.getenv('DEBUG') == 'true' else "An unexpected error occurred"
      }
    )


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
  """Log all requests and responses."""
  start_time = None
  try:
    import time
    start_time = time.time()
    logger.info(f"Request: {request.method} {request.url}")
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(f"Response: {request.method} {request.url} - {response.status_code} - {process_time:.3f}s")
    return response
  except Exception as e:
    process_time = time.time() - start_time if start_time else 0
    logger.error(f"Request failed: {request.method} {request.url} - {process_time:.3f}s - {str(e)}")
    raise


app.add_middleware(
  CORSMiddleware,
  allow_origins=['http://localhost:3000', 'http://127.0.0.1:3000'],
  allow_credentials=True,
  allow_methods=['*'],
  allow_headers=['*'],
)

app.include_router(router, prefix='/api', tags=['api'])


@app.get('/health')
async def health():
  """Health check endpoint."""
  return {'status': 'healthy'}


# ============================================================================
# SERVE STATIC FILES FROM CLIENT BUILD DIRECTORY (MUST BE LAST!)
# ============================================================================
# This static file mount MUST be the last route registered!
# It catches all unmatched requests and serves the React app.
# Any routes added after this will be unreachable!
if os.path.exists('client/build'):
  app.mount('/', StaticFiles(directory='client/build', html=True), name='static')
