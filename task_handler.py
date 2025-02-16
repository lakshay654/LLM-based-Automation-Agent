import os
import json
import glob
import re
import subprocess
import sqlite3
from datetime import datetime
from sentence_transformers import util
import numpy as np
import pytesseract
from PIL import Image
import urllib.request
import zipfile
import requests
import logging
import torch
import duckdb
from bs4 import BeautifulSoup
import markdown
import csv
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import speech_recognition as sr
from pydub import AudioSegment
import httpx
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()



# Set up the OpenAI API URL and key
openai_api_base = os.getenv("OPENAI_API_CHAT", "https://aiproxy.sanand.workers.dev/openai/v1")
openai_api_key = os.getenv("AIPROXY_TOKEN")
if not openai_api_key:
    raise Exception("AIPROXY_TOKEN environment variable not set.")

headers = {
    "Authorization": f"Bearer {openai_api_key}",
    "Content-Type": "application/json",
}

# List of valid function names
VALID_FUNCTIONS = [
    'task_a1', 'task_a2', 'task_a3', 'task_a4', 'task_a5',
    'task_a6', 'task_a7', 'task_a8', 'task_a9', 'task_a10',
    'task_b1', 'task_b2', 'task_b3', 'task_b4', 'task_b5',
    'task_b6', 'task_b7', 'task_b8', 'task_b9', 'task_b10'
]


def call_llm(prompt: str):
    """
    Calls the OpenAI API with the given prompt and returns the generated response.
    """
    logging.debug(f"Sending prompt to LLM: {prompt}")

    openai_api_base = os.getenv("OPENAI_API_CHAT", "https://aiproxy.sanand.workers.dev/openai/v1")
    openai_api_key = os.getenv("AIPROXY_TOKEN")
    if not openai_api_key:
        raise Exception("AIPROXY_TOKEN environment variable not set.")

    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json",
    }

    try:
        with httpx.Client(timeout=20) as client:
            response = client.post(
                f"{openai_api_base}/chat/completions",
                headers=headers,
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": "You are a function classifier that extracts structured parameters from queries."},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 600,
                    "temperature": 0.6,
                    "n": 1,
                    "stop": None
                },
            )
        logging.debug(f"OpenAI response: {response.json()}")
        content = response.json()["choices"][0]["message"]["content"].strip()

        # Extract JSON content from the code block
        if content.startswith("```json") and content.endswith("```"):
            content = content[7:-3].strip()
        return content
    except Exception as e:
        logging.error(f"An error occurred while communicating with the LLM API: {e}")
        raise Exception(f"An error occurred while communicating with the LLM API: {e}")


def parse_task_with_llm(task_description, max_retries=2):
    """
    Parses the given task description using GPT-4o-Mini to determine the corresponding
    task function name and parameters. Retries up to `max_retries` times if an invalid function
    name is returned.
    """
    # Construct the base prompt with function definitions
    function_definitions_llm = [
        {
            "name": "task_a1",
            "description": "Install uv (if required) and run script with email as the only argument.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {"type": "string", "pattern": r"[\w\.-]+@[\w\.-]+\.\w+"},
                    "url": {"type": "string", "pattern": r"https://api.example.com/users", "default": "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/refs/heads/tds-2025-01/project-1/datagen.py"}
                },
                "required": ["email", "url"]
            }
        },
        {
            "name": "task_a2",
            "description": "Format the contents of /data/format.md using prettier@3.4.2, updating the file in-place.",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string", "pattern": r".*/(.*\.md)"},
                    "prettier_version": {"type": "string", "pattern": r"prettier@\d+\.\d+\.\d+"}
                },
                "required": ["input_file", "prettier_version"]
            }
        },
        {
            "name": "task_a3",
            "description": "Count the number of Wednesdays in /data/dates.txt and write the count to /data/dates-wednesdays.txt.",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string", "pattern": r"/data/.*dates.*\.txt"},
                    "output_file": {"type": "string", "pattern": r"/data/.*/(.*\.txt)"},
                    "day_of_week": {"type": "string", "pattern": r"(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)"}
                },
                "required": ["input_file", "output_file", "day_of_week"]
            }
        },
        {
            "name": "task_a4",
            "description": "Sort the array of contacts in /data/contacts.json by last_name, then first_name, and write the result to /data/contacts-sorted.json.",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {
                        "type": "string",
                        "pattern": r".*/(.*\.json)",
                    },
                    "output_file": {
                        "type": "string",
                        "pattern": r".*/(.*\.json)",
                    }
                },
                "required": ["input_file", "output_file"]
            }
        },
        {
            "name": "task_a5",
            "description": "Write the first line of the 10 most recent .log files in /data/logs/ to /data/logs-recent.txt, most recent first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "log_dir_path": {
                        "type": "string",
                        "pattern": r".*/logs",
                        "default": "/data/logs"
                    },
                    "output_file": {
                        "type": "string",
                        "pattern": r".*/(.*\.txt)",
                        "default": "/data/logs-recent.txt"
                    },
                    "num_files": {
                        "type": "integer",
                        "minimum": 1,
                        "default": 10
                    }
                },
                "required": ["log_dir_path", "output_file", "num_files"]
            }
        },
        {
            "name": "task_a6",
            "description": "Create an index file /data/docs/index.json mapping filenames to titles extracted from Markdown files in /data/docs/.",
            "parameters": {
                "type": "object",
                "properties": {
                    "docs_dir": {
                        "type": "string",
                        "pattern": r".*/docs",
                        "default": "/data/docs"
                    },
                    "output_file": {
                        "type": "string",
                        "pattern": r".*/(.*\.json)",
                        "default": "/data/docs/index.json"
                    }
                },
                "required": ["docs_dir", "output_file"]
            }
        },
        {
            "name": "task_a7",
            "description": "Extract the sender's email address from /data/email.txt using an LLM and write it to /data/email-sender.txt.",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {
                        "type": "string",
                        "pattern": r".*/(.*\.txt)",
                        "default": "/data/email.txt"
                    },
                    "output_file": {
                        "type": "string",
                        "pattern": r".*/(.*\.txt)",
                        "default": "/data/email-sender.txt"
                    }
                },
                "required": ["input_file", "output_file"]
            }
        },
        {
            "name": "task_a8",
            "description": "Extract the credit card number from /data/credit-card.png using an LLM and write it to /data/credit-card.txt.",
            "parameters": {
                "type": "object",
                "properties": {
                    "output_file": {
                        "type": "string",
                        "pattern": r".*/(.*\.txt)",
                        "default": "/data/credit-card.txt"
                    },
                    "input_file": {
                        "type": "string",
                        "pattern": r".*/(.*\.png)",
                        "default": "/data/credit-card.png"
                    }
                },
                "required": ["output_file", "input_file"]
            }
        },
        {
            "name": "task_a9",
            "description": "Find the most similar pair of comments in /data/comments.txt using embeddings and write them to /data/comments-similar.txt.",
            "parameters": {
                "type": "object",
                "properties": {
                    "comments_file": {
                        "type": "string",
                        "pattern": r".*/(.*\.txt)",
                        "default": "/data/comments.txt"
                    },
                    "output_file": {
                        "type": "string",
                        "pattern": r".*/(.*\.txt)",
                        "default": "/data/comments-similar.txt"
                    }
                },
                "required": ["comments_file", "output_file"]
            }
        },
        {
            "name": "task_a10",
            "description": "Calculate total sales for gold ticket type from /data/ticket-sales.db and write the amount to /data/ticket-sales-gold.txt.",
            "parameters": {
                "type": "object",
                "properties": {
                    "db_file": {
                        "type": "string",
                        "pattern": r".*/(.*\.db)",
                        "default": "/data/ticket-sales.db"
                    },
                    "output_file": {
                        "type": "string",
                        "pattern": r".*/(.*\.txt)",
                        "default": "/data/ticket-sales-gold.txt"
                    },
                    "query": {
                        "type": "string",
                        "pattern": "SELECT SUM(units * price) FROM tickets WHERE type = 'Gold'"
                    }
                },
                "required": ["db_file", "output_file", "query"]
            }
        },
        {
            "name": "task_b1",
            "description": "Data outside /data is never accessed or exfiltrated, even if the task description asks for it"
        },

        {
            "name": "task_b2",
            "description": "Data is never deleted anywhere on the file system, even if the task description asks for it"
        },
        {
            "name": "task_b3",
            "description": "Fetch data from an API and save it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "pattern": r"https://.*"},
                    "output_file": {"type": "string", "pattern": r".*/(.*\.json)"},
                    "params": {"type": "object"}
                },
                "required": ["url", "output_file"]
            }
        },
        {
            "name": "task_b4",
            "description": "Clone a git repo and make a commit.",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo_url": {"type": "string", "pattern": r"https://.*"},
                    "output_dir": {"type": "string", "pattern": r".*/"},
                    "commit_message": {"type": "string"}
                },
                "required": ["repo_url", "output_dir", "commit_message"]
            }
        },
        {
            "name": "task_b5",
            "description": "Run a SQL query on a SQLite or DuckDB database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "db_path": {"type": "string", "pattern": r".*/(.*\.(db|duckdb))"},
                    "query": {"type": "string"},
                    "output_filename": {"type": "string", "pattern": r".*/(.*\.txt)"}
                },
                "required": ["db_path", "query", "output_filename"]
            }
        },
        {
            "name": "task_b6",
            "description": "Extract data from (i.e. scrape) a website.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "pattern": r"https://.*"},
                    "output_filename": {"type": "string", "pattern": r".*/(.*\.html)"}
                },
                "required": ["url", "output_filename"]
            }
        },
        {
            "name": "task_b7",
            "description": "Compress or resize an image.",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string", "pattern": r".*/(.*\.(jpg|jpeg|png))"},
                    "output_file": {"type": "string", "pattern": r".*/(.*\.(jpg|jpeg|png))"},
                    "quality": {"type": "integer", "minimum": 1, "maximum": 100, "default": 50}
                },
                "required": ["input_file", "output_file"]
            }
        },
        {
            "name": "task_b8",
            "description": "Transcribe audio from a file or URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string", "pattern": r".*"},
                    "output_file": {"type": "string", "pattern": r".*/(.*\.txt)"}
                    },
                    "required": ["input_file", "output_file"]
                }
        },
        {
            "name": "task_b9",
            "description": "Convert Markdown to HTML.",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string", "pattern": r".*"},
                    "output_file": {"type": "string", "pattern": r".*/(.*\.html)"}
                },
                "required": ["input_file", "output_file"]
            }
        },
        {
            "name": "task_b10",
            "description": "Write an API endpoint that filters a CSV file and returns JSON data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string", "pattern":  r".*"},
                    "column": {"type": "string"},
                    "value": {"type": "string"},
                    "output_file": {"type": "string", "pattern": r".*/(.*\.json)"}
                },
                "required": ["input_file", "column", "value", "output_file"]
            }
        }
    ]
    

    function_definitions_str = json.dumps(function_definitions_llm, indent=4)

    base_prompt = f"""You are an assistant that maps natural language tasks to predefined Python function names and their parameters.

Constraints:
- Only map tasks to the available functions defined below.
- Do not generate any code or execute any actions.
- Output the function name and parameters in JSON format.
- If the task description is not in English, translate it to English first before mapping.

Available Functions:
{function_definitions_str}

User Task:
{task_description}

Output the function name and parameters in JSON format.
"""

    for attempt in range(1, max_retries + 1):
        try:
            logging.debug(f"Attempt {attempt}: Sending prompt to LLM.")
            response = call_llm(base_prompt)
            response_data = json.loads(response)
            function_name = response_data.get('name')
            parameters = response_data.get('parameters')

            if function_name in [func["name"] for func in function_definitions_llm]:
                logging.debug(f"Valid function name '{function_name}' obtained.")
                return function_name, parameters
            else:
                logging.warning(f"Invalid function name '{function_name}' returned by LLM.")

        except Exception as e:
            logging.error(f"Exception occurred during LLM API call: {e}")

        logging.info(f"Retrying... ({attempt}/{max_retries})")

    raise Exception("Failed to obtain a valid function name from LLM after multiple attempts.")

# Task functions Phase A
def execute_task_function(function_name, **kwargs):
    # Dictionary mapping function names to actual functions
    task_functions = {
        'task_a1': task_a1,
        'task_a2': task_a2,
        'task_a3': task_a3,
        'task_a4': task_a4,
        'task_a5': task_a5,
        'task_a6': task_a6,
        'task_a7': task_a7,
        'task_a8': task_a8,
        'task_a9': task_a9,
        'task_a10': task_a10,
        'task_b1': task_b1,
        'task_b2': task_b2,
        'task_b3': task_b3,
        'task_b4': task_b4,
        'task_b5': task_b5,
        'task_b6': task_b6,
        'task_b7': task_b7,
        'task_b8': task_b8,
        'task_b9': task_b9,
        'task_b10': task_b10
        # Implement task_b1 to task_b10 as needed
    }

    if function_name not in task_functions:
        raise ValueError(f"Unknown task function: {function}")
    
    # B1: Ensure data outside /data is never accessed or exfiltrated
    for key, value in kwargs.items():
        if isinstance(value, str) and value.startswith('/') and not value.startswith('/data'):
            raise ValueError(f"Access to data outside /data is not allowed: {value}")

    # B2: Ensure data is never deleted anywhere on the file system
    original_remove = os.remove
    original_rmdir = os.rmdir

    def restricted_remove(path):
        raise ValueError("Data deletion is not allowed.")

    def restricted_rmdir(path):
        raise ValueError("Data deletion is not allowed.")

    os.remove = restricted_remove
    os.rmdir = restricted_rmdir

    try:
        # Execute the task function with the provided parameters
        result = task_functions[function_name](**kwargs)
    finally:
        # Restore the original os.remove and os.rmdir functions
        os.remove = original_remove
        os.rmdir = original_rmdir

    return result

# Task A1 Example
def task_a1(email, url='https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/refs/heads/tds-2025-01/project-1/datagen.py'):
    # Install uv if required and run datagen.py with the user's email
    user_email = email
    if not user_email:
        raise Exception("Email parameter not provided.")

    # Check if 'uv' is installed
    try:
        import uv
    except ImportError:
        # Install uv
        subprocess.check_call(['pip', 'install', 'uv'])

    # Run datagen.py
    command = f'python -m uv run {url} {user_email}'
    exit_code = os.system(command)
    if exit_code != 0:
        raise Exception("Failed to run datagen.py")
    return "Task A1 completed successfully."

# Task A2 Function
def task_a2(input_file='data/format.md', prettier_version="prettier@3.4.2"):
    """
    Formats the file using prettier.
    """
    # Ensure Prettier is installed
    try:
        subprocess.check_call(['npm', 'list', prettier_version], shell=True)
    except subprocess.CalledProcessError:
        try:
            subprocess.check_call(['npm', 'install', prettier_version], shell=True)
        except subprocess.CalledProcessError as e:
            raise Exception(f"Failed to install Prettier: {e}")

    # Check if the file exists
    if not os.path.exists(input_file):
        raise Exception(f"File not found: {input_file}")
    
    # Read the current contents of the file.
    with open(input_file, 'r') as f:
        original = f.read()
    
    try:
        # Build the command as a single string.
        cmd = f"npx {prettier_version} --stdin-filepath {input_file}"
        # Run Prettier using the command string, passing the current working directory and environment.
        proc = subprocess.run(
            cmd,
            input=original,
            capture_output=True,
            text=True,
            check=True,
            shell=True,  # Command is provided as a string.
            cwd=os.getcwd(),         # Ensure we run in the project root.
            env=os.environ.copy()      # Pass the current environment.
        )
        formatted = proc.stdout
        
        # Write the formatted content back to the file.
        with open(input_file, 'w') as f:
            f.write(formatted)
        
        return {"stdout": formatted, "stderr": proc.stderr}
    except subprocess.CalledProcessError as e:
        raise Exception(f"Error formatting file: {e.stderr}")
    except FileNotFoundError as e:
        raise Exception(f"File not found error: {e}")

# Task A3 Example
def task_a3(input_file='data/dates.txt', output_file='data/dates-wednesdays.txt', day_of_week='Wednesday'):
    """
    Reads a file, counts the number of specified days of the week,
    and writes the count to the target file.
    """

    # Define a list of possible date formats.
    date_formats = [
        "%Y/%m/%d %H:%M:%S",  # e.g., 2008/04/22 06:26:02
        "%Y-%m-%d",           # e.g., 2006-07-21
        "%b %d, %Y",          # e.g., Sep 11, 2006
        "%d-%b-%Y",           # e.g., 28-Nov-2021
    ]

    # Convert day_of_week to a number (Monday=0, Tuesday=1, ..., Sunday=6)
    try:
        day_of_week_num = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"].index(day_of_week.lower())
    except ValueError:
        raise Exception(f"Invalid day of the week: {day_of_week}")

    day_count = 0

    with open(input_file, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue  # Skip empty lines

            parsed_date = None
            # Try each date format until one succeeds.
            for fmt in date_formats:
                try:
                    parsed_date = datetime.strptime(line, fmt)
                    break  # Exit loop if parsing is successful.
                except ValueError:
                    continue

            if parsed_date is None:
                # Optionally log the unparsable line.
                print(f"Warning: Could not parse date: {line}")
                continue

            # datetime.weekday() returns Monday=0, Tuesday=1, Wednesday=2, etc.
            if parsed_date.weekday() == day_of_week_num:
                day_count += 1

    # Write just the count to the output file.
    with open(output_file, 'w') as file:
        file.write(str(day_count))
    
    if not os.path.exists(output_file):
        raise Exception(f"File not found: {output_file}")
    return {
        "status": "success task a3",
        "day_count": day_count,
        "day_of_week": day_of_week}

# Task A4 Function
def task_a4(input_file = 'data/contacts.json', output_file = 'data/contacts-sorted.json'):
    
    with open(input_file, 'r') as file:
        data = json.load(file)
    sorted_data = sorted(data, key=lambda x: (x.get("last_name", ""), x.get("first_name", "")))
    with open(output_file, 'w') as file:
        json.dump(sorted_data, file, indent=2)
    return {
        "status": "success task a4",
        "written_file": output_file,
        "sorted_data": sorted_data}

def task_a5(log_dir_path = 'data/logs/' , output_file= 'data/logs-recent.txt', num_files=10):
    
    log_files = sorted(
        glob.glob(os.path.join(log_dir_path, "*.log")),
        key=os.path.getmtime,
        reverse=True
    )[:num_files]
    with open(output_file, 'w') as outfile:
        for log_file in log_files:
            with open(log_file, "r") as infile:
                first_line = infile.readline()
                outfile.write(first_line)
    return {
        "status": "success task a5",
        "written_file": output_file, 
        "log_files": log_files}

# Task A6: Extract H1 Titles from Markdown Files
def task_a6(docs_dir = "/data/docs/", output_file = "/data/docs/index.json"):
    
    """
    Find all .md files in /data/docs/, extract the first occurrence of an H1 title (# Title),
    and save them in /data/docs/index.json as { "file.md": "Title", ... }.
    """
    index = {}

    # Walk through /data/docs/ recursively
    for root, _, files in os.walk(docs_dir):
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, docs_dir)
                normalized_path = relative_path.replace(os.sep, '/')  # Use forward slashes

                # Extract the first H1 title from the file
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            match = re.match(r"^# (.+)", line.strip())
                            if match:
                                index[normalized_path] = match.group(1)
                                break  # Stop after first H1
                except Exception as e:
                    index[normalized_path] = f"Error reading file: {str(e)}"

    # Write to index.json
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(index, f, indent=4)

    return {
        "status": "success task a7",
        "written_file": output_file,
        "index": index}

# Task A7: Extract Email Sender Using LLM
def task_a7(input_file = "data/email.txt", output_file = "data/email-sender.txt"):
    
    with open(input_file, 'r') as file:
        email_content = file.read()
    prompt = (
        "You are a helpful assistant. I have an email message:\n\n"
        f"{email_content}\n\n"
        "Please extract only the sender’s email address from this email. "
        "Return your answer in a JSON object with a single key 'sender_email'. For example:\n"
        "{\n  \"sender_email\": \"example@domain.com\"\n}\n\n"
        "Return only the JSON object."
    )
    response = call_llm(prompt)
    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        return {
                "error": "LLM response was not valid JSON.",
                "response": response
            }
    sender_email = data.get("sender_email", "").strip()
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(sender_email)
    return  {
            "status": "success task a7",
            "sender_email": sender_email,
            "written_file": output_file
        }


def parse_task_with_ollama_with_image(image_path, prompt):
    try:
        # Perform OCR on the image
        text_from_image = pytesseract.image_to_string(Image.open(image_path))
        logging.debug(f"Extracted text from image: {text_from_image}")
    except Exception as e:
        logging.error(f"Error during OCR processing: {e}")
        raise Exception(f"Failed to extract text from image: {e}")

    # Combine the OCR result with the prompt
    full_prompt = f"{prompt}\n\n{text_from_image}"

    # Call Ollama
    try:
        response = call_llm(full_prompt).strip()
        return response
    except Exception as e:
        logging.error(f"Error during call to Ollama: {e}")
        raise Exception(f"Failed to call Ollama: {e}")

def ensure_tesseract_installed():
    tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    if not os.path.exists(tesseract_cmd):
        print("Tesseract not found. Downloading and installing...")
        tesseract_url = "https://github.com/UB-Mannheim/tesseract/wiki"
        tesseract_zip_path = os.path.join(os.getcwd(), "tesseract.zip")
        tesseract_extract_path = os.path.join(os.getcwd(), "tesseract")

        # Download Tesseract
        urllib.request.urlretrieve(tesseract_url, tesseract_zip_path)

        # Extract Tesseract
        with zipfile.ZipFile(tesseract_zip_path, 'r') as zip_ref:
            zip_ref.extractall(tesseract_extract_path)

        # Move Tesseract to the desired location
        tesseract_install_path = r'C:\Program Files\Tesseract-OCR'
        if not os.path.exists(tesseract_install_path):
            os.makedirs(tesseract_install_path)
        for item in os.listdir(tesseract_extract_path):
            s = os.path.join(tesseract_extract_path, item)
            d = os.path.join(tesseract_install_path, item)
            if os.path.isdir(s):
                os.rename(s, d)
            else:
                os.rename(s, d)

        # Clean up
        os.remove(tesseract_zip_path)
        os.rmdir(tesseract_extract_path)

        print("Tesseract installed successfully.")

    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

def task_a8(input_file= "data/credit-card.png", output_file= "data/credit-card.txt"):
    ensure_tesseract_installed()
    prompt = (
        "You are a helpful assistant. I have credit card image:\n\n"
        "Please Extracts a 16-digit card number from text from image.\n"
        "Return your answer in a JSON object with a single key 'card_number'. For example:\n"
        "{\n  \"card_number\": \"3254789652457852\"\n}\n\n"
        "Return only the JSON object."
    )
    card_number = parse_task_with_ollama_with_image(input_file, prompt)
    # Attempt to parse JSON
    try:
        data = json.loads(card_number)
    except json.JSONDecodeError:
        return {
                "error": "LLM response was not valid JSON.",
                "raw_response": card_number
            }
    card_number = data.get("card_number", "").strip()
    with open(output_file, 'w') as file:
        file.write(card_number.replace(" ", ""))
    return {
        "status": "success task a8",
        "written_file": output_file, 
        "card_number": card_number}

# Task A9: Find Most Similar Comments Using Embeddings
# Initialize the model
def get_embeddings(texts):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }
    data = {
        "model": "text-embedding-3-small",
        "input": texts
    }
    response = requests.post("http://aiproxy.sanand.workers.dev/openai/v1/embeddings", headers=headers, data=json.dumps(data))
    response.raise_for_status()
    return [item["embedding"] for item in response.json()["data"]]

def task_a9(comments_file="/data/comments.txt", output_file="/data/comments-similar.txt", batch_size=10):
    with open(comments_file, "r") as file:
        comments = [line.strip() for line in file]

    # Get embeddings for each comment in batches
    embeddings = []
    for i in range(0, len(comments), batch_size):
        batch_comments = comments[i:i + batch_size]
        batch_embeddings = get_embeddings(batch_comments)
        embeddings.extend(batch_embeddings)

    # Convert embeddings to tensor
    embeddings_tensor = torch.tensor(embeddings)

    # Calculate similarity matrix
    similarity_matrix = util.pytorch_cos_sim(embeddings_tensor, embeddings_tensor)

    # Convert the similarity matrix to a NumPy array
    similarity_matrix_np = similarity_matrix.cpu().numpy()

    np.fill_diagonal(similarity_matrix_np, -1)  # Exclude self-similarity
    most_similar = np.unravel_index(np.argmax(similarity_matrix_np), similarity_matrix_np.shape)

    with open(output_file, 'w') as file:
        file.write(f"{comments[most_similar[0]]}\n{comments[most_similar[1]]}")

    return {
        "status": "task a9 success",
        "most_similar_comments": [comments[most_similar[0]], comments[most_similar[1]]],
        "written_file": output_file
    }
# Task A10: Calculate Ticket Sales from SQLite DB
def task_a10(db_file="/data/ticket-sales.db", output_file="/data/ticket-sales-gold.txt" , query="SELECT SUM(quantity) FROM sales WHERE ticket_type='Gold'"):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute(query)
    total_sales = cursor.fetchone()[0] or 0
    conn.close()
    with open(output_file, 'w') as file:
        file.write(str(total_sales))
    return {
        "status": "success task a10",
        "total_sales": total_sales, 
        "written_file": output_file}


# Phase B TASKS
# B1: Ensure data outside /data is never accessed or exfiltrated
def task_b1():
    return {"status": "success", "message": "Data outside /data is never accessed or exfiltrated"}

# B2: Ensure data is never deleted anywhere on the file system
def task_b2():
    return {"status": "success", "message": "Data is never deleted anywhere on the file system"}

# B3: Fetch data from an API and save it
def task_b3(url, output_file, params=None):
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        with open(output_file, "w") as file:
            json.dump(data, file, indent=4)
        return {"status": "success", "written_file": output_file, "data": data}
    except requests.exceptions.RequestException as e:
        raise Exception(f"Error fetching data from API: {e}")

# B4: Clone a git repo and make a commit
def task_b4(repo_url, output_dir, commit_message):
    try:
        # Check if the output directory already exists
        if os.path.exists(output_dir):
            # Remove the existing directory
            subprocess.run(["rm", "-rf", output_dir], check=True)
        
        # Clone the repository
        subprocess.run(["git", "clone", repo_url, output_dir], check=True)
        
        # Create a new file to ensure there are changes to commit
        new_file_path = os.path.join(output_dir, "new_file.txt")
        with open(new_file_path, "w") as new_file:
            new_file.write("This is a test file.")
        
        # Add all files to the staging area
        subprocess.run(["git", "add", "."], cwd=output_dir, check=True)
        
        # Commit the changes
        subprocess.run(["git", "commit", "-m", commit_message], cwd=output_dir, check=True)
        
        return {"status": "success", "repo_url": repo_url, "commit_message": commit_message}
    except subprocess.CalledProcessError as e:
        return {"status": "error", "detail": f"An error occurred: {e}"}
    except Exception as e:
        return {"status": "error", "detail": f"An unexpected error occurred: {e}"}
# B5: Run a SQL query on a SQLite or DuckDB database
def task_b5(db_path, query, output_filename):
    try:
        conn = sqlite3.connect(db_path) if db_path.endswith('.db') else duckdb.connect(db_path)
        cur = conn.cursor()
        cur.execute(query)
        result = cur.fetchall()
        conn.close()
        with open(output_filename, 'w') as file:
            file.write(str(result))
        return {"status": "success", "written_file": output_filename, "result": result}
    except Exception as e:
        raise Exception(f"Error running SQL query: {e}")

# B6: Extract data from (i.e. scrape) a website
def task_b6(url, output_filename):
    try:
        # Set up Selenium with Chrome WebDriver
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
        
        # Fetch the website content
        driver.get(url)
        html_content = driver.page_source
        driver.quit()
        
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Save the prettified HTML content to the output file
        with open(output_filename, "w", encoding="utf-8") as file:
            file.write(soup.prettify())
        
        return {"status": "success", "written_file": output_filename}
    except Exception as e:
        raise Exception(f"Error scraping website with JavaScript: {e}")


# B7: Compress or resize an image
def task_b7(input_file, output_file, quality=50):
    try:
        # Open the input image file
        img = Image.open(input_file)
        
        # Convert the image to RGB mode if it is a PNG (to avoid issues with transparency)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        
        # Determine the output format based on the output file extension
        output_format = output_file.split('.')[-1].upper()
        
        # Save the image with the specified quality to the output file
        if output_format == "JPEG" or output_format == "JPG":
            img.save(output_file, "JPEG", quality=quality)
        elif output_format == "PNG":
            img.save(output_file, "PNG")
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        return {"status": "success", "written_file": output_file}
    except Exception as e:
        raise Exception(f"Error compressing or converting image: {e}")

# B8: Transcribe audio from an MP3 file

def task_b8(input_file, output_file):
    try:
        # Check if the input is a URL or a local file path
        if input_file.startswith("http://") or input_file.startswith("https://"):
            # Download the file from the URL
            response = requests.get(input_file)
            response.raise_for_status()
            local_file_path = os.path.join(os.getcwd(), "temp_audio_file.mp3")
            with open(local_file_path, "wb") as file:
                file.write(response.content)
        else:
            local_file_path = input_file

        # Check if the local file path exists
        if not os.path.exists(local_file_path):
            raise FileNotFoundError(f"The file {local_file_path} does not exist.")

        # Convert the audio file to WAV format
        audio = AudioSegment.from_file(local_file_path)
        wav_file = local_file_path.replace(os.path.splitext(local_file_path)[-1], ".wav")
        audio.export(wav_file, format="wav")

        # Check if the WAV file was created
        if not os.path.exists(wav_file):
            raise FileNotFoundError(f"The WAV file {wav_file} was not created.")

        # Initialize recognizer
        recognizer = sr.Recognizer()

        # Load the audio file
        with sr.AudioFile(wav_file) as source:
            audio_data = recognizer.record(source)

        # Perform transcription
        transcript = recognizer.recognize_google(audio_data)

        # Save the transcription to the output file
        with open(output_file, "w") as file:
            file.write(transcript)

        return {"status": "success", "written_file": output_file, "transcript": transcript}
    except Exception as e:
        raise Exception(f"Error transcribing audio: {e}")

# B9: Convert Markdown to HTML
def task_b9(input_file, output_file):
    try:
        # Check if the input is a URL or a local file path
        if input_file.startswith("http://") or input_file.startswith("https://"):
            # Download the file from the URL
            response = requests.get(input_file)
            response.raise_for_status()
            local_file_path = os.path.join(os.getcwd(), "temp_markdown_file.md")
            with open(local_file_path, "wb") as file:
                file.write(response.content)
        else:
            local_file_path = input_file

        # Convert Markdown to HTML
        with open(local_file_path, "r") as file:
            html = markdown.markdown(file.read())
        with open(output_file, "w") as file:
            file.write(html)
        return {"status": "success", "written_file": output_file}
    except Exception as e:
        raise Exception(f"Error converting Markdown to HTML: {e}")

# B10: Write an API endpoint that filters a CSV file and returns JSON data
def task_b10(input_file, column, value, output_file):
    try:
        # Check if the input is a URL or a local file path
        if input_file.startswith("http://") or input_file.startswith("https://"):
            # Download the file from the URL
            response = requests.get(input_file)
            response.raise_for_status()
            local_file_path = os.path.join(os.getcwd(), "temp_csv_file.csv")
            with open(local_file_path, "wb") as file:
                file.write(response.content)
        else:
            local_file_path = input_file

        # Filter the CSV file
        results = []
        with open(local_file_path, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row[column] == value:
                    results.append(row)
        with open(output_file, "w") as file:
            json.dump(results, file)
        return {"status": "success", "written_file": output_file, "results": results}
    except Exception as e:
        raise Exception(f"Error filtering CSV: {e}")

