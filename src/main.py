import asyncio
import os
import logging
import argparse
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI
import pdfplumber

# -----------------------------
# Configuration
# -----------------------------
# Load environment variables
# load_dotenv(override=True) # Removed duplicate load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("NDAAgent")

# Load environment variables once at the start
load_dotenv()

# Initialize OpenAI clients
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)
async_client = AsyncOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

# Vector Store Configuration
DEFAULT_VECTOR_STORE_ID = os.environ.get("DEFAULT_VECTOR_STORE_ID")
if not DEFAULT_VECTOR_STORE_ID:
    logger.error("DEFAULT_VECTOR_STORE_ID not found in environment variables. Please set it in the .env file.")
    exit(1)
else:
    logger.info(f"Default Vector Store ID loaded: {DEFAULT_VECTOR_STORE_ID}")

# -----------------------------
# Helper Functions
# -----------------------------
def get_user_vector_store_id(user_id: Optional[str]) -> Optional[str]:
    """
    Return a user-specific Vector Store ID if available.
    In a real app, this would query a database based on user context.
    
    Args:
        user_id: The ID of the user to lookup
        
    Returns:
        The user's vector store ID or None if not found
    """
    # Example logic - replace with actual database lookup
    if user_id == "user_with_custom_playbook":
        return "vs_user_specific_id_example"
    return None

def load_contract_file(filepath: str) -> Optional[str]:
    """Load the contract text from a file, supporting both text and PDF formats.

    Uses pdfplumber for PDF extraction and standard file reading for text files.
    Logs errors if the file is not found or cannot be read.

    Args:
        filepath: Path to the contract file (.txt, .md, .pdf).

    Returns:
        The extracted text content of the contract as a string, or None if an error occurs.
    """
    try:
        logger.info(f"Reading contract file: {filepath}")
        
        if filepath.lower().endswith('.pdf'):
            # Handle PDF files
            text = ""
            logger.info("Detected PDF file, using pdfplumber for extraction")
            with pdfplumber.open(filepath) as pdf:
                logger.info(f"PDF loaded successfully. Number of pages: {len(pdf.pages)}")
                for i, page in enumerate(pdf.pages):
                    page_text = page.extract_text() or ""
                    logger.info(f"Extracted text from page {i+1}: {len(page_text)} characters")
                    text += page_text + "\n"
            contract_text = text
            logger.info(f"Total extracted text: {len(contract_text)} characters")
        else:
            # Handle text files
            with open(filepath, 'r', encoding='utf-8') as f:
                contract_text = f.read()
        
        if not contract_text or contract_text.strip() == "":
            logger.error(f"Contract file '{filepath}' is empty or could not be read properly.")
            return None
            
        logger.info(f"Contract loaded successfully ({len(contract_text)} characters).")
        return contract_text
    except FileNotFoundError:
        logger.error(f"Contract file not found: {filepath}")
        return None
    except Exception as e:
        logger.error(f"Error reading contract file {filepath}: {e}")
        return None

# -----------------------------
# Prompt Templates
# -----------------------------
NDA_SUMMARIZER_INSTRUCTIONS = (
    "Summarize the provided NDA contract text in plain English. "
    "Highlight parties, duration, confidentiality obligations, non-compete terms "
    "(scope and duration), and governing law."
)

CLAUSE_BREAKER_INSTRUCTIONS = (
    "Analyze the provided NDA contract text and identify distinct legal clauses "
    "(e.g., confidentiality, non-compete, governing law). "
    "Output ONLY the text of the clauses, separated by the exact delimiter '|||'. "
    "Do NOT include any other text, explanations, numbering, or formatting. "
    "Example output format: Clause 1 text.|||Clause 2 text.|||Clause 3 text."
)

NDA_REDLINER_INSTRUCTIONS = (
    "You are an expert legal assistant reviewing a single clause from an NDA. Your task "
    "is to provide redlines based on standard and potentially user-specific fallback "
    "guidance.\n\n"
    "1. Use the file_search tool to find relevant fallback clauses and risk notes for the input clause. "
    "   Search across all provided vector stores. Note if guidance comes from a user-specific store "
    "   vs. a default store based on which vector store ID the result came from.\n"
    "2. Prioritize any fallback guidance that seems specific or customized (from a user store) "
    "   over standard guidance (from the default store) if both are found and relevant.\n"
    "3. Based on the *prioritized* fallback guidance (or standard guidance if no custom "
    "   guidance is found), redline the *original* input clause using markdown: use "
    "   ~~strike-through~~ for text to be removed and **bold** for text to be added.\n"
    "4. If no relevant fallback guidance is found in any store, state that clearly.\n"
    "5. Append a brief comment (`<!-- comment -->`) explaining the reason for the changes "
    "   based on the prioritized risk notes/guidance, or stating why no changes were "
    "   needed/found. Mention if the redline is based on standard vs. potentially custom "
    "   guidance if you inferred a difference.\n"
    "6. Return ONLY the redlined clause and the comment as a single string."
)

# -----------------------------
# Main Analysis Function
# -----------------------------
async def run_analysis(contract_filepath: str, user_id: Optional[str] = None):
    """Executes the full NDA contract analysis workflow.

    This is the main asynchronous function that orchestrates the process:
    1. Loads the contract text from the specified file path.
    2. Determines the Vector Store IDs to use (default + optional user-specific).
    3. Runs the summarizer agent (gpt-4o).
    4. Runs the clause breaker agent (gpt-4o).
    5. Runs the redliner agent (gpt-4o) sequentially for each clause using file_search.
    6. Aggregates the results (summary + redlined clauses) and prints the final output.

    Args:
        contract_filepath: Path to the contract file (.txt, .md, .pdf).
        user_id: Optional identifier for the user. Currently not used for loading
                 different vector stores but included for future extension.
    """
    # Load the contract
    nda_contract = load_contract_file(contract_filepath)
    if not nda_contract:
        return

    logger.info(f"Starting analysis for user: {user_id or 'Default'} using contract: {contract_filepath}")

    # Determine which Vector Store(s) to use
    target_vs_ids = [DEFAULT_VECTOR_STORE_ID]  # Always include the default store
    user_vs_id = get_user_vector_store_id(user_id)
    
    if user_vs_id:
        logger.info(f"Found user-specific Vector Store: {user_vs_id}")
        if user_vs_id not in target_vs_ids:  # Avoid duplicates
            target_vs_ids.append(user_vs_id)
    else:
        logger.info(f"No user-specific Vector Store found, using default only.")
    
    logger.info(f"Using Vector Store IDs for search: {target_vs_ids}")

    summary = "Error: Summary not generated."
    redlined_clauses = []

    try:
        # Step A: Summarize
        logger.info("Running summarizer...")
        logger.info("Attempting to call OpenAI API for summarization...")
        summarizer_response = await async_client.responses.create(
            model="gpt-4o",
            input=nda_contract,
            instructions=NDA_SUMMARIZER_INSTRUCTIONS
        )
        logger.info("OpenAI API call for summarization returned.")
        
        # Find the message content within the output list (robust handling)
        summary_text = None
        if (
            summarizer_response
            and hasattr(summarizer_response, "output")
            and summarizer_response.output
        ):
            for output_item in summarizer_response.output:
                if (
                    hasattr(output_item, "type") and output_item.type == "message"
                    and hasattr(output_item, "content") and output_item.content
                    and hasattr(output_item.content[0], "text")
                ):
                    summary_text = output_item.content[0].text
                    break

        if summary_text:
            summary = summary_text
            logger.info("Summarizer finished.")
        else:
            logger.error("Summarizer did not return expected output structure.")

        # Step B: Break Clauses
        logger.info("Running clause breaker...")
        clauses = []
        logger.info("Attempting to call OpenAI API for clause breaking...")
        breaker_response = await async_client.responses.create(
            model="gpt-4o",
            input=nda_contract,
            instructions=CLAUSE_BREAKER_INSTRUCTIONS
        )
        logger.info("OpenAI API call for clause breaking returned.")
        
        # Find the message content within the output list (robust handling)
        clauses_text = None
        if (
            breaker_response
            and hasattr(breaker_response, "output")
            and breaker_response.output
        ):
            for output_item in breaker_response.output:
                 if (
                    hasattr(output_item, "type") and output_item.type == "message"
                    and hasattr(output_item, "content") and output_item.content
                    and hasattr(output_item.content[0], "text")
                ):
                    clauses_text = output_item.content[0].text
                    break

        if clauses_text:
            # Split the output string by the delimiter
            clauses = clauses_text.split('|||')
            # Remove leading/trailing whitespace from each clause
            clauses = [c.strip() for c in clauses if c.strip()]
            logger.info(f"Clause breaker finished. Found {len(clauses)} clauses.")
        else:
            logger.error(f"Clause breaker did not return a valid output structure. Received: {breaker_response}")
            clauses = []  # Ensure clauses is an empty list if breaker failed

        # Step C: Redline Each Clause
        if clauses:
            logger.info("Running redliner for each clause...")
            redlined_clauses = await process_clauses_with_redliner(clauses, target_vs_ids)
        else:
            logger.warning("Skipping redlining step because clause breaking failed or returned no clauses.")

        # Step D: Aggregate Results
        final_output = format_final_output(summary, redlined_clauses)
        logger.info("Aggregation complete.")
        
        # Determine output directory and filename
        output_dir = "output_analysis"
        os.makedirs(output_dir, exist_ok=True) # Create output dir if it doesn't exist
        
        base_filename = os.path.basename(contract_filepath)
        name_part, _ = os.path.splitext(base_filename)
        output_filename_base = f"{name_part}_analysis.md"
        output_filepath = os.path.join(output_dir, output_filename_base)
        
        # Write final output to file
        try:
            with open(output_filepath, 'w', encoding='utf-8') as f:
                f.write(final_output)
            logger.info(f"Analysis complete. Output saved to: {output_filepath}")
            print(f"\nAnalysis complete. Output saved to: {output_filepath}")
        except IOError as e:
            logger.error(f"Failed to write output file {output_filepath}: {e}")
            # Optionally print to console as fallback if file writing fails
            print("\n=== Final Aggregated Output (File write failed) ===")
            print(final_output)

    except Exception as e:
        logger.exception("A critical error occurred during the workflow: %s", str(e))
        print(f"\n=== Error: A critical exception occurred during processing: {e} ===")

async def process_clauses_with_redliner(clauses: List[str], vector_store_ids: List[str]) -> List[str]:
    """Processes each clause with the redliner agent sequentially.

    Uses an asyncio.Semaphore with a limit of 1 to process clauses one by one,
    preventing potential rate limit issues associated with concurrent file_search calls.

    Args:
        clauses: A list of strings, where each string is a clause from the NDA.
        vector_store_ids: A list containing the ID(s) of the OpenAI Vector Store(s)
                          to be used for the file_search tool during redlining.

    Returns:
        A list of strings, where each string is the redlined version of the
        corresponding input clause, or an error message comment if redlining failed.
    """
    semaphore = asyncio.Semaphore(1) # Limit to 1 concurrent task (sequential)
    redline_tasks = []
    
    async def semaphored_redline_task(clause_text: str, vs_ids: List[str]) -> str:
        async with semaphore:
            logger.debug(f"Acquired semaphore for clause: {clause_text[:50]}...")
            result = await redline_clause(clause_text, vs_ids)
            logger.debug(f"Released semaphore for clause: {clause_text[:50]}...")
            return result
            
    # Create tasks wrapped with the semaphore
    for i, clause_text in enumerate(clauses):
        logger.info(f"Dispatching redliner task {i+1}/{len(clauses)} for clause: {clause_text[:50]}...")
        task = semaphored_redline_task(clause_text, vector_store_ids)
        redline_tasks.append(task)
    
    # Run redlining concurrently (respecting semaphore limit)
    redliner_results = await asyncio.gather(*redline_tasks, return_exceptions=True)
    
    # Process results
    logger.info("Redliner tasks completed. Processing results...")
    redlined_clauses = []
    
    for i, result in enumerate(redliner_results):
        original_clause_preview = clauses[i][:50]  # For logging
        
        if isinstance(result, Exception):
            logger.error(f"Redliner failed for clause {i+1} ('{original_clause_preview}...'): {result}")
            redlined_clauses.append(f"<!-- Error redlining clause: {clauses[i]} \n Exception: {result} -->")
        elif result and isinstance(result, str):
            redlined_clauses.append(result)
            logger.info(f"Redliner succeeded for clause {i+1} ('{original_clause_preview}...')")
        else:
            logger.error(f"Redliner returned unexpected format for clause {i+1} ('{original_clause_preview}...'): {result}")
            redlined_clauses.append(f"<!-- Error: Redliner returned invalid format for clause: {clauses[i]} -->")


    return redlined_clauses

async def redline_clause(clause_text: str, vector_store_ids: List[str]) -> str:
    """Redlines a single clause using the OpenAI API with file_search.

    Calls the OpenAI API (gpt-4o) with specific instructions (NDA_REDLINER_INSTRUCTIONS)
    and configures the file_search tool to use the provided Vector Store IDs.
    Handles the response structure, extracting the redlined text message.

    Args:
        clause_text: The text of the single clause to be redlined.
        vector_store_ids: A list containing the Vector Store ID(s) to search against.

    Returns:
        A string containing the redlined clause and explanatory comment, or an
        error message comment if the process failed.
    """
    try:
        # Configure file_search tool for each vector store
        file_search_tools = []
        
        for vs_id in vector_store_ids:
            # Corrected tool structure: vector_store_ids is top-level
            file_search_tools.append({
                "type": "file_search",
                "vector_store_ids": [vs_id] # Corrected structure
            })
            
        # Create a user message with just the clause
        # user_message = f"Please analyze and redline this clause from an NDA:\n\n{clause_text}"
        
        # Get redlining from the model using the file_search tool
        logger.debug(f"Calling redliner API for clause: {clause_text[:50]}...")
        redliner_response = await async_client.responses.create(
            model="gpt-4o",
            input=clause_text,
            instructions=NDA_REDLINER_INSTRUCTIONS,
            tools=file_search_tools
        )
        logger.debug(f"Redliner API call returned for clause: {clause_text[:50]}...")
        
        # Find the message content within the output list
        redlined_text = None
        if redliner_response and hasattr(redliner_response, "output") and redliner_response.output:
            for output_item in redliner_response.output:
                # Check if this item is the message containing the text
                if (
                    hasattr(output_item, "type") and output_item.type == "message"
                    and hasattr(output_item, "content") and output_item.content
                    and hasattr(output_item.content[0], "text")
                ):
                    redlined_text = output_item.content[0].text
                    break # Found the text, stop searching

        if redlined_text:
            return redlined_text
        else:
            logger.error(f"Redliner failed to find text content in expected structure for clause: {clause_text[:50]}... Response: {redliner_response}")
            return f"<!-- Error: Failed to extract redlining text from response for clause: {clause_text} -->"
        
    except Exception as e:
        logger.error(f"Error in redline_clause: {e}")
        return f"<!-- Error redlining clause: {clause_text} \n Exception: {e} -->"

def format_final_output(summary: str, redlined_clauses: List[str]) -> str:
    """Formats the final output string containing the summary and redlined clauses.

    Args:
        summary: The contract summary string generated by the summarizer agent.
        redlined_clauses: A list of strings, each containing a redlined clause
                          (or an error message comment).

    Returns:
        A single formatted string ready for printing to the console.
    """
    return (
        f"=== NDA Contract Summary ===\n{summary}\n\n"
        f"=== Redlined Clauses ({len(redlined_clauses)} processed) ===\n"
        f"{'\n\n'.join(redlined_clauses) if redlined_clauses else 'No clauses were redlined.'}"
    )

# -----------------------------
# Main Entry Point
# -----------------------------
if __name__ == "__main__":
    # Setup command-line argument parsing
    parser = argparse.ArgumentParser(description="Analyze an NDA contract using AI.")
    parser.add_argument("contract_file", help="Path to the NDA contract file to analyze.")
    # Add an optional argument for simulating a user context
    parser.add_argument("--user", help="Optional user ID to simulate user-specific playbook (e.g., 'user_with_custom_playbook')", default=None)
    args = parser.parse_args()

    # Ensure the Default Vector Store ID is loaded before running main
    if not DEFAULT_VECTOR_STORE_ID:
        print("Error: DEFAULT_VECTOR_STORE_ID environment variable not set. Exiting.")
    else:
        asyncio.run(run_analysis(args.contract_file, args.user))



