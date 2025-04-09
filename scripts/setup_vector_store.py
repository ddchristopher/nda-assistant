#!/usr/bin/env python
"""Helper script to create an OpenAI Vector Store and upload a playbook file.

This script automates the setup process required before running the main
NDA analysis script (src/main.py).

Usage:
    python scripts/setup_vector_store.py <path_to_playbook_file>

Example:
    python scripts/setup_vector_store.py default_docs/Default_NDA_Negotiation_Playbook.md
"""

import os
import sys
import time
import argparse
import logging
from openai import OpenAI
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("VectorStoreSetup")

# --- Main Setup Function ---

def setup_store(playbook_filepath: str):
    """Creates a Vector Store, uploads the playbook file, and adds it to the store."""
    logger.info("Starting Vector Store setup process...")

    # 1. Load Environment Variables and Initialize Client
    logger.info("Loading environment variables...")
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment variables (.env file).")
        sys.exit(1)

    try:
        client = OpenAI(
            api_key=api_key
        )
        logger.info("OpenAI client initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        sys.exit(1)

    # 2. Check if Playbook File Exists
    if not os.path.isfile(playbook_filepath):
        logger.error(f"Playbook file not found at: {playbook_filepath}")
        sys.exit(1)
    logger.info(f"Playbook file found: {playbook_filepath}")

    vector_store_id = None
    try:
        # 3. Upload the Playbook File
        logger.info("Uploading playbook file to OpenAI...")
        with open(playbook_filepath, "rb") as file_stream:
            file_object = client.files.create(file=file_stream, purpose="assistants") # Use 'assistants' purpose
        logger.info(f"File uploaded successfully. File ID: {file_object.id}")

        # Optional: Wait briefly for file processing if needed, though usually handled by vector store creation
        # time.sleep(5) # Uncomment if experiencing issues with file not being ready

        # 4. Create the Vector Store
        logger.info("Creating new Vector Store...")
        vector_store = client.beta.vector_stores.create(
            name=f"NDA Playbook Store ({os.path.basename(playbook_filepath)})",
            file_ids=[file_object.id]
        )
        vector_store_id = vector_store.id
        logger.info(f"Vector Store created successfully. Vector Store ID: {vector_store_id}")

        # 5. Poll until the file is processed within the vector store
        #    (Important step as adding file doesn't mean it's immediately ready)
        logger.info(f"Waiting for file {file_object.id} to be processed in Vector Store {vector_store_id}...")
        while True:
            try:
                file_status = client.beta.vector_stores.files.retrieve(
                    vector_store_id=vector_store_id,
                    file_id=file_object.id
                )
                logger.debug(f"Polling file status: {file_status.status}")
                if file_status.status == 'completed':
                    logger.info(f"File {file_object.id} processing completed in Vector Store.")
                    break
                elif file_status.status in ['failed', 'cancelled']:
                    logger.error(f"File processing failed with status: {file_status.status}")
                    # Attempt cleanup
                    client.beta.vector_stores.delete(vector_store_id)
                    logger.info(f"Deleted incomplete Vector Store {vector_store_id}")
                    sys.exit(1)
            except Exception as e:
                 # Catch potential transient errors during polling
                 logger.warning(f"Polling error checking file status (will retry): {e}")
            time.sleep(5) # Wait before polling again

        # 6. Print the final ID for the user
        print("\n--- Vector Store Setup Complete ---")
        print(f"Vector Store Name: {vector_store.name}")
        print(f"Successfully created and populated Vector Store.")
        print(f"Please add the following line to your .env file:")
        print(f"\nDEFAULT_VECTOR_STORE_ID=\"{vector_store_id}\"\n")

    except Exception as e:
        logger.error(f"An error occurred during Vector Store setup: {e}", exc_info=True)
        # Attempt to clean up the created vector store if an error occurred after creation
        if vector_store_id:
            try:
                logger.info(f"Attempting to delete partially created Vector Store {vector_store_id} due to error...")
                client.beta.vector_stores.delete(vector_store_id)
                logger.info(f"Successfully deleted Vector Store {vector_store_id}.")
            except Exception as delete_e:
                logger.error(f"Failed to delete Vector Store {vector_store_id} during cleanup: {delete_e}")
        sys.exit(1)

# --- Script Entry Point ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create an OpenAI Vector Store and upload a playbook file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "playbook_file",
        help="Path to the negotiation playbook file (e.g., .md, .txt)."
    )

    args = parser.parse_args()
    setup_store(args.playbook_file) 