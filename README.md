# AI Assistant for NDA Redlining

This project demonstrates an AI-powered assistant designed to analyze Non-Disclosure Agreements (NDAs). It leverages the OpenAI Assistants API with Vector Stores to summarize the contract, identify key clauses, and provide redline suggestions based on a pre-defined negotiation playbook.

## Overview

Manually reviewing NDAs against company standards or a negotiation playbook can be time-consuming. This assistant automates the initial review process by:

1.  **Loading:** Reading contract text from PDF or plain text files.
2.  **Summarizing:** Generating a concise summary of the NDA, highlighting key elements like parties, duration, and core obligations using.
3.  **Clause Identification:** Breaking down the contract into distinct legal clauses.
4.  **Redlining:** Analyzing each clause against guidance stored in an OpenAI Vector Store using the `file_search` tool. It suggests modifications (strike-throughs for deletions, bold for additions) and adds comments explaining the reasoning based on the playbook.

The project uses `asyncio` for managing API calls and includes concurrency control (`asyncio.Semaphore`) to handle potential API rate limits, particularly those associated with the `file_search` tool during the redlining phase.

## Features

*   Supports PDF and plain text NDA files.
*   Uses `gpt-4o` for high-quality summarization and clause identification.
*   Leverages OpenAI Vector Stores and the `file_search` tool for context-aware redlining against a custom playbook.
*   Provides explanations for redline suggestions.
*   Implements sequential processing for redlining to manage API rate limits effectively.

## Technology Stack

*   Python 3.10+
*   OpenAI Python Library (`openai`)
*   OpenAI Responses API (including Vector Stores & File Search)
*   `pdfplumber` for PDF text extraction
*   `asyncio` for asynchronous operations

## Setup

Follow these steps to set up and run the project locally:

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

2.  **Create and activate a virtual environment:**
    *   On macOS/Linux:
        ```bash
        python3 -m venv .venv
        source .venv/bin/activate
        ```
    *   On Windows:
        ```bash
        python -m venv .venv
        .venv\Scripts\activate
        ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Create `.env` file:**
    *   Create a file named `.env` in the root directory of the project.
    *   Copy the contents of `.env.example` into `.env`.
    *   Open the `.env` file and add your OpenAI Project API Key:
        ```dotenv
        OPENAI_API_KEY="OPENAI_API_KEY"
        # DEFAULT_VECTOR_STORE_ID will be added in the next step
        ```

5.  **Set up OpenAI Vector Store via Script:**
    *   Run the setup script, providing the path to your playbook file. You can use the sample playbook provided:

        ```bash
        python scripts/setup_vector_store.py <path_to_your_nda_file.pdf>
        ```

    *   The script will:
        *   Upload the playbook file to OpenAI.
        *   Create a new Vector Store containing this file.
        *   Wait for the file to be processed.
        *   Print the new `DEFAULT_VECTOR_STORE_ID` to the console.
    *   **Copy the printed `DEFAULT_VECTOR_STORE_ID=...` line and paste it into your `.env` file.**

## Usage

Run the analysis script from the root directory, providing the path to the NDA file you want to analyze:

```bash
python src/main.py <path_to_your_nda_file.pdf>
```

The script will output the summary and redlined clauses to a new Markdown (`.md`) file named after the input contract. **This output file is saved in the project's root directory.**

## Example Output Snippet

```text
=== Final Aggregated Output ===
=== NDA Contract Summary ===
**Summary of the NDA:**

**Parties Involved:**
- **Discloser:**
- **Recipient:**

... (rest of summary) ...

=== Redlined Clauses (8 processed) ===
~~Recipient shall~~ **The Recipient agrees to** treat as confidential, non-public, and proprietary any an
d all **Confidential Information, which includes** data and other information obtained from or on behalf
of ~~Disclosers~~ **the Discloser** and its affiliates... **[Standard carve-outs added]**

<!-- comment -->
The redlined changes incorporate standard carve-outs...

... (rest of redlined clauses) ...
```

## Project Structure

```
.
├── .env.example            # Example environment variables file
├── .gitignore              # Git ignore file
├── LICENSE                 # Project license (e.g., MIT)
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── default_docs/           # Directory for sample documents
└── src/
    └── main.py             # Main application script
```

## Limitations

*   **Sequential Redlining:** Due to potential rate limits with the `file_search` tool when run concurrently, the redlining step processes clauses sequentially. This ensures stability but is slower than parallel processing.
*   **Playbook Dependency:** The quality of redlines heavily depends on the content and structure of the negotiation playbook uploaded to the OpenAI Vector Store.
*   **User-Specific Playbooks:** The current implementation uses a single default Vector Store. The code includes stubs (`get_user_vector_store_id`) for potentially adding user-specific stores in the future but is not fully implemented.
*   **Structured Output Reliance:** The clause breaker relies on the LLM strictly following the "|||" delimiter instruction. Variations could break clause separation.