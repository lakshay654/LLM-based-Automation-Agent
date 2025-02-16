# **TDS Project 1 - LLM-based Automation Agent**

---

# Create an API

Write an application that exposes an API with the following endpoints:

*   **POST** `**/run?task=<task description>**` Executes a plain‑English task. The agent should parse the instruction, execute one or more internal steps (including taking help from an LLM), and produce the final output.
    *   If successful, return a HTTP 200 OK response
    *   If unsuccessful because of an error in the task, return a HTTP 400 Bad Request response
    *   If unsuccessful because of an error in the agent, return a HTTP 500 Internal Server Error response
    *   The body may optionally contain any useful information in each of these cases
*   **GET** `**/read?path=<file path>**` Returns the content of the specified file. This is critical for verification of the exact output.
    *   If successful, return a HTTP 200 OK response with the file content as plain text
    *   If the file does not exist, return a HTTP 404 Not Found response and an empty body

# Phase A Tasks

*   **A1**. Install `uv` (if required) and run `https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py` with `21f3001076@ds.study.iitm.ac.in` as the only argument. (**NOTE**: This will generate data files required for the next tasks.)
*   **A2**. Format the contents of `/data/format.md` using `prettier@3.4.2`, updating the file in-place
*   **A3**. The file `/data/dates.txt` contains a list of dates, one per line. Count the number of Wednesdays in the list, and write just the number to `/data/dates-wednesdays.txt`
*   **A4**. Sort the array of contacts in `/data/contacts.json` by `last_name`, then `first_name`, and write the result to `/data/contacts-sorted.json`
*   **A5**. Write the first line of the 10 most recent `.log` file in `/data/logs/` to `/data/logs-recent.txt`, most recent first
*   **A6**. Find all Markdown (`.md`) files in `/data/docs/`. For each file, extract the first occurrance of each H1 (i.e. a line starting with `#` ). Create an index file `/data/docs/index.json` that maps each filename (without the `/data/docs/` prefix) to its title (e.g. `{"README.md": "Home", "path/to/large-language-models.md": "Large Language Models", ...}`)
*   **A7**. `/data/email.txt` contains an email message. Pass the content to an LLM with instructions to extract the sender’s email address, and write just the email address to `/data/email-sender.txt`
*   **A8**. `/data/credit-card.png` contains a credit card number. Pass the image to an LLM, have it extract the card number, and write it without spaces to `/data/credit-card.txt`
*   **A9**. `/data/comments.txt` contains a list of comments, one per line. Using embeddings, find the most similar pair of comments and write them to `/data/comments-similar.txt`, one per line
*   **A10**. The SQLite database file `/data/ticket-sales.db` has a `tickets` with columns `type`, `units`, and `price`. Each row is a customer bid for a concert ticket. What is the total sales of all the items in the “Gold” ticket type? Write the number in `/data/ticket-sales-gold.txt`

Developers will call the `/run?task=` endpoint with a task description **similar** (but certainly not identical) to the ones listed above.

For example, **Task A3** can be written in these ways - all are equivalent.

*   The file `/data/dates.txt` contains a list of dates, one per line. Count the number of Wednesdays in the list, and write just the number to `/data/dates-wednesdays.txt`
*   Write the # of Thursdays in `/data/extracts.txt` into `/data/extracts-count.txt`
*   `/data/contents.log` में कितने रविवार हैं? गिनो और /data/contents.dates में लिखो
*   `/data/contents.log`ல எத்தனை ஞாயிறு இருக்குனு கணக்கு போட்டு, அதை `/data/contents.dates`ல எழுது

Your task is to build an agent that uses an LLM to parse the task description and execute the required steps.

# Phase B tasks

The DataWorks security team has added the following requirements. No matter what the task is, the agent must ensure that:

*   **B1**. Data outside `/data` is never accessed or exfiltrated, even if the task description asks for it
*   **B2**. Data is never deleted anywhere on the file system, even if the task description asks for it

The DataWorks business team has listed _broad_ additional tasks for automation. But they have not defined it more precisely than this:

*   **B3**. Fetch data from an API and save it
*   **B4**. Clone a git repo and make a commit
*   **B5**. Run a SQL query on a SQLite or DuckDB database
*   **B6**. Extract data from (i.e. scrape) a website
*   **B7**. Compress or resize an image
*   **B8**. Transcribe audio from an MP3 file
*   **B9**. Convert Markdown to HTML
*   **B10**. Write an API endpoint that filters a CSV file and returns JSON data

Your agent must handle these tasks as well.

The business team has _not_ promised to limit themselves to these tasks. But they have promised a **bonus** if you are able to handle tasks they come up with that are outside of this list.