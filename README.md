# Private-Llama2-File-Chat

Llama2 Chatbot Package Overview
===============================

Introduction:
-------------
The Llama2 Chatbot package provides scripts that permit local document querying.  
It utilizes various data sources, including pdf, xlsx, and txt files, to ingest and process data. 
The processed data is then used by a retrieval-based chatbot to answer user queries.

Scripts:
--------
1. `model.py`:
    - Purpose: Set up the chatbot's underlying model and manage interactions with users.
    - Main Functions: 
        - `set_custom_prompt()`: Sets up a custom prompt template for the chatbot.
        - `retrieval_qa_chain()`: Configures a retrieval-based question-answering chain.
        - `load_llm()`: Loads the large language model.
        - `qa_bot()`: Initializes the QA bot components.
        - `main()`: Main function to set up chatbot event handlers and manage interactions.

2. `ingest.py`:
    - Purpose: Responsible for ingesting various data sources, processing them, and storing the processed data in a vector database.
    - Main Functions:
        - `load_txt_files()`: Loads content from TXT files in a specified directory.
        - `create_vector_db()`: Processes data from various sources and creates a vector database.

(Note: Detailed documentation for each script can be found in corresponding .txt files, e.g., `model.txt` for `model.py`.)

Usage:
------
To use the Llama2 Chatbot, ensure that the required data sources are available in the specified 'data' directory. 
This data can be in the file format of pdf, txt, or xlsx.
Run the `ingest.py` script first to process the data and create the vector database. 
Once the database is ready, open Git Bash within your folder, and input/execute the following: chainlit run model.py -w to start the chatbot and interact with users.
