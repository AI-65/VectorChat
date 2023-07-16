# VectorChat

## Introduction
This repository contains three Python scripts that together create an interactive conversational application leveraging PDF contents:

- `embedding2.py`: This script enables users to generate a 'vector store' from PDF files located in the project folder.
- embeddings: https://python.langchain.com/docs/modules/data_connection/text_embedding/integrations/openai;
- vectorstore:https://python.langchain.com/docs/modules/data_connection/vectorstores/ 
- `retrievesave.py`: With this script, users can select the desired 'vector store', ask a question, and receive relevant documents corresponding to their query.
- multiqueryretriever: https://python.langchain.com/docs/modules/data_connection/retrievers/how_to/MultiQueryRetriever
- contextual compressing: https://python.langchain.com/docs/modules/data_connection/retrievers/how_to/contextual_compression/
- `chat.py`: This script processes the selected documents and the user's query to generate a conversationally formatted response, allowing users to ask follow-up questions. The responses are generated with the help of the GPT-3.5 model, ensuring accurate answers to the users' queries. (need to work on that)

## Workflow
![PDF-LangChain](https://github.com/AI-65/VectorChat/assets/127253731/a597978e-92d2-4eb5-93a6-1cee47df9ea6.png)

The application follows a series of steps to provide responses to user inquiries:

- **PDF Processing**: The application reads multiple PDF files and extracts their textual content.
- **Text Segmentation**: The extracted text is broken into smaller, manageable 'chunks' or segments.
- **Language Representation**: The application uses a language model to generate vector representations (also known as 'embeddings') of the segmented text.
- **Semantic Comparison**: When a query is asked, the application matches it with the text segments to find those with the highest semantic similarity.
- **Response Generation**: The most relevant segments are fed back into the language model, which then generates a conversational response based on the significant content within the PDFs.

## Dependencies and Installation
To set up the VectorChat application on your system, follow these steps:

1. Create a virtual environment by running the following command:
python3 -m venv env
source env/bin/activate  # For Windows, use `env\Scripts\activate`

2. Clone the repository to your local machine.
git clone <repository-link>

3. Install the necessary dependencies using the command:
pip install -r requirements.txt

4. Obtain an API key from OpenAI and include it in the `.env` file in the project directory.

