# AvioBot- Pilot Voice Assistant

## Project Description
**Aviobot** is an LLM-powered voice assistant bot developed to help pilots manage various tasks more efficiently during flights. The project enables pilots to query flight data in real-time through voice commands, providing quick access to critical information. It is an advanced copilot.

Aviobot leverages LangChain technology for natural language processing (NLP) capabilities, allowing it to understand and respond to pilot queries instantly. The project is specifically designed to enhance flight safety while reducing the workload on pilots. It uses OpenAI API Key to communicate with OpenAI GP4-4-turbo-Preview model. 

Aviobot uses OpenAI's TTS-1 Nova voice model as the speech synthesis engine, enabling natural and clear voice interactions. Leveraging LangChain technology for natural language processing (NLP), Aviobot understands and responds to pilot queries instantly, enhancing flight safety while reducing the workload on pilots. In addition, Aviobot employs Pinecone as the vector database, which facilitates efficient data storage and retrieval, ensuring that queries are processed quickly and accurately. This combination of technologies is specifically designed to enhance flight safety while reducing the workload on pilots.

You can find the Jupyter Notebook files in the `notebooks` folder. This folder contains several notebooks, each focused on different aspects of the project:

- **`OpenAI.ipynb`**: Demonstrates how to work with OpenAI models using LangChain.
- **`GoogleAI.ipynb`**: Explores the use of Google Models.

Feel free to explore these notebooks in the `notebooks` folder to better understand the implementation details.

## Installation 

To get Aviobot up and running on your local machine, follow the steps below:

  **Clone the Repository**

First, clone the Aviobot repository from GitHub to your local machine. I recommand that if you use PyCharm IDE, It going to be very simple to run. For PyCharm IDE,
1. Select VCS option from main bar
2. Choose "Get from Version Control"
3. copy "https://github.com/bmoz23/AvioBot.git" and paste it.
4. After these steps, you have to skip your clone folder.
5. Open your terminal, run this command --> **` pip install -r requirements.txt `**
6. Therefore, all requirements of our project must be installed


## Running
1. Open your clone folder.
2. Open your terminal.
3. Run the following command **`streamlit run app.py`**
4. If you want to run our chat model, you have to run the following command **`streamlit run chat.py`**
     
## Information

Please make sure to replace the API keys in the **`req.env`** file with **your own API keys**. This step is essential for AvioBot to function correctly.

**Steps to Update the `req.env` File:**

1. Replace the `OPENAI_API_KEY` with your own OpenAI API key.
2. Replace the `PINECONE_API_KEY` with your own Pinecone API key.

Your file should look like this:

```plaintext
OPENAI_API_KEY="your-openai-api-key"
PINECONE_API_KEY="your-pinecone-api-key"


    

