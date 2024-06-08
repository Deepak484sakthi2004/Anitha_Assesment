import nest_asyncio
from llama_parse import LlamaParse
import os
from scripts.saveProcessedData import addData
nest_asyncio.apply()

# API access to llama-cloud 
os.environ["LLAMA_CLOUD_API_KEY"] = "llx-....."

def load_pdf_files(directory):
    current_directory = os.path.join(os.getcwd(), directory)
    
    # Check if the directory exists
    if not os.path.exists(current_directory):
        print(f"Directory '{current_directory}' does not exist.")
        return {}

    files = os.listdir(current_directory)
    
    pdf_files = [file for file in files if file.endswith('.pdf')]
    
    for file in pdf_files:
        file_path = os.path.join(current_directory, file)
        print(f"Loading content from {file_path}...")

        # Load data from the PDF file
        text_data = LlamaParse(result_type="markdown").load_data(file_path)
        
        # Extract text content if the result is a list of Document objects
        if isinstance(text_data, list):
            text = "\n".join([doc.text for doc in text_data])
        else:
            text = str(text_data)
        
        # Save the processed content using addResume function
        file_name_without_extension = os.path.splitext(file)[0]
        addData("processedData", file_name_without_extension, text)
        print("-" * 10, file +" has been parsed!!", "-" * 10)
        

def main(directory):
    load_pdf_files(directory)
