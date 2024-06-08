import os

def extract_txt_files(directory):
    current_directory = os.path.join(os.getcwd(), directory)
    
    # Check if the directory exists
    if not os.path.exists(current_directory):
        return f"Directory '{current_directory}' does not exist."

    files = os.listdir(current_directory)
    
    txt_files = [file for file in files if file.endswith('.txt')]
    file_contents = {}
    
    for file in txt_files:
        file_path = os.path.join(current_directory, file)

        # Read data from the .txt file
        with open(file_path, 'r') as txt_file:
            content = txt_file.read()
        
        file_contents[file] = content
    
    return file_contents

def readParsedData(directory):
     file_contents = extract_txt_files(directory)
     if isinstance(file_contents, str):
         return file_contents  # Return the error message as a string
     
     # Format the file contents dictionary into a string
     result = ""
     for file, content in file_contents.items():
         result += f"File: {file}\nContent:\n{content}\n\n"
     
     return result.strip()

