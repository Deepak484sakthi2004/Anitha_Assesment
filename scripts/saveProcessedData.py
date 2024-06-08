import os

def addData(folder_path, file_name, content):
    try:
        # Ensure the folder exists
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # Create the full file path
        file_path = os.path.join(folder_path, file_name + ".txt")
        
        # Open the file in write mode and add content
        with open(file_path, "w") as txt_file:
            txt_file.write(content)
        
        print(f"File '{file_path}' created and content added successfully.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")






