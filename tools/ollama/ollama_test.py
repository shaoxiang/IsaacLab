import requests
import json
import numpy as np

url = 'http://127.0.0.1:11434/api/generate'
url = 'http://192.168.1.33:11434/api/generate'

# "qwen2.5:0.5b" Function took 0.6855 seconds to complete.
# "qwen2.5:3b"  Function took 0.7327 seconds to complete.
# "qwen2.5:3b"  local Function took 0.5532 seconds to complete.
# "qwen2.5:7b" local Function took 0.7391 seconds to complete.
# "qwen2.5:14b" Function took 1.0559 seconds to complete.

# 指定文件路径（这里假设文件在当前工作目录下）  
file_path = 'prompt_en.txt'  # 请将'your_file.txt'替换为你的实际文件名  
# 打开文件并读取内容  
with open(file_path, 'r', encoding='utf-8') as file:
    data_lines = file.readlines()
    file.seek(0)
    file_content = file.read()
    
# 现在file_content包含了文件的全部内容作为一个字符串  
print(file_content)

items_index_and_coordinates = []

for line in data_lines:
    if line.find("、") > 0:
        items_index_and_coordinates.append(line.split(": ")[-1].replace('\n', ''))
        
print(items_index_and_coordinates)

# user_input = "I would like to drink some water"

user_input_list = ["I would like to drink some water",
                   "get some cold cola",
                   "shutdown the computer",
                   "drag heavy things",
                   "drag light things",
                   "Fire! Fire! ",
                   "Clean the floor",
                   "Check the forklift in the middle of the house",
                   "Take inventory of the materials",
                   "Wipe the rectangular wooden table",
                   "Look at what's on the round table",
                   "Throw away the garbage"]

# print("data:", llm_data)

headers = {
    'Content-Type': 'application/json'
}

import time
start_time = time.time()  

for user_input in user_input_list:
    
    llm_data = {
        "model": "qwen2.5:32b",
        "prompt": file_content + user_input,
        "stream": False,
        "temperature": 0.1,
    }

    response = requests.post(url, data=json.dumps(llm_data), headers=headers)
    print(f"Status Code: {response.status_code}")
    try:
        response_json = response.json()
        print("input:", user_input, f",response:", response_json["response"])
        item_index = int(response_json["response"]) - 1
        # print(f"item coordinates:", items_index_and_coordinates[item_index])
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")

end_time = time.time()  
  
elapsed_time = end_time - start_time 
print(f"Function took {elapsed_time / 20.0 :.4f} seconds to complete.")
