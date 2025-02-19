import re
import ast
import torch

# read files
with open(r'E:\LLM_CODE\pred_score\data\output.txt', 'r', encoding='utf-8') as file:
    content = file.read()

# find all tensor representations using regular expressions
# tensor_pattern = re.compile(r'tensor$[\s\S]*?$\s*,?', re.DOTALL)
tensor_pattern = re.compile(r'\[.*?\]', re.DOTALL)
matches = tensor_pattern.findall(content)


# print matches
# print("Matches:", matches)
print(len(matches))

# parsing each match and converting to tensor
tensors = []
for match in matches:
    try:
        # delete "tensor" and "device" parts
        tensor_str = match.replace('[tensor(', '').replace(')', '').replace('tensor(', '').replace(', device=\'cuda:0\'', '').strip()
        # converting strings to lists
        tensor_list = ast.literal_eval(tensor_str)
        # converting lists to tensors
        tensor = torch.tensor(tensor_list)
        tensors.append(tensor)
    except Exception as e:
        print(f"Error parsing tensor: {match}")
        print(e)

# print the parsed tensor
print("Parsed Tensors:")
a=torch.zeros_like(tensors[0])
print(a.shape)
for tensor in tensors:
    a = a + tensor
a=a/len(tensors)
print(a)
