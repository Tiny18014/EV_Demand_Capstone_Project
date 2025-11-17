import json

json_string = '{"Political": {"Subsidies": 0.18, "Charging Policy": 0.11, "Import Duties": 0.06},"Economical": {"Subsidies": 0.18, "Charging Policy": 0.11, "Import Duties": 0.06}}'

# Convert string to Python dict
try:
    data = json.loads(json_string)
    print(data)
    print(type(data))  # <class 'dict'>
except json.JSONDecodeError as e:
    print("Invalid JSON:", e)