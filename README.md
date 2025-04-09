# Python Basics for Data Analytics

## Table of Contents
1. [Introduction to Python](#introduction-to-python)
2. [Setting Up Python Environment](#setting-up-python-environment)
3. [Basic Python Syntax](#basic-python-syntax)
4. [Data Types](#data-types)
5. [Variables and Assignment](#variables-and-assignment)
6. [Operators](#operators)
7. [Control Flow](#control-flow)
8. [Functions](#functions)
9. [Data Structures](#data-structures)
10. [File Handling](#file-handling)
11. [Error Handling](#error-handling)
12. [Modules and Packages](#modules-and-packages)
13. [Next Steps](#next-steps)

## Introduction to Python

Python is a high-level, interpreted programming language known for its readability and simplicity. It has become the language of choice for data science and analytics due to several key features:

- **Readability**: Python's clean syntax makes code easy to read and write
- **Versatility**: Suitable for web development, automation, scientific computing, and data analysis
- **Rich Ecosystem**: Powerful libraries for data manipulation (Pandas), numerical computing (NumPy), visualization (Matplotlib), and machine learning (Scikit-learn)
- **Community Support**: Active community with extensive documentation and resources

For data analytics specifically, Python offers:
- Simple syntax that allows analysts to focus on data problems rather than programming complexities
- Libraries designed specifically for data tasks
- Integration capabilities with various data sources and formats
- Scalability from simple analysis to complex machine learning projects

## Setting Up Python Environment

### Installing Python

For data analytics, I recommend installing Python via Anaconda, which comes with most data science libraries pre-installed:

1. Download Anaconda from [https://www.anaconda.com/download/](https://www.anaconda.com/download/)
2. Follow installation instructions for your operating system
3. Verify installation by opening a terminal/command prompt and typing:
   ```python
   python --version
   ```

### Creating a Virtual Environment

Virtual environments allow you to manage dependencies for different projects:

```bash
# Create a new environment
conda create --name dataanalytics python=3.10

# Activate the environment
conda activate dataanalytics

# Install essential data analytics packages
conda install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### IDEs and Notebooks

For data analytics, consider using:
- **Jupyter Notebooks**: Interactive environment ideal for exploratory data analysis
- **VS Code**: Full-featured IDE with excellent Python support
- **PyCharm**: Powerful IDE with data science tools
- **Spyder**: IDE designed specifically for scientific computing

To launch Jupyter Notebook:
```bash
jupyter notebook
```

## Basic Python Syntax

Python's syntax is designed to be readable and straightforward:

### Comments
```python
# This is a single-line comment

"""
This is a 
multi-line comment or docstring
"""
```

### Print Statement
```python
print("Hello, Data Analysts!")
```

### Indentation
Python uses indentation (typically 4 spaces) to define code blocks instead of braces:

```python
if True:
    print("This is indented")
    if True:
        print("This is further indented")
print("Back to normal")
```

### Line Continuation
Long lines can be broken using backslashes or implied line continuation within parentheses, brackets, or braces:

```python
# Using backslash
total = 1 + 2 + 3 + \
        4 + 5

# Implied continuation within parentheses
total = (1 + 2 + 3 +
         4 + 5)
```

## Data Types

Python has several built-in data types that are essential for data analytics:

### Numeric Types
```python
# Integer
count = 10

# Float
temperature = 98.6

# Complex number
complex_num = 3 + 4j
```

### Strings
```python
# String literals
name = "Data Analyst"
multiline_string = """This is a 
multiline string"""

# String operations
print(name.upper())  # DATA ANALYST
print(name.split())  # ['Data', 'Analyst']
print(len(name))     # 12
print(name[0:4])     # Data (slicing)
```

### Boolean
```python
is_valid = True
has_data = False

# Boolean operations
print(is_valid and has_data)  # False
print(is_valid or has_data)   # True
print(not is_valid)           # False
```

### None Type
```python
result = None  # Represents absence of value
```

### Type Checking and Conversion
```python
# Check type
print(type(count))       # <class 'int'>
print(type(temperature)) # <class 'float'>

# Type conversion
int_to_float = float(10)      # 10.0
float_to_int = int(10.9)      # 10 (truncates decimal part)
number_to_string = str(10.5)  # "10.5"
string_to_float = float("10.5")  # 10.5
```

## Variables and Assignment

Variables in Python are dynamically typed, meaning the type is determined at runtime:

```python
# Basic assignment
x = 10
name = "Python"

# Multiple assignment
a, b, c = 1, 2, 3

# Augmented assignment
count = 0
count += 1  # Same as count = count + 1

# Variable naming conventions
# Use snake_case for variables
first_name = "John"
last_name = "Doe"

# Constants are typically UPPERCASE
PI = 3.14159
MAX_SIZE = 100
```

### Variable Scope

```python
# Global variable
global_var = "I'm global"

def function():
    # Local variable
    local_var = "I'm local"
    print(global_var)  # Can access global
    print(local_var)   # Can access local

function()
print(global_var)      # Can access global
# print(local_var)     # ERROR: Cannot access local outside function
```

## Operators

Python provides various operators for different operations:

### Arithmetic Operators
```python
a = 10
b = 3

addition = a + b        # 13
subtraction = a - b     # 7
multiplication = a * b  # 30
division = a / b        # 3.3333... (returns float)
floor_division = a // b # 3 (returns integer)
modulus = a % b         # 1 (remainder)
exponent = a ** b       # 1000 (10^3)
```

### Comparison Operators
```python
a = 10
b = 20

equal = a == b              # False
not_equal = a != b          # True
greater_than = a > b        # False
less_than = a < b           # True
greater_or_equal = a >= b   # False
less_or_equal = a <= b      # True
```

### Logical Operators
```python
x = True
y = False

and_result = x and y  # False
or_result = x or y    # True
not_result = not x    # False
```

### Identity Operators
```python
a = [1, 2, 3]
b = [1, 2, 3]
c = a

print(a is c)      # True (same object)
print(a is b)      # False (different objects)
print(a is not b)  # True
```

### Membership Operators
```python
numbers = [1, 2, 3, 4, 5]

print(3 in numbers)      # True
print(6 not in numbers)  # True
```

### Bitwise Operators (less common in data analytics)
```python
a = 10  # 1010 in binary
b = 3   # 0011 in binary

print(a & b)   # 2 (0010) - AND
print(a | b)   # 11 (1011) - OR
print(a ^ b)   # 9 (1001) - XOR
print(~a)      # -11 - NOT
print(a << 1)  # 20 (10100) - Left shift
print(a >> 1)  # 5 (0101) - Right shift
```

## Control Flow

Control flow statements determine the execution path of your code.

### Conditional Statements

```python
# If statement
age = 25

if age < 18:
    print("Minor")
elif age < 65:
    print("Adult")
else:
    print("Senior")

# Conditional expressions (ternary operator)
status = "Adult" if age >= 18 else "Minor"
```

### Loops

#### For Loops
```python
# Iterate over a list
fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(fruit)

# Iterate with index
for i, fruit in enumerate(fruits):
    print(f"Index {i}: {fruit}")

# Range-based for loop
for i in range(5):  # 0, 1, 2, 3, 4
    print(i)

for i in range(2, 8, 2):  # 2, 4, 6 (start, stop, step)
    print(i)
```

#### While Loops
```python
# Basic while loop
count = 0
while count < 5:
    print(count)
    count += 1

# Break statement
number = 0
while True:
    print(number)
    number += 1
    if number >= 5:
        break  # Exit loop when number reaches 5
```

#### Loop Control
```python
# Continue statement - skips the current iteration
for i in range(10):
    if i % 2 == 0:
        continue  # Skip even numbers
    print(i)  # Prints 1, 3, 5, 7, 9

# Break statement - exits the loop
for i in range(10):
    if i > 5:
        break  # Exit loop when i > 5
    print(i)  # Prints 0, 1, 2, 3, 4, 5
```

#### List Comprehensions
A concise way to create lists - very useful in data analytics:

```python
# Create a list of squares
squares = [x**2 for x in range(10)]
# [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# With condition
even_squares = [x**2 for x in range(10) if x % 2 == 0]
# [0, 4, 16, 36, 64]

# With two loops (nested)
coordinates = [(x, y) for x in range(3) for y in range(2)]
# [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
```

## Functions

Functions are reusable blocks of code that perform a specific task:

### Function Definition and Calling
```python
# Define a function
def greet(name):
    """This function greets the person passed in as a parameter"""
    return f"Hello, {name}!"

# Call the function
message = greet("Data Analyst")
print(message)  # Hello, Data Analyst!
```

### Parameters and Arguments
```python
# Default parameters
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

print(greet("John"))           # Hello, John!
print(greet("John", "Hi"))     # Hi, John!

# Keyword arguments
def describe_person(name, age, job):
    return f"{name} is {age} years old and works as a {job}."

print(describe_person(age=30, job="Data Analyst", name="Alice"))

# Variable-length arguments
def add_numbers(*args):
    """Sum any number of arguments"""
    return sum(args)

print(add_numbers(1, 2, 3, 4))  # 10

# Variable-length keyword arguments
def person_info(name, **kwargs):
    """Print name and any additional info provided"""
    info = f"Name: {name}\n"
    for key, value in kwargs.items():
        info += f"{key}: {value}\n"
    return info

print(person_info("Alice", age=30, job="Data Analyst", city="New York"))
```

### Lambda Functions (Anonymous Functions)
```python
# Regular function
def square(x):
    return x**2

# Equivalent lambda function
square_lambda = lambda x: x**2

print(square(5))        # 25
print(square_lambda(5)) # 25

# Common use with map/filter
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, numbers))
evens = list(filter(lambda x: x % 2 == 0, numbers))

print(squared)  # [1, 4, 9, 16, 25]
print(evens)    # [2, 4]
```

### Scope and Closures
```python
def outer_function(x):
    """Demonstrates a closure - a function that remembers its environment"""
    def inner_function(y):
        return x + y  # Uses x from outer scope
    return inner_function

add_five = outer_function(5)
result = add_five(10)  # 15
```

## Data Structures

Python has several built-in data structures that are vital for data analytics:

### Lists
```python
# Creating lists
empty_list = []
numbers = [1, 2, 3, 4, 5]
mixed = [1, "hello", 3.14, True]

# Accessing elements
first = numbers[0]      # 1 (0-indexed)
last = numbers[-1]      # 5 (negative indexing)
subset = numbers[1:3]   # [2, 3] (slicing)

# List methods
numbers.append(6)       # Add to end: [1, 2, 3, 4, 5, 6]
numbers.insert(0, 0)    # Insert at index: [0, 1, 2, 3, 4, 5, 6]
numbers.remove(3)       # Remove value: [0, 1, 2, 4, 5, 6]
popped = numbers.pop()  # Remove & return last: 6, numbers is now [0, 1, 2, 4, 5]
numbers.sort()          # Sort in place: [0, 1, 2, 4, 5]
numbers.reverse()       # Reverse in place: [5, 4, 2, 1, 0]

# Checking properties
length = len(numbers)           # 5
contains = 4 in numbers         # True
count_ones = numbers.count(1)   # 1
index_of_4 = numbers.index(4)   # 1 (index of first occurrence)

# List operations
combined = numbers + [6, 7, 8]  # Concatenation
repeated = [0] * 3              # [0, 0, 0]
```

### Tuples
Immutable sequences (cannot be changed after creation):

```python
# Creating tuples
empty_tuple = ()
single_item = (1,)  # Comma is required for single item
coordinates = (10, 20)
mixed = (1, "hello", 3.14)

# Accessing elements (same as lists)
x = coordinates[0]  # 10

# Unpacking
x, y = coordinates  # x = 10, y = 20

# Tuple methods (fewer than lists since immutable)
count_ones = (1, 2, 1, 3).count(1)  # 2
index_of_2 = (1, 2, 1, 3).index(2)  # 1

# Immutability demonstration
try:
    coordinates[0] = 5  # This will raise TypeError
except TypeError as e:
    print("Tuples are immutable!")
```

### Dictionaries
Key-value pairs, extremely useful for structured data:

```python
# Creating dictionaries
empty_dict = {}
person = {
    "name": "Alice",
    "age": 30,
    "job": "Data Analyst",
    "skills": ["Python", "SQL", "Tableau"]
}

# Accessing values
name = person["name"]  # Alice
# Or safely with get method:
age = person.get("age", 0)  # Returns 0 if key doesn't exist

# Modifying dictionaries
person["age"] = 31  # Update value
person["city"] = "New York"  # Add new key-value pair
del person["job"]  # Remove key-value pair
skill = person.pop("skills")  # Remove and return value

# Dictionary methods
keys = list(person.keys())    # List of keys
values = list(person.values())  # List of values
items = list(person.items())    # List of (key, value) tuples

# Checking for keys
has_name = "name" in person  # True

# Dictionary comprehensions
squares = {x: x**2 for x in range(5)}
# {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}
```

### Sets
Unordered collections of unique elements:

```python
# Creating sets
empty_set = set()  # Not {} which creates empty dict
fruits = {"apple", "banana", "cherry"}
numbers = set([1, 2, 2, 3, 3, 3])  # {1, 2, 3} (duplicates removed)

# Set operations
fruits.add("orange")      # Add element
fruits.remove("banana")   # Remove element (raises error if not found)
fruits.discard("kiwi")    # Remove if present (no error if not found)
popped = fruits.pop()     # Remove and return arbitrary element

# Set operations (mathematical)
a = {1, 2, 3, 4}
b = {3, 4, 5, 6}

union = a | b             # {1, 2, 3, 4, 5, 6}
intersection = a & b      # {3, 4}
difference = a - b        # {1, 2}
symmetric_diff = a ^ b    # {1, 2, 5, 6}

# Checking membership
contains = "apple" in fruits  # True or False
```

## File Handling

Working with files is essential for data analytics:

### Reading Files
```python
# Basic file reading
with open('data.txt', 'r') as file:
    content = file.read()  # Read entire file
    print(content)

# Reading line by line
with open('data.txt', 'r') as file:
    for line in file:
        print(line.strip())  # strip() removes trailing newline

# Reading all lines at once
with open('data.txt', 'r') as file:
    lines = file.readlines()  # Returns list of lines
```

### Writing Files
```python
# Writing to a file
with open('output.txt', 'w') as file:  # 'w' overwrites existing file
    file.write("Hello, Data Analytics!\n")
    file.write("Python is awesome.")

# Appending to a file
with open('output.txt', 'a') as file:  # 'a' appends to existing file
    file.write("\nAppending new data.")

# Writing multiple lines
lines = ["Line 1", "Line 2", "Line 3"]
with open('output.txt', 'w') as file:
    file.writelines([line + '\n' for line in lines])
```

### Working with CSV Files

CSV (Comma-Separated Values) is a common format for data analytics:

```python
import csv

# Reading CSV
with open('data.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        print(row)  # Each row is a list

# Reading CSV with column names
with open('data.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        print(row)  # Each row is a dictionary with column names as keys

# Writing CSV
data = [
    ['Name', 'Age', 'Job'],
    ['Alice', 30, 'Data Analyst'],
    ['Bob', 25, 'Engineer']
]

with open('output.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(data)

# Writing CSV with column names
data = [
    {'Name': 'Alice', 'Age': 30, 'Job': 'Data Analyst'},
    {'Name': 'Bob', 'Age': 25, 'Job': 'Engineer'}
]

with open('output_dict.csv', 'w', newline='') as csvfile:
    fieldnames = ['Name', 'Age', 'Job']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    writer.writerows(data)
```

### Working with JSON Files

JSON (JavaScript Object Notation) is another common format:

```python
import json

# Reading JSON
with open('data.json', 'r') as jsonfile:
    data = json.load(jsonfile)
    print(data)  # Python dictionary

# Writing JSON
data = {
    'name': 'Alice',
    'age': 30,
    'skills': ['Python', 'SQL', 'Tableau'],
    'is_analyst': True
}

with open('output.json', 'w') as jsonfile:
    json.dump(data, jsonfile, indent=4)  # indent for pretty printing

# Converting JSON string to Python object
json_string = '{"name": "Bob", "age": 25}'
person = json.loads(json_string)
print(person['name'])  # Bob

# Converting Python object to JSON string
data = {'name': 'Charlie', 'age': 35}
json_string = json.dumps(data)
print(json_string)  # {"name": "Charlie", "age": 35}
```

## Error Handling

Proper error handling is crucial for robust data applications:

### Try-Except Blocks
```python
# Basic try-except
try:
    # Code that might raise an exception
    x = 10 / 0
except ZeroDivisionError:
    # Handle specific exception
    print("Cannot divide by zero!")

# Handling multiple exceptions
try:
    number = int(input("Enter a number: "))
    result = 10 / number
except ValueError:
    print("That's not a valid number!")
except ZeroDivisionError:
    print("Cannot divide by zero!")

# Catching any exception
try:
    # Risky code
    with open('nonexistent.txt', 'r') as file:
        content = file.read()
except Exception as e:
    print(f"An error occurred: {e}")

# Finally clause (always executes)
try:
    file = open('data.txt', 'r')
    # Process file...
except FileNotFoundError:
    print("File not found!")
finally:
    # This runs no matter what
    if 'file' in locals() and not file.closed:
        file.close()
        print("File closed")

# Else clause (runs if no exception)
try:
    number = int(input("Enter a number: "))
except ValueError:
    print("That's not a valid number!")
else:
    # This only runs if no exception occurred
    print(f"You entered {number}")
```

### Raising Exceptions
```python
def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero!")
    return a / b

try:
    result = divide(10, 0)
except ValueError as e:
    print(e)

# Custom exceptions
class InvalidDataError(Exception):
    """Exception raised for invalid data in analytics pipeline."""
    pass

def process_data(data):
    if not data:
        raise InvalidDataError("Data cannot be empty")
    # Process data...

try:
    process_data([])
except InvalidDataError as e:
    print(e)
```

## Modules and Packages

Python's module system helps organize and reuse code:

### Importing Modules
```python
# Import entire module
import math
print(math.sqrt(16))  # 4.0

# Import specific items
from math import sqrt, pi
print(sqrt(16))  # 4.0
print(pi)        # 3.141592653589793

# Import with alias
import math as m
print(m.sqrt(16))  # 4.0

# Import all (not recommended in production code)
from math import *
print(sqrt(16))  # 4.0
```

### Creating Modules

You can create your own modules by saving Python files and importing them:

```python
# utils.py
def square(x):
    return x ** 2

def cube(x):
    return x ** 3

# In another file
import utils
print(utils.square(4))  # 16
```

### Data Analytics Modules

Here are key modules for data analytics:

```python
# NumPy for numerical computing
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
mean = np.mean(arr)
std = np.std(arr)

# Pandas for data manipulation
import pandas as pd
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
filtered = df[df['A'] > 1]

# Matplotlib for visualization
import matplotlib.pyplot as plt
plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('Square Function')
plt.show()

# Seaborn for statistical visualization
import seaborn as sns
sns.set_theme()
iris = sns.load_dataset("iris")
sns.pairplot(iris, hue="species")
plt.show()

# Scikit-learn for machine learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LinearRegression()
model.fit(X_train, y_train)
```

## Next Steps

After mastering these Python basics, I recommend exploring:

1. **Data manipulation with Pandas**: The workhorse of data analytics
2. **Data visualization**: Matplotlib and Seaborn for creating informative charts
3. **Statistical analysis**: SciPy and StatsModels for statistical tests and modeling
4. **Machine learning**: Scikit-learn for predictive modeling
5. **Big data tools**: PySpark for handling large datasets
6. **Deep learning**: TensorFlow or PyTorch for neural networks
7. **Dashboard development**: Streamlit or Dash for creating interactive data applications

Each of these topics builds on the Python fundamentals covered in this guide and opens new possibilities for data analytics.

Happy coding and data analyzing!
