import matplotlib.pyplot as plt
import numpy as np
import csv

# Function to read data from CSV file
def read_csv(filename):
    models = []
    original = []
    onnx = []
    gguf = []
    
    with open(filename, 'r') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            models.append(row['Base Model'])
            original.append(int(row['Original Downloads']))
            onnx.append(int(row['ONNX Downloads']))
            gguf.append(int(row['GGUF Downloads']))
    
    return models, original, onnx, gguf

# Read data from CSV file
models, original, onnx, gguf = read_csv('model_analysis_results.csv')

# Set up the plot
plt.figure(figsize=(19.2, 10.8))  # 1920x1080 pixels

# Create the bar chart
x = np.arange(len(models))
width = 0.25

plt.bar(x - width, original, width, label='Original Downloads', color='#8884d8')
plt.bar(x, onnx, width, label='ONNX Downloads', color='#82ca9d')
plt.bar(x + width, gguf, width, label='GGUF Downloads', color='#ffc658')

# Customize the plot
plt.xlabel('Models')
plt.ylabel('Number of Downloads')
plt.title('Model Download Comparison')
plt.xticks(x, models, rotation=45, ha='right')
plt.legend()

# Adjust layout and save
plt.tight_layout()
plt.savefig('chart.png', dpi=100)
plt.close()

print("Chart has been saved as 'chart.png'")