import subprocess
import re
import matplotlib.pyplot as plt

results = {}

heads = 6
layers = 6


file_path = f"/content/drive/MyDrive/CS839_HW1/training_info/{layers}_layers_{heads}_heads_training.txt"

with open(file_path, "r") as file:
    output = file.read()

print(output)
            
# Use regular expressions to extract training and validation loss
training_loss = re.findall(r".*train loss (\d+\.\d+).*", output)
validation_loss = re.findall(r".*val loss (\d+\.\d+).*", output)
            
if training_loss and validation_loss:
    # Store the results for this combination
    results[(heads, layers)] = {
        "Training Loss": [float(loss) for loss in training_loss],
        "Validation Loss": [float(loss) for loss in validation_loss]
    }
            
# Create a big graph with subgraphs for different combinations
plt.figure(figsize=(12, 8))

for (heads, layers), data in results.items():
    training_loss = data["Training Loss"]
    validation_loss = data["Validation Loss"]
    
    # Plot training loss and validation loss with different colors
    plt.plot(range(len(training_loss)), training_loss, label=f"Heads={heads}, Layers={layers} (Train)")
    plt.plot(range(len(validation_loss)), validation_loss, label=f"Heads={heads}, Layers={layers} (Validation)")

# Add labels and legends
plt.xlabel("Epochs in multiples of 50")
plt.ylabel("Loss")
plt.title("Loss Variation with Different Head and Layer Combinations")
#plt.legend()

# Show the plot with a tight layout
plt.grid(True)
plt.tight_layout()

# Adjust the aspect ratio to fit the graphs to scale
plt.gca().set_aspect('auto')

# Show the plot
plt.savefig(f"/content/drive/MyDrive/CS839_HW1/images/{layers}_layers_{heads}_heads.png", bbox_inches='tight')
