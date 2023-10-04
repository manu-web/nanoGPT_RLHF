import subprocess
import re
import matplotlib.pyplot as plt

# Define the range of heads and layers to vary in multiples of 2
min_heads, max_heads = 4, 16
min_layers, max_layers = 4, 16

# Initialize dictionaries to store results
results = {}

# Iterate over different combinations of heads and layers
for heads in range(min_heads, max_heads + 1, 2):
    for layers in range(min_layers, max_layers + 1, 2):
        # Construct the command to run your model (replace with your actual command)
        command = f"python train.py config/train_shakespeare_char.py --n_layer={layers} --n_head={heads}"

        # Run the command and capture its output
        try:
            output = subprocess.check_output(command, shell=True, text=True)
            
            # Use regular expressions to extract training and validation loss
            training_loss = re.findall(r".*train loss \d+\.\d+.*", output)
            validation_loss = re.findall(r".*val loss \d+\.\d+.*", output)
            
            if training_loss and validation_loss:
                # Store the results for this combination
                results[(heads, layers)] = {
                    "Training Loss": [float(loss) for loss in training_loss],
                    "Validation Loss": [float(loss) for loss in validation_loss]
                }
            
        except subprocess.CalledProcessError as e:
            print(f"Error running command: {command}")
            print(e)

# Create a big graph with subgraphs for different combinations
plt.figure(figsize=(12, 8))

for (heads, layers), data in results.items():
    training_loss = data["Training Loss"]
    validation_loss = data["Validation Loss"]
    
    # Plot training loss and validation loss with different colors
    plt.plot(range(len(training_loss)), training_loss, label=f"Heads={heads}, Layers={layers} (Train)")
    plt.plot(range(len(validation_loss)), validation_loss, label=f"Heads={heads}, Layers={layers} (Validation)")

# Add labels and legends
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Variation with Different Head and Layer Combinations")
plt.legend()

# Show the plot with a tight layout
plt.grid(True)
plt.tight_layout()

# Adjust the aspect ratio to fit the graphs to scale
plt.gca().set_aspect('auto')

# Show the plot
plt.savefig('/content/drive/MyDrive/sample_run.png', bbox_inches='tight')
