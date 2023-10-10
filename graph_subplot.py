import subprocess
import re
import matplotlib.pyplot as plt
from matplotlib.rcsetup import validate_any
import numpy as np

# Define the range of heads and layers to vary in multiples of 2
min_heads, max_heads = 6, 12
min_layers, max_layers = 6, 12

i = 0
j = 0

epochs = []
for val in range(0,5001,250):
    epochs.append(val)

fig, axs = plt.subplots(3, 4, figsize=(16, 12))

# Iterate over different combinations of heads and layers
for heads in range(min_heads, max_heads + 1, 2):
    if(heads == 10):
        continue

    j = 0
    for layers in range(min_layers, max_layers + 1, 2):

        file_path = f"/content/drive/MyDrive/CS839_HW1/training_info/{layers}_layers_{heads}_heads_training.txt"
        
        with open(file_path, "r") as file:
            output = file.read()

        print(output)
            
        # Use regular expressions to extract training and validation loss
        training_loss = re.findall(r".*train loss (\d+\.\d+).*", output)
        validation_loss = re.findall(r".*val loss (\d+\.\d+).*", output)
        
        axs[i, j].plot(epochs, [float(loss) for loss in training_loss], label=f"Training loss")
        axs[i, j].plot(epochs, [float(loss) for loss in validation_loss], label=f"Validation loss")
        axs[i, j].set_title(f"Heads = {heads}, Layers = {layers}")
        axs[i, j].set_xlabel('Epochs')
        axs[i, j].set_ylabel('Loss')
        #axs[i, j].invert_yaxis()
        axs[i, j].set_yticks(np.arange(0, 5, step=1))
        axs[i, j].legend(loc="upper right")
        j = j + 1

    i = i + 1
  
#plt.title("Loss Variation with different heads and layers combinations")
plt.tight_layout()
#plt.show()
#plt.gca().set_aspect('auto')
plt.savefig('/content/drive/MyDrive/CS839_HW1/images/hyperparameter.png', bbox_inches='tight')
