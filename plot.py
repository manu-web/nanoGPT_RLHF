import subprocess
import re
import matplotlib.pyplot as plt

# Initialize lists to store training and validation losses
training_losses = []
validation_losses = []

# Define the external command to run (replace with your command)
command = f"python train.py config/finetune_harrypotter_char.py"

# Use subprocess to run the command and capture stdout and stderr
proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# Regular expressions to match training and validation loss lines
train_loss_pattern = re.compile(r".*train loss (\d+\.\d+).*")
val_loss_pattern = re.compile(r".*val loss (\d+\.\d+).*")

# Function to extract loss from a line
def extract_loss(line, pattern):
    match = pattern.search(line)
    if match:
        return float(match.group(1))
    return None

# Process the stdout and stderr lines
for line in proc.stdout:
    training_loss = extract_loss(line, train_loss_pattern)
    validation_loss = extract_loss(line, val_loss_pattern)
    if training_loss is not None:
        training_losses.append(training_loss)
    if validation_loss is not None:
        validation_losses.append(validation_loss)

# Close the subprocess
proc.stdout.close()

# Create a plot
plt.figure(figsize=(10, 6))
plt.plot(training_losses, label='Training Loss', color='blue')
plt.plot(validation_losses, label='Validation Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.grid()

# Save the plot to a specified path
plot_path = '/content/drive/MyDrive/CS839_HW1/images/finetuning.png'
plt.savefig(plot_path)
plt.show()

print(f"Plot saved to {plot_path}")