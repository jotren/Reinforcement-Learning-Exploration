import os
from tensorboard import program

current_dir = r'C:\projects\personal_projects\RL-Projects'
training_log_path = os.path.join(current_dir, 'training', 'logs', 'PPO_6')

print(training_log_path)

# Initialize a TensorBoard instance
tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', training_log_path, '--port', '6007'])  # Added the port argument here
url = tb.launch()

print(f"TensorBoard started and can be accessed here: {url}")
