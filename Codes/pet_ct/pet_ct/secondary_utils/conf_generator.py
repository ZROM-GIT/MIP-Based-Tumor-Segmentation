import tkinter as tk
from tkinter import messagebox, filedialog
import yaml
from collections import OrderedDict

# Function to generate YAML file
def generate_yaml(data, filename='config.yaml'):
    with open(filename, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)
    messagebox.showinfo("Success", f"Configuration saved to {filename}")

# Function to load YAML configuration file
def load_configuration():
    file_path = filedialog.askopenfilename(filetypes=[("YAML files", "*.yaml")])
    if file_path:
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
            # Populate the fields with the loaded data
            resume_training_var.set(config.get('resume_training', False))
            file_path_entry.delete(0, tk.END)
            file_path_entry.insert(0, config.get('json_path', ''))
            update_extra_options_visibility()

# Function to handle submission of the form
def submit_form():
    # Collect data from input fields into a dictionary
    data = {
        'resume_training': resume_training_var.get(),
        'json_path': file_path_entry.get(),
        'option3': option3_var.get(),
        'text_option1': text_option1_entry.get(),
        'text_option2': text_option2_entry.get(),
        'extra_option1': extra_option1_entry.get() if extra_option1_entry.winfo_ismapped() else None,
        'extra_option2': extra_option2_entry.get() if extra_option2_entry.winfo_ismapped() else None
    }
    generate_yaml(data)

# Function to update visibility of extra options based on resume_training_var
def update_extra_options_visibility():
    if resume_training_var.get():
        extra_option1_label.grid(row=6, column=0, sticky="w", pady=2)
        extra_option1_entry.grid(row=6, column=1, columnspan=2, sticky="ew", pady=2)
        extra_option2_label.grid(row=7, column=0, sticky="w", pady=2)
        extra_option2_entry.grid(row=7, column=1, columnspan=2, sticky="ew", pady=2)
    else:
        extra_option1_label.grid_forget()
        extra_option1_entry.grid_forget()
        extra_option2_label.grid_forget()
        extra_option2_entry.grid_forget()

# Create the main application window
root = tk.Tk()
root.title("YAML Configuration Generator")
root.geometry("500x400")

# Create a main frame
main_frame = tk.Frame(root, padx=10, pady=10)
main_frame.pack(fill=tk.BOTH, expand=True)

# Create a frame for options
options_frame = tk.Frame(main_frame)
options_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

# Create and place resume_training (check button for boolean True/False)
tk.Label(options_frame, text="Resume Training:").grid(row=0, column=0, sticky="w", pady=2)
resume_training_var = tk.BooleanVar()
resume_training_var.set(True)  # Default value
tk.Checkbutton(options_frame, variable=resume_training_var).grid(row=0, column=1, columnspan=2, sticky="w", pady=2)

# Create and place json_path (text entry for file with browse button)
tk.Label(options_frame, text="JSON Path:").grid(row=1, column=0, sticky="w", pady=2)
file_path_entry = tk.Entry(options_frame)
file_path_entry.grid(row=1, column=1, sticky="ew", pady=2)

def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
    if file_path:
        file_path_entry.delete(0, tk.END)
        file_path_entry.insert(0, file_path)

browse_button = tk.Button(options_frame, text="Browse", command=browse_file)
browse_button.grid(row=1, column=2, padx=5, pady=2)

# Create and place Option 3 (check button)
tk.Label(options_frame, text="Option 3:").grid(row=2, column=0, sticky="w", pady=2)
option3_var = tk.BooleanVar()
option3_var.set(True)  # Default value
tk.Checkbutton(options_frame, variable=option3_var).grid(row=2, column=1, columnspan=2, sticky="w", pady=2)

# Create and place Text Option 1
tk.Label(options_frame, text="Text Option 1:").grid(row=3, column=0, sticky="w", pady=2)
text_option1_entry = tk.Entry(options_frame)
text_option1_entry.grid(row=3, column=1, columnspan=2, sticky="ew", pady=2)

# Create and place Text Option 2
tk.Label(options_frame, text="Text Option 2:").grid(row=4, column=0, sticky="w", pady=2)
text_option2_entry = tk.Entry(options_frame)
text_option2_entry.grid(row=4, column=1, columnspan=2, sticky="ew", pady=2)

# Create placeholders for extra options (initially hidden)
extra_option1_label = tk.Label(options_frame, text="Extra Option 1:")
extra_option1_entry = tk.Entry(options_frame)
extra_option2_label = tk.Label(options_frame, text="Extra Option 2:")
extra_option2_entry = tk.Entry(options_frame)

# Create a frame for buttons
buttons_frame = tk.Frame(main_frame)
buttons_frame.pack(fill=tk.X)

# Create and place the load configuration button
load_button = tk.Button(buttons_frame, text="Load Configuration", command=load_configuration)
load_button.pack(side=tk.LEFT, padx=5, pady=5)

# Create and place the submit button
submit_button = tk.Button(buttons_frame, text="Generate YAML", command=submit_form)
submit_button.pack(side=tk.RIGHT, padx=5, pady=5)

# Run the application
root.mainloop()
