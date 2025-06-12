# Natural Language Processing


# Navigate to or create the desired project folder
mkdir myproject         # Create the folder (skip if it already exists)
cd myproject            # Move into the project folder

# Create a virtual environment named 'myvenv' inside this folder
python -m venv myvenv

# Activate the virtual environment on Unix/Linux/MacOS
source myvenv/bin/activate

# (Alternative) Activate the virtual environment on Windows (use this instead if on Windows)
myvenv\Scripts\activate

# Optional: Check currently installed packages (should be minimal at this point)
pip list

# Install all required packages from requirements.txt (this file should be inside 'myproject')
pip install -r requirements.txt
