# Application Documentation

## Run Locally

To run this application locally, follow these steps:

1. Create a Python virtual environment:
   ```bash
   python3.11 -m venv venv
   ```

2. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create the required directory structure:
   ```bash
   mkdir -p ignored
   ```

5. Place the required ROM files in the ignored directory:
   - Copy `dd.gb` to `ignored/dd.gb`
   - Copy `dd.gb.state` to `ignored/dd.gb.state`

6. Run the application:
   ```bash
   python3 run-ray-train.py
   ```

Note: Make sure you have Python 3.11 installed as this version is required for compatibility with all dependencies.
