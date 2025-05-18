# Python Environment Setup with pyenv and pipenv

This README documents the setup process for creating a Python environment using pyenv and pipenv with Python 3.11.11 for this project.

## Prerequisites

Before getting started, ensure you have the following tools installed:

- Git
- Homebrew (for macOS users)
- Command-line tools for your operating system

## pyenv Installation and Setup

### 1. Install pyenv

#### macOS (using Homebrew)
```bash
brew update
brew install pyenv
```

#### Linux
```bash
curl https://pyenv.run | bash
```

### 2. Add pyenv to your shell configuration

Add the following to your shell configuration file (`.bashrc`, `.zshrc`, etc.):

```bash
# For bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc

# For zsh
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
```

Restart your shell or source your configuration file:
```bash
source ~/.bashrc  # or source ~/.zshrc
```

### 3. Install Python 3.11.11 using pyenv

Install required dependencies for building Python:

#### macOS
```bash
brew install openssl readline sqlite3 xz zlib
```

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install -y make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev \
libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl
```

Install Python 3.11.11:
```bash
pyenv install 3.11.11
```

### 4. Set Python 3.11.11 as the local version for this project

Navigate to your project directory and run:
```bash
cd /path/to/project
pyenv local 3.11.11
```

This creates a `.python-version` file in your project directory.

## pipenv Setup and Configuration

### 1. Install pipenv

With Python 3.11.11 active from pyenv, install pipenv:
```bash
pip install pipenv
```

### 2. Initialize the pipenv environment

```bash
# Create a new virtual environment using the Python version managed by pyenv
pipenv --python 3.11.11
```

### 3. Install project dependencies

```bash
# For production dependencies
pipenv install package_name

# For development dependencies
pipenv install --dev package_name
```

### 4. Generate lock file

Ensure all dependencies are properly locked:
```bash
pipenv lock
```

### 5. Activate the virtual environment

```bash
pipenv shell
```

## Using the Environment

### Running Python code in the environment

```bash
# With the environment activated
python your_script.py

# Without activating the environment
pipenv run python your_script.py
```

### Managing dependencies

- Add a new package: `pipenv install package_name`
- Add a development package: `pipenv install --dev package_name`
- Update all packages: `pipenv update`
- Remove a package: `pipenv uninstall package_name`

## Additional Configuration

### Project-specific settings

- The `.python-version` file specifies Python 3.11.11 for this project
- The `Pipfile` and `Pipfile.lock` track dependencies
- Use `pipenv graph` to visualize dependencies

### IDE Integration

If you're using an IDE like VSCode, configure it to use the Python interpreter from your pipenv environment:

```bash
# Get the path to your pipenv Python interpreter
pipenv --py
```

Then configure your IDE to use this Python interpreter path.

## Troubleshooting

- If pyenv cannot find Python 3.11.11, check available versions with `pyenv install --list`
- If pipenv is slow, set `PIPENV_VENV_IN_PROJECT=1` to keep the virtual environment in your project directory
- For permission issues, avoid using `sudo` with pipenv commands

