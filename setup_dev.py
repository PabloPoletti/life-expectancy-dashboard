#!/usr/bin/env python3
"""
Development Environment Setup Script
Automatically sets up the development environment for the Life Expectancy Dashboard.
"""

import subprocess
import sys
import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_command(command, description="", check=True):
    """Run a shell command and handle errors."""
    logger.info(f"🔧 {description}")
    logger.info(f"Running: {command}")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=check,
            capture_output=True,
            text=True
        )
        
        if result.stdout:
            logger.info(f"Output: {result.stdout.strip()}")
        
        return result
    
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Command failed: {e}")
        if e.stderr:
            logger.error(f"Error: {e.stderr.strip()}")
        if not check:
            return e
        sys.exit(1)


def check_python_version():
    """Check if Python version is compatible."""
    logger.info("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major != 3 or version.minor < 9:
        logger.error(f"❌ Python 3.9+ required, found {version.major}.{version.minor}")
        sys.exit(1)
    
    logger.info(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")


def setup_virtual_environment():
    """Set up Python virtual environment."""
    venv_path = Path("venv")
    
    if venv_path.exists():
        logger.info("📦 Virtual environment already exists")
        return
    
    logger.info("📦 Creating virtual environment...")
    run_command(f"{sys.executable} -m venv venv", "Creating virtual environment")
    
    # Activate virtual environment
    if os.name == 'nt':  # Windows
        activate_cmd = r"venv\Scripts\activate"
        pip_cmd = r"venv\Scripts\pip"
    else:  # Unix/Linux/MacOS
        activate_cmd = "source venv/bin/activate"
        pip_cmd = "venv/bin/pip"
    
    logger.info("✅ Virtual environment created")
    return pip_cmd


def install_dependencies(pip_cmd="pip"):
    """Install project dependencies."""
    logger.info("📚 Installing dependencies...")
    
    # Upgrade pip first
    run_command(f"{pip_cmd} install --upgrade pip", "Upgrading pip")
    
    # Install main dependencies
    run_command(f"{pip_cmd} install -r requirements.txt", "Installing main dependencies")
    
    # Install development dependencies
    dev_packages = [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "black>=22.0.0",
        "isort>=5.0.0", 
        "flake8>=5.0.0",
        "bandit>=1.7.0",
        "safety>=2.0.0",
        "pre-commit>=3.0.0",
    ]
    
    dev_packages_str = " ".join(dev_packages)
    run_command(f"{pip_cmd} install {dev_packages_str}", "Installing development dependencies")
    
    logger.info("✅ Dependencies installed")


def setup_pre_commit():
    """Set up pre-commit hooks."""
    logger.info("🔧 Setting up pre-commit hooks...")
    
    if not Path(".pre-commit-config.yaml").exists():
        logger.warning("⚠️ .pre-commit-config.yaml not found, skipping pre-commit setup")
        return
    
    run_command("pre-commit install", "Installing pre-commit hooks")
    run_command("pre-commit install --hook-type commit-msg", "Installing commit message hooks")
    
    logger.info("✅ Pre-commit hooks installed")


def setup_git_hooks():
    """Set up additional git hooks."""
    logger.info("🔧 Setting up git hooks...")
    
    git_hooks_dir = Path(".git/hooks")
    if not git_hooks_dir.exists():
        logger.warning("⚠️ Not in a git repository, skipping git hooks setup")
        return
    
    # Create pre-push hook for running tests
    pre_push_hook = git_hooks_dir / "pre-push"
    pre_push_content = '''#!/bin/sh
# Pre-push hook to run tests
echo "🧪 Running tests before push..."
python -m pytest tests/ -v --tb=short
if [ $? -ne 0 ]; then
    echo "❌ Tests failed. Push aborted."
    exit 1
fi
echo "✅ Tests passed. Proceeding with push."
'''
    
    with open(pre_push_hook, 'w') as f:
        f.write(pre_push_content)
    
    # Make executable
    os.chmod(pre_push_hook, 0o755)
    
    logger.info("✅ Git hooks configured")


def create_env_file():
    """Create .env file from template."""
    logger.info("📄 Setting up environment file...")
    
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if env_file.exists():
        logger.info("📄 .env file already exists")
        return
    
    env_content = '''# Life Expectancy Dashboard Environment Configuration

# Data Settings
DATA_UPDATE_FREQUENCY_HOURS=24
CACHE_TTL_SECONDS=3600

# Model Settings
ENABLE_MODEL_OPTIMIZATION=true
OPTIMIZATION_TRIALS=30
DEFAULT_PREDICTION_YEARS=10

# Streamlit Settings
STREAMLIT_THEME_PRIMARY_COLOR=#FF6B6B
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_HEADLESS=false

# Development Settings
DEBUG=false
LOG_LEVEL=INFO

# External APIs (if needed)
# WORLD_BANK_API_KEY=your_api_key_here
'''
    
    with open(env_file, 'w') as f:
        f.write(env_content)
    
    logger.info("✅ .env file created")


def run_initial_tests():
    """Run initial tests to verify setup."""
    logger.info("🧪 Running initial tests...")
    
    # Test data fetcher
    logger.info("Testing data fetcher...")
    result = run_command("python src/data_fetcher.py", "Testing data fetcher", check=False)
    
    if result.returncode != 0:
        logger.warning("⚠️ Data fetcher test had issues, but continuing...")
    else:
        logger.info("✅ Data fetcher test passed")
    
    # Test if we can import modules
    test_imports = [
        "import streamlit",
        "import pandas",
        "import plotly",
        "import sklearn",
        "import xgboost",
        "import lightgbm",
    ]
    
    for import_stmt in test_imports:
        try:
            exec(import_stmt)
            logger.info(f"✅ {import_stmt} - OK")
        except ImportError as e:
            logger.error(f"❌ {import_stmt} - Failed: {e}")
    
    logger.info("✅ Initial tests completed")


def display_next_steps():
    """Display next steps for the user."""
    logger.info("\n🎉 Development environment setup completed!")
    logger.info("\n📋 Next steps:")
    logger.info("1. Activate virtual environment:")
    
    if os.name == 'nt':  # Windows
        logger.info("   venv\\Scripts\\activate")
    else:  # Unix/Linux/MacOS
        logger.info("   source venv/bin/activate")
    
    logger.info("2. Run the dashboard:")
    logger.info("   python run_dashboard.py")
    logger.info("   or")
    logger.info("   streamlit run app.py")
    
    logger.info("3. Run tests:")
    logger.info("   pytest tests/ -v")
    
    logger.info("4. Code formatting:")
    logger.info("   black .")
    logger.info("   isort .")
    
    logger.info("5. Run pre-commit checks:")
    logger.info("   pre-commit run --all-files")
    
    logger.info("\n🌐 Dashboard will be available at: http://localhost:8501")
    logger.info("📖 See README.md for more information")


def main():
    """Main setup function."""
    logger.info("🚀 Setting up Life Expectancy Dashboard development environment")
    logger.info("=" * 70)
    
    try:
        # Basic checks
        check_python_version()
        
        # Setup steps
        pip_cmd = setup_virtual_environment()
        install_dependencies(pip_cmd or "pip")
        setup_pre_commit()
        setup_git_hooks()
        create_env_file()
        
        # Verification
        run_initial_tests()
        
        # Final instructions
        display_next_steps()
        
        logger.info("\n✅ Setup completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("\n⏹️ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
