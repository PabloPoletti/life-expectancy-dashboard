"""
Life Expectancy Dashboard Launcher
Convenient script to launch the Streamlit dashboard with proper configuration.
"""

import subprocess
import sys
import os
from pathlib import Path


def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        'streamlit', 'pandas', 'plotly', 'scikit-learn', 
        'xgboost', 'lightgbm', 'prophet', 'optuna'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Please run: pip install -r requirements.txt")
        return False
    
    return True


def setup_environment():
    """Setup environment variables and paths."""
    # Add current directory to Python path
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    
    # Set environment variables for better performance
    os.environ['PYTHONPATH'] = str(current_dir)
    os.environ['STREAMLIT_THEME_BASE'] = 'light'
    os.environ['STREAMLIT_THEME_PRIMARY_COLOR'] = '#FF6B6B'


def launch_dashboard():
    """Launch the Streamlit dashboard."""
    print("🚀 Launching Life Expectancy Dashboard...")
    print("📊 Features:")
    print("  • Global life expectancy trends")
    print("  • Country-specific analysis")
    print("  • Machine learning predictions")
    print("  • Interactive data exploration")
    print("\n🌐 The dashboard will open in your default web browser.")
    print("📍 URL: http://localhost:8501")
    print("\n⏹️  Press Ctrl+C to stop the server\n")
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port=8501",
            "--server.headless=false",
            "--browser.gatherUsageStats=false"
        ], cwd=Path(__file__).parent)
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped. Thank you for using the Life Expectancy Dashboard!")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")


def main():
    """Main function to run the launcher."""
    print("🌍 Life Expectancy Dashboard Launcher")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Setup environment
    setup_environment()
    
    # Launch dashboard
    launch_dashboard()


if __name__ == "__main__":
    main()
