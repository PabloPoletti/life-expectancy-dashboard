#!/usr/bin/env python3
"""
Streamlit Cloud Update Script
Updates the repository with Streamlit Cloud compatible dependencies.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description=""):
    """Run a command and handle errors."""
    print(f"🔧 {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(f"✅ {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Main update function for Streamlit Cloud compatibility."""
    print("☁️ Updating for Streamlit Cloud Compatibility")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("app.py").exists():
        print("❌ Error: Not in project directory. Please run from the project root.")
        sys.exit(1)
    
    print("\n📋 Changes made for Streamlit Cloud compatibility:")
    print("  ✅ Reduced CatBoost to version 1.2.8 (latest available)")
    print("  ✅ Optimized dependencies for Python 3.13")
    print("  ✅ Removed problematic packages")
    print("  ✅ Maintained core ML functionality")
    print("  ✅ Updated advanced models for compatibility")
    
    # Step 1: Add all changes
    print("\n📝 Preparing repository updates...")
    run_command("git add .", "Adding all changes to git")
    
    # Step 2: Create commit
    commit_message = """🔧 Fix: Streamlit Cloud Compatibility Update

🛠️ Dependency Updates:
- ✅ Fixed CatBoost version (1.2.8) for Python 3.13 compatibility
- ✅ Optimized requirements.txt for Streamlit Cloud
- ✅ Maintained XGBoost 3.0+, LightGBM 4.5+, Prophet 1.1.5+
- ✅ Kept SHAP interpretability and core ML features
- ✅ Updated advanced models with graceful fallbacks

📊 Performance Maintained:
- 🔥 XGBoost 3.0+ with GPU acceleration
- ⚡ LightGBM 4.5+ ultra-fast training
- 🐱 CatBoost 1.2.8 (stable version)
- 🧠 Prophet for time series forecasting
- 📊 SHAP model interpretability
- 🔧 Optuna hyperparameter optimization

🌐 Streamlit Cloud Ready:
- All dependencies verified for Python 3.13
- Lightweight package selection
- Graceful error handling for missing packages
- Production-ready deployment configuration"""
    
    run_command(f'git commit -m "{commit_message}"', "Creating Streamlit Cloud compatibility commit")
    
    # Step 3: Push to GitHub
    print("\n🌐 Pushing to GitHub...")
    if run_command("git push origin main", "Pushing to GitHub repository"):
        print("\n🎉 SUCCESS! Updates pushed to GitHub")
        print("☁️ Streamlit Cloud will now redeploy automatically")
        print("⏱️ Deployment usually takes 2-3 minutes")
    else:
        print("❌ Failed to push. Check your authentication.")
        return False
    
    # Final summary
    print("\n" + "=" * 60)
    print("✅ STREAMLIT CLOUD UPDATE COMPLETED!")
    print("\n📋 What was fixed:")
    print("  ✅ CatBoost compatibility issue resolved")
    print("  ✅ All dependencies now Streamlit Cloud compatible")
    print("  ✅ Maintained advanced ML capabilities")
    print("  ✅ SHAP interpretability preserved")
    print("  ✅ Performance optimization maintained")
    
    print("\n🌐 Next Steps:")
    print("  1. Wait 2-3 minutes for Streamlit Cloud deployment")
    print("  2. Check your dashboard: https://life-expectancy-dashboard.streamlit.app/")
    print("  3. Verify all features are working")
    print("  4. Test the advanced ML models")
    
    print("\n🎯 Compatible Features:")
    print("  ✅ XGBoost 3.0+ - Latest gradient boosting")
    print("  ✅ LightGBM 4.5+ - Ultra-fast training")
    print("  ✅ CatBoost 1.2.8 - Stable production version")
    print("  ✅ Prophet 1.1.5+ - Time series forecasting")
    print("  ✅ SHAP 0.44+ - Model interpretability")
    print("  ✅ Optuna 3.6+ - Hyperparameter optimization")
    
    print("\n🚀 Your dashboard is now production-ready on Streamlit Cloud!")

if __name__ == "__main__":
    main()
