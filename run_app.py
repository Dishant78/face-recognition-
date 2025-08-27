    #!/usr/bin/env python3
"""
Simple startup script for the Missing Person Face Detection System
This script ensures all dependencies are available and starts the web application
"""

import sys
import os
import subprocess

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'flask',
        'opencv-python',
        'numpy',
        'pillow'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} is missing")
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                print(f"✓ Installed {package}")
            except subprocess.CalledProcessError:
                print(f"✗ Failed to install {package}")
                return False
    
    return True

def check_database():
    """Check if database and required files exist"""
    if not os.path.exists('data'):
        os.makedirs('data')
        print("✓ Created data directory")
    
    if not os.path.exists('data/missing_persons.db'):
        print("✓ Database will be created on first run")
    
    return True

def main():
    """Main startup function"""
    print("=" * 60)
    print("Missing Person Face Detection System")
    print("=" * 60)
    
    # Check dependencies
    print("\n1. Checking dependencies...")
    if not check_dependencies():
        print("Error: Failed to install required packages")
        return False
    
    # Check database
    print("\n2. Checking database...")
    if not check_database():
        print("Error: Database setup failed")
        return False
    
    # Check if dishant.jpg exists
    if os.path.exists('data/dishant.jpg'):
        print("✓ Found dishant.jpg in database")
    else:
        print("⚠ Warning: dishant.jpg not found in data directory")
        print("   Make sure to add your face image to the database")
    
    print("\n3. Starting Flask application...")
    print("   Access the web interface at: http://localhost:5000")
    print("   Press Ctrl+C to stop the application")
    print("=" * 60)
    
    # Start the Flask app
    try:
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\nApplication stopped by user")
    except Exception as e:
        print(f"Error starting application: {e}")
        return False
    
    return True

if __name__ == '__main__':
    success = main()
    if not success:
        sys.exit(1)