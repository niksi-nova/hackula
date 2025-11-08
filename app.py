import subprocess, sys

# Force install dependencies from requirements.txt
subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "-r", "requirements.txt"])

# Then run your main app
import health_v4
