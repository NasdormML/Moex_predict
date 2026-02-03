import subprocess
import sys

def update_all():
    """Обновить все пакеты до последних версий"""
    result = subprocess.run([sys.executable, "-m", "pip", "list", "--outdated", "--format=json"], 
                          capture_output=True, text=True)
    
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", 
                   "pandas", "numpy", "scikit-learn", "torch", "mlflow", 
                   "fastapi", "uvicorn", "pydantic"], check=False)
    
    subprocess.run([sys.executable, "-m", "pip", "freeze", ">", "requirements.txt"], 
                  shell=True)

if __name__ == "__main__":
    update_all()