import os
import sys

python_exe = sys.executable

print("--- Starting Training Phase ---")
os.system(f'"{python_exe}" train.py')

print("--- Starting Evaluation Phase ---")
os.system(f'"{python_exe}" evaluate.py')

print("--- Starting XAI (Grad-CAM) Phase ---")
os.system(f'"{python_exe}" xai.py')

print("--- Entire Project Pipeline Completed Successfully! ---")
