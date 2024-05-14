import json
import utils.telegram_notifications as tel_notif
import time
import sys
import subprocess
import os

def run_python_script(script_path):
    try:
        subprocess.run(["python", script_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python run_script.py <script_path>")
        sys.exit(1)
    
    script_path = sys.argv[1]

    config_file = "../telegram_config"
    with open(config_file, "r") as f:
        config = json.load(f)

    notifier = tel_notif.TelegramNotifier(config["bot_token"], config["chat_id"])

    try:
        start_time = time.time()
        run_python_script(script_path)
        execution_time = time.time() - start_time
        notifier.send_message(f"Script ran by {os.getlogin()} execution finished successfully in {execution_time} seconds.")
        notifier.upload_pdf_files('../figures/')
    except Exception as e:
        notifier.notify_error(str(e))