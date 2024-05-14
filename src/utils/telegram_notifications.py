import requests
import traceback
import os

class TelegramNotifier:
    def __init__(self, bot_token, chat_id):
        self.bot_token = bot_token
        self.chat_id = chat_id

    def send_message(self, message):
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        data = {"chat_id": self.chat_id, "text": message}
        response = requests.post(url, data=data)
        return response.json()

    def notify_error(self, error_message):
        self.send_message(f"Error occurred:\n{error_message}\n\nTraceback:\n{traceback.format_exc()}")

    def upload_file(self, file_path, caption=None):
        url = f"https://api.telegram.org/bot{self.bot_token}/sendDocument"
        files = {"document": open(file_path, "rb")}
        data = {"chat_id": self.chat_id}
        if caption:
            data["caption"] = caption
        response = requests.post(url, files=files, data=data)
        return response.json()

    def upload_pdf_files(self, folder_path):
        pdf_files = [file for file in os.listdir(folder_path) if file.endswith(".pdf")]

        for pdf_file in pdf_files:
            file_path = os.path.join(folder_path, pdf_file)
            self.upload_file(file_path, caption=f"PDF: {pdf_file}")


"""

####################
#       TEST       #
####################

if __name__ == "__main__":
    bot_token = "7001559566:AAHNoseOCknt2EVVP20jP4JlcJ9h2V0puHE"
    chat_id = "-4172496113"

    notifier = TelegramNotifier(bot_token, chat_id)

    try:
        # notifier.notify_script_finished(f"Script ran by {os.getlogin()} execution finished successfully!")
        notifier.upload_pdf_files('../figures/')
    except Exception as e:
        notifier.notify_error(str(e))
"""