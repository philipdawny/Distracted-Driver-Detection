import warnings
warnings.filterwarnings("ignore")

import kaggle
import os
from dotenv import load_dotenv
import subprocess

load_dotenv()
kaggle_username = os.getenv('KAGGLE_USERNAME')
kaggle_key = os.environ['KAGGLE_KEY']


def download_data():

    process = subprocess.run(['kaggle competitions download -c state-farm-distracted-driver-detection -p "../data"'], capture_output=True, text=True)

    print(r"\nData Downloaded\n\n")

    process = subprocess.run(['cd ..'], capture_output=True, text=True)

    print(r"\n Unzipping\n\n")

    process = subprocess.run(['unzip state-farm-distracted-driver-detection.zip'], capture_output=True, text=True)

    print(r"\n Unzipping complete")


    print(r"\nDeleting zip file")
    process = subprocess.run(['rm state-farm-distracted-driver-detection.zip'], capture_output=True, text=True)

    print(r"\nZip file deleted\n")

def main():
    download_data()


if __name__ == "__main__":
    main()