![Logo](./app/static/LOGO.png)

# Audio Classifier Trainer

This project allows you to train an audio classifier using Random Forest on your custom dataset.

## Getting Started

Follow these instructions to set up the project on your local machine.

### Prerequisites

Ensure that you have Python 3.6 or higher and ngrok installed on your system.

### ngrok Configuration

Before running the application, make sure to configure ngrok by adding your authentication token. Open a terminal and run the following command:

```bash
ngrok config add-authtoken Your-Token
```

### Installation



1. Clone the repository:
   ```bash
   git clone https://github.com/Jss-on/Audio-Classifier-Trainer.git
   ```

2. Navigate to the project directory:
   ```bash
   cd Audio-Classifier-Trainer
   ```

3. Create a virtual environment:
   ```bash
   python3 -m venv myenv
   ```

4. Activate the virtual environment:
   ```bash
   source myenv/bin/activate
   ```

5. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
6. (Optional) Ngrok configuration
   ```bash
   ngrok config add-authtoken <Your-Token>
   ```
7. Run the application:
   ```bash
   python run.py
   ```

## Usage

Follow the on-screen instructions to upload your audio samples and train the model.

## License

This project is for View Only. You may look at the code, but you are not authorized to use, modify, or distribute the code or any of its components. For more details, refer to the LICENSE.md file.