# Video Game Data Classifier
This project uses Google's Gemini AI to classify video games by genre, provide short descriptions, and identify player modes (singleplayer, multiplayer, or both).

## Prerequisites
- Python 3.8 or higher
- Google Gemini API key

## Installation
1. Clone or download this repository to your local machine.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
3. Create a .env file in the project root directory with your Google API key:
```plaintext
GOOGLE_API_KEY=your_api_key_here
 ```
4. The script will:

    - Load the video game data from the CSV file
    - Process the game titles in batches
    - Classify each game by genre, description, and player mode
    - Save the enhanced data to a new CSV file
    - Progress will be displayed in the console with a progress bar, and detailed logs will be saved to video_game_processing.log.

## Output
The script generates an enhanced CSV file with the following additional columns:

- genre : The primary genre of the game (e.g., RPG, Shooter, Puzzle)
- short_description : A concise description of the game (under 30 words)
- player_mode : Whether the game is Singleplayer, Multiplayer, or Both