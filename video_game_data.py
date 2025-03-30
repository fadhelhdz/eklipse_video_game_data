import pandas as pd
from google import genai
import re
from dotenv import load_dotenv
import os
import time

# Load environment variables from .env file
load_dotenv()

# Configure the API key
try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Please set GOOGLE_API_KEY in your .env file.")
    genai.Client(api_key=api_key)
except Exception as e:
    print(f"Error configuring API: {e}")
    raise

def load_video_game_data(file_path):
    """
    Load video game data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pandas.DataFrame: Loaded data
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {len(df)} video game records.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def classify_games(titles, client=None, batch_size=5):
    """
    Classifies multiple video game titles for genre, short description, and player mode in a single API call per batch.
    
    Args:
        titles (list or pandas.Series): List or Series of video game titles to classify
        client (genai.Client, optional): The Gemini API client. If None, uses the global client.
        batch_size (int): Number of titles to process in each batch
    
    Returns:
        tuple: Three lists containing genres, short descriptions, and player modes in order
    """
    if client is None:
        from google import genai
        client_to_use = genai.Client(api_key=api_key)
    else:
        client_to_use = client
    
    if hasattr(titles, 'tolist'):
        titles = titles.tolist()
    
    genres, descriptions, player_modes = [], [], []
    
    for i in range(0, len(titles), batch_size):
        batch = titles[i:i+batch_size]
        
        prompt = """
        Classify each of the following video games with three attributes:
        - Genre (one-word, e.g., RPG, Shooter, Puzzle)
        - Short description (under 30 words)
        - Player mode (Singleplayer, Multiplayer, Both)
        
        Respond in the format:
        Game Title | Genre | Short Description | Player Mode
        
        """
        prompt += "\n".join(batch) + "\n"
        
        response = client_to_use.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt
        )
        
        lines = response.text.strip().split('\n')
        
        for line in lines:
            parts = line.split('|')
            if len(parts) == 4:
                title, genre, desc, mode = [p.strip() for p in parts]
                genres.append(genre)
                descriptions.append(desc)
                player_modes.append(mode)
        
        time.sleep(4)  # Avoid exceeding 15 requests per minute
        
    return genres, descriptions, player_modes

def main():
    """
    Main function to orchestrate the video game data enhancement process.
    """
    # File paths
    input_file = "c:\\Users\\hariz\\projects\\eklipse_video_game_data\\Game Thumbnail.csv"
    output_file = "c:\\Users\\hariz\\projects\\eklipse_video_game_data\\enhanced_video_games.csv"
    
    # Load data
    df = load_video_game_data(input_file)
    # Apply classification
    df['genre'], df['short_description'], df['player_mode'] = classify_games(df['game_title'])
    # Save the results
    df.to_csv(output_file, index=False)
    print("Classification complete. Results saved to enhanced_video_games.csv")

if __name__ == "__main__":
    main()