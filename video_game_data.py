import pandas as pd
from google import genai
import re
from dotenv import load_dotenv
import os
import time
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("video_game_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Configure the API key
try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Please set GOOGLE_API_KEY in your .env file.")
    genai_client = genai.Client(api_key=api_key)
    logger.info("API client configured successfully")
except Exception as e:
    logger.error(f"Error configuring API: {e}")
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
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded {len(df)} video game records.")
        
        # Basic data validation
        if 'game_title' not in df.columns:
            raise ValueError("CSV file must contain a 'game_title' column")
            
        # Remove duplicates and nulls
        original_count = len(df)
        df = df.drop_duplicates(subset=['game_title'])
        df = df[df['game_title'].notna()]
        if len(df) < original_count:
            logger.info(f"Removed {original_count - len(df)} duplicate or null entries")
            
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def classify_games(titles, client=None, batch_size=5, max_retries=3, retry_delay=5):
    """
    Classifies multiple video game titles for genre, short description, and player mode in a single API call per batch.
    
    Args:
        titles (list or pandas.Series): List or Series of video game titles to classify
        client (genai.Client, optional): The Gemini API client. If None, uses the global client.
        batch_size (int): Number of titles to process in each batch
        max_retries (int): Maximum number of retries for failed API calls
        retry_delay (int): Delay in seconds between retries
    
    Returns:
        tuple: Three lists containing genres, short descriptions, and player modes in order
    """
    if client is None:
        client_to_use = genai_client
    else:
        client_to_use = client
    
    if hasattr(titles, 'tolist'):
        titles = titles.tolist()
    
    genres, descriptions, player_modes = [], [], []
    failed_titles = []
    
    # Create progress bar
    pbar = tqdm(total=len(titles), desc="Classifying games")
    
    for i in range(0, len(titles), batch_size):
        batch = titles[i:i+batch_size]
        
        prompt = """
        You are an expert in video games. Classify each of the following video games with the following attributes:
        - Genre (the primary one-word genre of the game. (e.g., RPG, Shooter, Puzzle, Strategy, Fighting, Horror, Sports, Adventure, Racing, Platformer, Simulation, MOBA, Battle Royale, etc.))
        - Short description (a concise (under 30 words) and engaging description that summarizes the core gameplay and setting of the game.)
        - Player mode (respond with ONLY one of these three options: 'Singleplayer', 'Multiplayer', or 'Both'. A game should be classified as 'Both' if it includes any cooperative or online multiplayer features, even if primarily designed for single-player.)
        
        Respond in the format:
        Game Title | Genre | Short Description | Player Mode
        
        """
        prompt += "\n".join(batch) + "\n"
        
        # Implement retry logic
        for attempt in range(max_retries):
            try:
                response = client_to_use.models.generate_content(
                    model='gemini-2.0-flash',
                    contents=prompt
                )
                
                lines = response.text.strip().split('\n')
                batch_genres, batch_descriptions, batch_modes = [], [], []
                
                for line in lines:
                    parts = line.split('|')
                    if len(parts) == 4:
                        title, genre, desc, mode = [p.strip() for p in parts]
                        batch_genres.append(genre)
                        batch_descriptions.append(desc)
                        batch_modes.append(mode)
                
                # Verify we got responses for all titles in the batch
                if len(batch_genres) != len(batch):
                    logger.warning(f"Received {len(batch_genres)} classifications for {len(batch)} titles in batch")
                    # Fill missing entries with placeholders
                    while len(batch_genres) < len(batch):
                        batch_genres.append("Unknown")
                        batch_descriptions.append("No description available")
                        batch_modes.append("Unknown")
                
                genres.extend(batch_genres)
                descriptions.extend(batch_descriptions)
                player_modes.extend(batch_modes)
                
                # Update progress bar
                pbar.update(len(batch))
                break  # Success, exit retry loop
                
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"API call failed (attempt {attempt+1}/{max_retries}): {e}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"API call failed after {max_retries} attempts: {e}")
                    # Add placeholder data for failed batch
                    for _ in range(len(batch)):
                        genres.append("API Error")
                        descriptions.append("Classification failed")
                        player_modes.append("Unknown")
                    failed_titles.extend(batch)
                    pbar.update(len(batch))
        
        # Rate limiting
        time.sleep(4)  # Avoid exceeding 15 requests per minute
    
    pbar.close()
    
    if failed_titles:
        logger.warning(f"Failed to classify {len(failed_titles)} titles")
    
    return genres, descriptions, player_modes

def save_results(df, output_file):
    """
    Save the enhanced dataframe to CSV with error handling.
    
    Args:
        df (pandas.DataFrame): The dataframe to save
        output_file (str): Path to save the CSV file
    """
    try:
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        df.to_csv(output_file, index=False)
        logger.info(f"Results saved successfully to {output_file}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        # Try to save to a backup location
        backup_file = "enhanced_video_games_backup.csv"
        try:
            df.to_csv(backup_file, index=False)
            logger.info(f"Results saved to backup file: {backup_file}")
        except:
            logger.critical("Failed to save results to backup location")

def main():
    """
    Main function to orchestrate the video game data enhancement process.
    """
    try:
        # File paths
        input_file = "c:\\Users\\hariz\\projects\\eklipse_video_game_data\\Game Thumbnail.csv"
        output_file = "c:\\Users\\hariz\\projects\\eklipse_video_game_data\\enhanced_video_games.csv"
        
        # Load data
        df = load_video_game_data(input_file)
        
        # Check if we have data to process
        if len(df) == 0:
            logger.warning("No valid game titles to process")
            return
            
        # Apply classification
        logger.info(f"Starting classification of {len(df)} game titles")
        start_time = time.time()
        df['genre'], df['short_description'], df['player_mode'] = classify_games(df['game_title'])
        
        # Save the results
        save_results(df, output_file)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Classification complete. Processed {len(df)} games in {elapsed_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"An error occurred in the main process: {e}")

if __name__ == "__main__":
    main()