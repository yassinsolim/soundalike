# spotify_program.py
# Yassin Soliman, ENDG 233 F23
# A terminal-based application to process and plot data based on given user input and provided data values.

import numpy as np
import matplotlib.pyplot as plt
import math

# ******************************************************************************************************
# Data is imported from the included csv file.
column_names = ['title', 'artist(s)', 'release', 'num_of_streams', 'bpm', 'key', 'mode', 'danceability', 'valence', 'energy', 'acousticness', 'instrumentalness', 'liveness', 'speechiness']
data = np.genfromtxt('spotify_data.csv', delimiter = ',', skip_header = True, dtype = str)
# ******************************************************************************************************


# ******************************************************************************************************
class Song:
    """
    Represents a Song object constructed from a single row of data.

    Attributes:
    - title (str): The title of the song.
    - artist (str): The artist(s) of the song.
    - release (str): The release date of the song.
    - num_of_streams (float): The number of streams for the song.
    - bpm (float): The beats per minute of the song.
    - key (str): The key of the song.
    - mode (str): The mode of the song.
    - danceability (float): The danceability score of the song.
    - valence (float): The valence score of the song.
    - energy (float): The energy score of the song.
    - acousticness (float): The acousticness score of the song.
    - instrumentalness (float): The instrumentalness score of the song.
    - liveness (float): The liveness score of the song.
    - speechiness (float): The speechiness score of the song.
    - percentages (list): A list containing danceability through speechiness values.
    """

    def __init__(self, data_row):
        """
        Initializes a Song object with data from a single row.

        Args:
        - data_row (list): A single row of data representing a song.

        Initializes attributes for title, artist, release date, stream count,
        BPM, key, mode, and various song feature scores as attributes of the Song object.
        Also computes a list of percentages for danceability through speechiness.
        """
        self.title = data_row[0]
        self.artist = data_row[1]
        self.release = data_row[2]
        self.num_of_streams = float(data_row[3])
        self.bpm = float(data_row[4])
        self.key = data_row[5]
        self.mode = data_row[6]
        self.danceability = float(data_row[7])
        self.valence = float(data_row[8])
        self.energy = float(data_row[9])
        self.acousticness = float(data_row[10])
        self.instrumentalness = float(data_row[11])
        self.liveness = float(data_row[12])
        self.speechiness = float(data_row[13])
        self.percentages = [self.danceability, self.valence, self.energy, self.acousticness,
                            self.instrumentalness, self.liveness, self.speechiness]


# ******************************************************************************************************
def feature_stats(input_value):
    """
    Calculate the highest, lowest, and mean values for a selected feature column.

    Args:
    input_value (int): The index representing the column of the selected feature.

    Returns:
    int: The index of the highest value in the selected feature column.
    """
    feature_column = data[:, input_value].astype(int)
    highest_value = math.floor(np.max(feature_column))
    lowest_value = math.floor(np.min(feature_column))
    mean_value = math.floor(np.mean(feature_column))

    index_highest_value = np.argmax(feature_column)

    # Print statistics
    print(f"Highest Value: {highest_value}")
    print(f"Lowest Value: {lowest_value}")
    print(f"Mean Value: {mean_value}")

    return index_highest_value


def age_stats(input_value):
    """
    Calculate statistics related to the release years of songs.

    Args:
    input_value (int): The index representing the column containing release years.

    Prints:
    Prints the span of years, artist of the oldest song, key, and mode of the oldest song.
    """
    release_years = data[:, input_value].astype(int)
    min_year = np.min(release_years)
    max_year = np.max(release_years)
    span_of_years = max_year - min_year

    oldest_song_index = np.argmin(release_years)
    oldest_song_details = data[oldest_song_index]

    oldest_song_artist = oldest_song_details[1]
    oldest_song_key = oldest_song_details[5]
    oldest_song_mode = oldest_song_details[6]

    print(f"Span of years: {span_of_years}")
    print(f"Artist of oldest song: {oldest_song_artist}")
    print(f"Key and mode of oldest song: {oldest_song_key} {oldest_song_mode}")

# ******************************************************************************************************

def main():
    print("ENDG 233 Spotify Statistics\n")
    print("Song analysis options: ")
    for menu, option in enumerate(column_names):
        print(menu, option)
    print("Type -1 to end the program.")

    # User input for song feature analysis
    choice = int(input("\nPlease enter a song feature to analyze: "))

    # Loop until user exits (-1)
    while choice != -1:
        if choice == 2:
            # Calculate statistics for the age of songs
            age_stats(2)
        elif choice in [3, 4, 7, 8, 9, 10, 11, 12, 13]:
            # Calculate statistics for specific song features and find the title of the song with the highest value
            title_highest_value = data[feature_stats(choice), 0]
            print(f"Title of Song with Highest Value: {title_highest_value}")
        elif choice in [0, 1, 5, 6]:
            # Prompt for valid input for song feature analysis
            choice = int(input("\nPlease enter an actual song feature to analyze: "))
            continue
        else:
            # Notify user about invalid menu option
            print("You must enter a valid menu option.")

        # Continue input until user exits (-1)
        choice = int(input("\nPlease enter a song feature to analyze: "))

if __name__ == '__main__':
    main()

# Allow the user to select a row to create a Song object
row_number = int(input("Enter a song feature to visually analyze: "))
selected_song = Song(data[row_number])

# Plot percentage values of the selected song
plt.figure(figsize=(8, 6))
plt.bar(column_names[7:], selected_song.percentages)
plt.ylim(0, 100)  # Set y-axis limit to range [0, 100] for percentages
plt.title(f'Song Stats for {selected_song.title}')
plt.xlabel('Feature')
plt.ylabel('Percentage')
plt.tight_layout()
plt.savefig('selected_song_percentages.png')  # Save the plot as a .png file
plt.show()

# Create danceability vs. bpm plot
danceability = data[:, 7].astype(float)
bpm = data[:, 4].astype(float)

plt.figure(figsize=(8, 6))
plt.scatter(bpm, danceability, alpha=1, color='blue', label='Song Stats')  # Set alpha to 1 for solid dots
plt.title('Danceability vs. Beats per minute')
plt.xlabel('BPM')
plt.ylabel('Danceability')
plt.legend()
plt.grid(True)
plt.savefig('danceability_vs_bpm.png')
plt.show()


