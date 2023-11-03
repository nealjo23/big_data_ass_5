import pandas as pd
import os
from typing import List, Tuple
import matplotlib.pyplot as plt

"""
Cluster - Big Data and Machine Learning
Semester 2 2023
Assessment 5: Analysing Baby Names
John Neal
20003366

The program will first plot the top names for a given gender and 
year against either a year in the past or a year in the future.
This is to see how the names trended in the years leading up
to the chosen year, or how they trended in the years after.
The user selects the gender, the years and how many top names to show.

The program then shows a column graph for the top 5 boys and girls 
names in a given year entered by the user.

The plots are saved in .png format.  
"""


def load_and_combine_data(data_folder_path: str) -> pd.DataFrame:
    """
    Loads and combines data from CSV files in the specified folder.

    Args:
    data_folder_path (str): Path to the folder containing the CSV files.

    Returns:
    pandas.DataFrame: Combined DataFrame of all data.
    """

    def load_data(file_name: str) -> pd.DataFrame:
        """Loads a single file into a DataFrame and adds a year column."""
        file_year = int(file_name[3:7])  # Extract year from file name (e.g., 'yob1880.txt')
        file_path = os.path.join(data_folder_path, file_name)
        df = pd.read_csv(file_path, names=['name', 'sex', 'number'])
        df['year'] = file_year
        return df

    # Get all file names in the folder
    file_names: List[str] = os.listdir(data_folder_path)

    # Load all the files into DataFrames and combine them
    all_dfs: List[pd.DataFrame] = [load_data(file_name) for file_name in file_names]
    combined_df: pd.DataFrame = pd.concat(all_dfs, ignore_index=True)

    return combined_df


def get_valid_integer(prompt: str, min_value: int, max_value: int) -> int:
    """Function to get a valid integer input within a specified range."""
    while True:
        try:
            value = int(input(prompt))
            if min_value <= value <= max_value:
                return value
            else:
                print(f"Please enter a number between {min_value} and {max_value}.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")


def get_user_input() -> Tuple[str, int, int, int]:
    """
    Prompts the user for sex, sample year, comparison year, and number of names.

    Returns:
    Tuple[str, int, int, int]: A tuple containing the sex, sample year, comparison year, and number of top names.
    """
    # Prompt for sex
    _sex: str = input("Enter the sex (M/F): ").upper()
    while _sex not in ['M', 'F']:
        print("Invalid input. Please enter 'M' for male or 'F' for female.")
        _sex = input("Enter the sex (M/F): ").upper()

    # Prompt for the sample year with constraints and protection against non-numeric input
    _sample_year: int = get_valid_integer(
        "Enter the sample year (1880-2022): ", 1880, 2022)

    # Prompt for the comparison year with constraints and protection against non-numeric input
    _compare_year: int = get_valid_integer(
        "Enter the comparison year (different from sample year, 1880-2022): ", 1880, 2022)
    while _compare_year == _sample_year:
        print("Comparison year must be different from the sample year.")
        _compare_year = get_valid_integer(
            "Enter the comparison year (different from sample year, 1880-2022): ", 1880, 2022)

    # Prompt for the number of top names with constraints and protection against non-numeric input
    _num_names: int = get_valid_integer(
        "How many top names would you like to look at (1-100)? ", 1, 100)

    return _sex, _sample_year, _compare_year, _num_names


def prepare_data_for_plotting(_baby_names_df: pd.DataFrame,
                              _sex: str, _sample_year: int,
                              _compare_year: int, _num_names: int) -> pd.DataFrame:
    """
    Prepares the data for plotting the top baby names over a range of years.

    Args:
    _baby_names_df (pd.DataFrame): DataFrame containing the baby name data.
    _sex (str): Sex for which to filter the names ('M' or 'F').
    _sample_year (int): The year for which to find the top names.
    _compare_year (int): The year to compare with.
    _num_names (int): The number of top names to consider.

    Returns:
    pd.DataFrame: A DataFrame with ranks of top names over the specified years.
    """
    # Determine the range of years
    start_year = min(_sample_year, _compare_year)
    end_year = max(_sample_year, _compare_year)

    # Filter data for the given sex and years
    # Add .copy() to ensure it's an independent copy
    year_range_df = _baby_names_df[(_baby_names_df['sex'] == _sex) &
                                   (_baby_names_df['year'].between(start_year, end_year))].copy()

    # Calculate rank for each name per year for all names
    year_range_df['rank'] = \
        year_range_df.groupby('year')['number'].rank(method="min", ascending=False).astype(int)

    # Find top names in the sample year
    top_names = (year_range_df[year_range_df['year'] == _sample_year].nlargest(_num_names, 'number')['name'])

    # Filter data for only the top names
    top_names_df = year_range_df[year_range_df['name'].isin(top_names)]

    return top_names_df.pivot(index='year', columns='name', values='rank')


def plot_top_names(_plot_data: pd.DataFrame, _sex: str, _sample_year: int, _compare_year: int, _num_names: int):
    """
    Plots the top baby names over a range of years.

    Args:
    _plot_data (pd.DataFrame): DataFrame with the ranks of top names over the specified years.
    _sex (str): The gender for which the names are plotted ('M' for male, 'F' for female).
    _sample_year (int): The sample year.
    _compare_year (int): The comparison year.
    _num_names (int): The number of top names.
    """
    # Setting up the plot
    plt.figure(figsize=(10, 10))

    # Adjust the x-axis range to add a gap on the left for name labels
    year_range = _plot_data.index
    year_span = max(year_range) - min(year_range)
    plt.xlim(min(year_range) - 0.5, max(year_range) + 0.5)  # Add gap on both sides

    # Plot each name as a separate line with label to the left or right
    for name in _plot_data.columns:
        # Get data for this name, limit to top 50 ranks
        name_data = _plot_data[name].reindex(year_range).fillna(51)
        name_data = name_data.where(name_data <= 50, other=50)

        # Plot line for the name
        line, = plt.plot(year_range, name_data, marker='o')

        # Determine label position based on the comparison of years
        if _compare_year > _sample_year:
            # Label on the left with reduced gap
            label_x = year_range[0] - 0.1  # Reduced left side gap
            label_y = name_data.iloc[0]
            horizontal_align = 'right'
        else:
            # Label on the right
            label_x = year_range[-1] + 0.1  # Right side
            label_y = name_data.iloc[-1]
            horizontal_align = 'left'

        plt.text(label_x, label_y, name,
                 color=line.get_color(),
                 verticalalignment='center',
                 horizontalalignment=horizontal_align)

    # Inverting the y-axis and setting y-ticks to integer values
    plt.gca().invert_yaxis()
    plt.ylim(50, 0.5)  # Limit y-axis to top 50 ranks and add space above the top rank

    # Set y-ticks to integer values up to 50
    plt.yticks(list(range(1, 51)))

    # Determine tick spacing based on year span
    if year_span <= 10:
        x_tick_spacing = 1
    elif year_span <= 30:
        x_tick_spacing = 5
    else:
        x_tick_spacing = 10

    # Setting x-ticks with the determined spacing
    plt.xticks(list(range(min(year_range), max(year_range) + 1, x_tick_spacing)))

    # Adding labels and title
    plt.xlabel("Year")
    plt.ylabel("Position")
    plt.title(f"Top {_num_names} {'Boy' if _sex == 'M' else 'Girl'} Baby Names in {_sample_year}")

    # Save the plot to a file
    plt.savefig("top names over time.png", format='png')  # You can change the format to 'pdf', 'jpg', etc.

    # Show the plot
    plt.show()


def create_column_chart(_baby_names_df: pd.DataFrame, _year: int) -> None:
    """
    Creates and displays a column chart for the top 5 girl and boy baby names for a given year.

    Args:
    _baby_names_df (pd.DataFrame): The DataFrame containing baby names data.
    _year (int): The year for which the top baby names are to be displayed.
    """

    # Filter top 5 names for each gender
    top_boys = _baby_names_df[(_baby_names_df['year'] == _year) & (_baby_names_df['sex'] == 'M')].head(5)
    top_girls = _baby_names_df[(_baby_names_df['year'] == _year) & (_baby_names_df['sex'] == 'F')].head(5)

    # Prepare data for plotting
    names = [name for pair in zip(top_girls['name'], top_boys['name']) for name in pair]
    numbers = [number for pair in zip(top_girls['number'], top_boys['number']) for number in pair]
    colors = ['pink', 'blue'] * 5

    # Set up the plot
    plt.figure(figsize=(10, 6))

    # Create the column chart
    plt.bar(list(range(10)), numbers, color=colors)
    plt.xticks(list(range(10)),
               labels=[f'#{i // 2 + 1} Girl' if i % 2 == 0 else f'#{i // 2 + 1} Boy' for i in range(10)], rotation=45)

    # Add labels at the top of each column
    for i, (name, number) in enumerate(zip(names, numbers)):
        plt.text(i, number, name, ha='center', va='bottom')

    # Add labels and title
    plt.xlabel("Rank and Gender")
    plt.ylabel("Number of Occurrences")
    plt.title(f"Top 5 Girl and Boy Baby Names in {_year}")

    # Adjust layout to prevent clipping of x-label
    plt.tight_layout()

    # Save the plot to a file
    plt.savefig("top 5 names.png", format='png')  # You can change the format to 'pdf', 'jpg', etc.

    # Show the plot
    plt.show()


# Main script or main function
if __name__ == "__main__":
    folder_path = 'names_data'
    baby_names_df = load_and_combine_data(folder_path)

    # Get user input
    sex, sample_year, compare_year, num_names = get_user_input()

    plot_data = prepare_data_for_plotting(baby_names_df, sex, sample_year, compare_year, num_names)
    plot_top_names(plot_data, sex, sample_year, compare_year, num_names)

    # chart top 5 names for a given year. Prompt to egt the year.
    sample_year: int = get_valid_integer(
        "Enter a year to see top 5 boys and girls names (1880-2022): ", 1880, 2022)

    create_column_chart(baby_names_df, sample_year)
