import pandas as pd

def adjust_pandas_display(max_rows=None, max_columns=None, width=1000):
    """
    Adjusts pandas display settings to control the number of rows, columns, and display width.

    Parameters:
        max_rows (int, optional): Maximum number of rows to display. Default is None (show all rows).
        max_columns (int, optional): Maximum number of columns to display. Default is None (show all columns).
        width (int): Width of the display in characters. Default is 1000.
    """
    pd.set_option('display.max_rows', max_rows)  # Show all rows or up to max_rows
    pd.set_option('display.max_columns', max_columns)  # Show all columns or up to max_columns
    pd.set_option('display.width', width)  # Adjust the display width
    print("Pandas display settings adjusted.")


def load_csv(filename):
    """
    Loads a CSV file into a DataFrame. Exits the program if the file is not found.

    Parameters:
        filename (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    try:
        data_frame = pd.read_csv(filename)
        print(f"File '{filename}' successfully loaded.")
        print(data_frame.info())
        print(data_frame.head())
        return data_frame
    except FileNotFoundError:
        print(f"File '{filename}' not found. Please check the path.")
        exit()
