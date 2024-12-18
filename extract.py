import mlbstatsapi
from datetime import *
import sqlite3

def played_last_three():
    """
    Identifies MLB players who appeared on team rosters in all of the most recent 
    four MLB seasons (2021–2024) and prints their names (up to 10 as a sample).
    """
    #initializing API client
    mlb = mlbstatsapi.Mlb()
    seasons = {2022, 2023, 2024}
    #dictionary to store mlb player-ids who have played games in 2021, 2022, 2023, and/or 2024
    season_players = {}
    for season in seasons:
        season_players[season] = set()
        teams = mlb.get_teams(season = season)
        for team in teams:
            team_id = team.id
            roster = mlb.get_team_roster(team_id=team_id, season=season)
            #addiing each "rostered" player to the dictionary
            for player in roster:
                player_id = player.id
                season_players[season].add(player_id)
    
    # Find the intersection of player sets across all seasons
    # This identifies players who were on a roster in every season
    players_all_seasons = set.intersection(*season_players.values())
    player_data = []
    for player_id in players_all_seasons:
        player_info = mlb.get_person(player_id)
        #filtering out pitchers (only want to analyze hitters)
        if(player_info.primaryposition.abbreviation != "P"):
            player_data.append((player_id, player_info.fullname))
    #storing dict. of players to DB
    save_players_to_db(player_data)


def save_players_to_db(player_names, db_name="mlb_players.db"):
    """
    Saves player data to an SQLite database.

    Parameters:
        player_names (list of tuples): List of tuples containing player IDs and names.
        db_name (str): Name of the SQLite database file.
    """
    # Connect to the SQLite database (or create it if it doesn't exist)
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Create the players table if it doesn't exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS players (
            player_id INTEGER PRIMARY KEY,
            fullname TEXT NOT NULL
        )
    """)

    # Insert player data into the table
    cursor.executemany("""
        INSERT OR IGNORE INTO players (player_id, fullname)
        VALUES (?, ?)
    """, player_names)

    # Commit the transaction and close the connection
    conn.commit()
    conn.close()
    print(f"Saved {len(player_names)} players to {db_name}.")


def hitting_stats():
    """
    This function retrieves the hitting statistics of all players from the 'players' table in the SQLite database 
    for the years 2022, 2023, and 2024. The function fetches the player's stats from the mlbstatsapi API 
    and stores them in a new SQLite database (mlb_2022-2024_stats.db).

    Steps:
        1. Initializes the mlbstatsapi API client.
        2. Connects to the SQLite database to fetch player IDs.
        3. Retrieves the player stats for the years 2022, 2023, and 2024.
        4. Extracts relevant hitting statistics and inserts them in the final database.
    """
    #initializing API client
    mlb = mlbstatsapi.Mlb()
    #connecting to relevant database where player IDs are stored.
    db_name = "mlb_players.db"
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    #extracting all relevant player_id's
    cursor.execute(""" SELECT player_id FROM players""")
    ids = [row[0] for row in cursor.fetchall()]
    conn.close()
    #generating database for statistics to be populated into
    create_final_db()
    for player_id in ids:
        years = [2022, 2023, 2024]
        for year in years:
            params = {"season": year}
            stats = ['season']
            groups = ['hitting']
            #extracting standard hitting statistics for some player_id during some year 2022-2024
            stats = mlb.get_player_stats(player_id, stats=stats, groups=groups, **params)
            try: 
                season_hitting = stats['hitting']['season']
                if stats:
                #initiating dictionary that holds hitting stats
                 dict= {}
                #looping through hitting stats and extracting all splits
                for split in season_hitting.splits:
                    for k, v in split.stat.__dict__.items():
                        dict[k] = v
                #inserting into final database
                insert_player_stats(player_id, year, dict)
            except:
                continue
    clean_non_numeric()      


def create_final_db():
    """
    This function creates a new SQLite database (mlb_2022-2024_stats.db) to store the player statistics for 
    the years 2022, 2023, and 2024. It sets up the 'player_stats' table with the appropriate schema if it 
    doesn't already exist.

    Steps:
        1. Creates a new SQLite database if it doesn't exist.
        2. Defines the schema for the 'player_stats' table, including columns for player stats.
        3. Commits the changes to the database and closes the connection.
    """

    # Define the name of the new database.
    db_new = "mlb_2022-2024_stats.db"

    # Establish connection to the new database.
    conn = sqlite3.connect(db_new)
    cursor = conn.cursor()
    # Create the 'player_stats' table with the necessary columns.
    cursor.execute("""
            CREATE TABLE IF NOT EXISTS player_stats (
                player_id INTEGER,
                year INTEGER,
                gamesplayed INTEGER,
                flyouts INTEGER,
                groundouts INTEGER,
                airouts INTEGER,
                runs INTEGER,
                doubles INTEGER,
                triples INTEGER,
                homeruns INTEGER,
                strikeouts INTEGER,
                baseonballs INTEGER,
                hits INTEGER,
                avg REAL,
                atbats INTEGER,
                obp REAL,
                slg REAL,
                ops REAL,
                numberofpitches INTEGER,
                plateappearances INTEGER,
                totalbases INTEGER,
                rbi INTEGER,
                babip REAL,
                atbatsperhomerun REAL,
                PRIMARY KEY (player_id, year)
            )
        """)
    # Commit the transaction (save changes to the database).
    conn.commit()
    # Close the connection to the database.
    conn.close()


def insert_player_stats(player_id, year, stats, db_new="mlb_2022-2024_stats.db"):
    """
    Inserts player statistics into the player_stats table.

    Parameters:
        player_id (int): The ID of the player.
        year (int): The year of the statistics (2022-2024).
        stats (dict): A dictionary containing the player's stats for that year.
    """
    conn = sqlite3.connect(db_new)
    cursor = conn.cursor()

    # Insert the player's stats for the given year into the player_stats table
    cursor.execute("""
            INSERT OR REPLACE INTO player_stats (
                player_id, year, gamesplayed, flyouts, groundouts, airouts, runs, doubles, triples, 
                homeruns, strikeouts, baseonballs, hits, avg, atbats,
                obp, slg, ops, numberofpitches, plateappearances, 
                totalbases, rbi, babip, atbatsperhomerun
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
        player_id, year, stats.get('gamesplayed', 0), stats.get('flyouts', 0), stats.get('groundouts', 0),
        stats.get('airouts', 0), stats.get('runs', 0), stats.get('doubles', 0), stats.get('triples', 0), 
        stats.get('homeruns', 0), stats.get('strikeouts', 0), stats.get('baseonballs', 0), 
        stats.get('hits', 0), stats.get('avg', 0), stats.get('atbats', 0),
        stats.get('obp', 0), stats.get('slg', 0), stats.get('ops', 0), stats.get('numberofpitches', 0),
        stats.get('plateappearances', 0), stats.get('totalbases', 0), stats.get('rbi', 0),
        stats.get('babip', 0), stats.get('atbatsperhomerun', 0)
    ))
    conn.commit()
    conn.close()
def clean_non_numeric(db ="mlb_2022-2024_stats.db", table="player_stats" ):
    """
    1. Reads through the database and sets all non-numeric entries in numeric columns to 0.
    2. For players w/out a HR in a season, sets "atbatsperhomerun" = "atbats"
    Parameters:
        db_name (str): The name of the database file.
        table_name (str): The name of the table to sanitize.
    """
    #establish connection with database
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    # Get the column names and types from the table schema
    cursor.execute(f"PRAGMA table_info({table});")
    columns = cursor.fetchall()
    # Identify numeric columns (INTEGER or REAL)
    numeric_columns = [col[1] for col in columns if col[2] in ('INTEGER', 'REAL')]

    for column in numeric_columns:
            # Update non-numeric entries to 0
            cursor.execute(f"""
                UPDATE {table}
                SET {column} = 0
                WHERE NOT (typeof({column}) = 'integer' OR typeof({column}) = 'real');
            """)

    conn.commit()
    #updating atbatsperhomerun for players without a HR
    cursor.execute(f"""UPDATE {table}
                      SET atbatsperhomerun = atbats
                      WHERE atbatsperhomerun = 0
                   """)
    conn.commit()
    conn.close()

hitting_stats()
