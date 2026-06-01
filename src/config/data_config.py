YEARS = list(range(2014, 2025))
CLEAN_THRESHOLD = 0.2
SURFACES = ['Hard', 'Clay', 'Grass']
IMPORTANCE_COLUMNS = [
    'target', 
    'elo_diff', 
    'elo_hard_diff', 
    'rank_diff', 
    'rank_points_diff', 
    'elo_clay_diff', 
    'elo_grass_diff', 
    'age_diff', 
    'height_diff', 
    # New encoded columns
    'surface_Hard',
    'surface_Clay',
    'surface_Grass',
    'surface_Carpet',
    'tourney_level_A',  # ATP level
    'tourney_level_M',  # Masters level
    'tourney_level_G',  # Grand Slam level
    'tourney_level_D',  # Davis Cup
    'tourney_level_F'   # Tour Finals
]