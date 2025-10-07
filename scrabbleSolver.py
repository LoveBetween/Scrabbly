standard_multi_grid = [
        [4,0,0,1,0,0,0,4,0,0,0,1,0,0,4],
        [0,3,0,0,0,2,0,0,0,2,0,0,0,3,0],
        [0,0,3,0,0,0,1,0,1,0,0,0,3,0,0],
        [1,0,0,3,0,0,0,1,0,0,0,3,0,0,1],
        [0,0,0,0,3,0,0,0,0,0,3,0,0,0,0],
        [0,2,0,0,0,2,0,0,0,2,0,0,0,2,0],
        [0,0,1,0,0,0,1,0,1,0,0,0,1,0,0],
        [4,0,0,1,0,0,0,5,0,0,0,1,0,0,4],
        [0,0,1,0,0,0,1,0,1,0,0,0,1,0,0],
        [0,2,0,0,0,2,0,0,0,2,0,0,0,2,0],
        [0,0,0,0,3,0,0,0,0,0,3,0,0,0,0],
        [1,0,0,3,0,0,0,1,0,0,0,3,0,0,1],
        [0,0,3,0,0,0,1,0,1,0,0,0,3,0,0],
        [0,3,0,0,0,2,0,0,0,2,0,0,0,3,0],
        [4,0,0,1,0,0,0,4,0,0,0,1,0,0,4],
    ]

def find_placement_vertical_perpendicular(grid, x, y):
    placement = "_"
    before = 1
    while x - before >= 0 and grid[x - before][y] != "_":
        placement = grid[x - before][y] + placement
        before += 1
    after = 1
    while x + after < 15 and grid[x + after][y] != "_":
        placement += grid[x + after][y]
        after += 1
    return placement

def find_placement_vertical(grid, x, y, dir):
    if y > 0 and grid[x][y - 1] != "_":
        return None
    correct_placement = False
    letter_nb = 0
    space_nb = 0
    placement = {}
    placement['x'] = x
    placement['y'] = y
    placement['direction'] = dir
    main_word = ""
    perpendicular_words = []

    while space_nb < 7 and y + space_nb + letter_nb < 15:
        if grid[x][y + space_nb + letter_nb] == "_":
            main_word += "_"
            perpendicular_placement = find_placement_vertical_perpendicular(grid, x, y + space_nb + letter_nb)
            perpendicular_words.append(perpendicular_placement)
            if len(perpendicular_placement) > 1:
                correct_placement = True
            space_nb += 1
        else:
            main_word += grid[x][y + space_nb + letter_nb]
            perpendicular_words.append("")
            letter_nb += 1
            correct_placement = True
    if not correct_placement:
        return None
    placement['next_letter'] = "_"
    if y + space_nb + letter_nb < 15:
        placement['next_letter'] = grid[x][y + space_nb + letter_nb]
    placement['word'] = main_word
    placement['perpendicular'] = perpendicular_words
    return placement

def find_all_placements(grid):
    transposed_grid = [list(col) for col in zip(*grid)]
    all_placements = []
    for x in range(15):
        for y in range(15):
            placement = find_placement_vertical(grid, x, y, "vertical")
            placement2 = find_placement_vertical(transposed_grid, y, x, "horizontal")
            if placement is not None:
                all_placements.append(placement)
            if placement2 is not None:
                placement2['x'] = x
                placement2['y'] = y
                all_placements.append(placement2)
    return all_placements

def is_placed(cell):
    return isinstance(cell, str) and len(cell) == 2 and cell[1] == "°"

def find_play(grid, width, height):
    placed_l = []
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if is_placed(cell):
                placed_l.append([r, c])
    placed_nb = 1
    dir = 0
    x = placed_l[0][0]
    y = placed_l[0][1]
    if len(placed_l) == 1:
        if (x > 0 and grid[x - 1][y] != "_") or x < width and grid[x + 1][y] != "_":
            dir = 1
    else:
        x_end = placed_l[-1][0]
        y_end = placed_l[-1][1]
        dir = 1 if (x_end - x) >= (y_end - y) else 0
        while x < width and y < height and placed_nb < len(placed_l) and ((dir > 0 and x < x_end) or (dir < 1 and y < y_end)):
            x += dir
            y += 1 - dir
            if grid[x][y] == "_":
                return None
            elif is_placed(grid[x][y]):
                placed_nb += 1
    if placed_nb == len(placed_l):
        x = placed_l[0][0]
        y = placed_l[0][1]
        while x > 0 and y > 0 and grid[x][y] != "_":
            x -= dir
            y -= 1 - dir
        if grid[x][y] == "_":
            x += dir
            y += 1 - dir
        cleaned_grid = [[value if len(value) > 1 else "_" for value in row] for row in grid]
        transposed_grid = [list(col) for col in zip(*cleaned_grid)]
        if dir < 1:
            return find_placement_vertical(cleaned_grid, x, y, "vertical")
        else:
            placement = find_placement_vertical(transposed_grid, y, x, "horizontal")
            if placement is not None:
                placement['x'] = x
                placement['y'] = y
            return placement
    return None

alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
nb_letters = [9, 2, 2, 3, 15, 2, 2, 2, 8, 1, 1, 5, 3, 6, 6, 2, 1, 6, 6, 6, 6, 2, 1, 1, 1, 1]
valeur_des_lettres = [1, 3, 3, 2, 1, 4, 2, 4, 1, 8, 10, 1, 2, 1, 1, 3, 8, 1, 1, 1, 1, 4, 10, 10, 10, 10]

nb_obj = {letter: nb_letters[i] for i, letter in enumerate(alphabet)}
valeur_obj = {letter: valeur_des_lettres[i] for i, letter in enumerate(alphabet)}

sac_de_lettres = "".join([letter * nb_letters[i] for i, letter in enumerate(alphabet)])

def pick_letters(bag, nb):
    letters = ""
    for _ in range(nb):
        if not bag:
            break
        rnd = random.randint(0, len(bag) - 1)
        letters += bag[rnd]
        bag = bag[:rnd] + bag[rnd + 1:]
    return bag, letters

def remove_letters(letters, word):
    for letter in word:
        letters = letters.replace(letter, "")
    return letters

def place_word(grid, placement, word):
    placed = False
    if placement['direction'] == "horizontal":
        if placement['x'] + len(word) < 16:
            for i in range(len(word)):
                grid[placement['x'] + i][placement['y']] = word[i]
            placed = True
    elif placement['direction'] == "vertical":
        if placement['y'] + len(word) < 16:
            for i in range(len(word)):
                grid[placement['x']][placement['y'] + i] = word[i]
            placed = True
    return placed

def calculate_placement_score(placement, word, letter_points, multi_grid):
    x, y = placement['x'], placement['y']
    multiplier = 1
    total_score = 0
    word_score = 0
    cell_type = 0
    letters_used = 0
    for i in range(len(word)):
        current_multi = 1
        current_letter_points = 0
        if placement['direction'] == "horizontal":
            cell_type = multi_grid[x + i][y]
        else:
            cell_type = multi_grid[x][y + i]
        if cell_type < 3:
            if placement['word'][i] == "_":
                letters_used += 1
                current_letter_points = letter_points[word[i]] * (cell_type + 1)
            else:
                current_letter_points = letter_points[word[i]]
        elif cell_type < 5:
            current_letter_points = letter_points[word[i]]
            if placement['word'][i] == "_":
                current_multi = cell_type - 1
                multiplier = max(multiplier, cell_type - 1)
        word_score += current_letter_points
        pp = placement['perpendicular'][i]
        pp_score = 0
        if len(pp) > 1:
            hole = pp.index("_")
            for j in range(len(pp)):
                if j == hole:
                    pp_score += current_letter_points
                else:
                    pp_score += letter_points[pp[j]]
            total_score += pp_score * current_multi
    total_score += word_score * multiplier
    total_score = letters_used >= 7 and total_score + 50 or total_score
    return total_score

def get_all_correct_placements(placements, letters, dawg):
    return [(placement, list(dawg.check_placement(placement, letters))) for placement in placements if list(dawg.check_placement(placement, letters))]

def connections_nb(w):
    return sum(1 for p_word in w['placement']['perpendicular'] if p_word.length > 1)

from enum import Enum
from functools import reduce

# Assuming calculate_placement_score and connections_nb are defined elsewhere
# Also assuming valeur_obj and standard_multi_grid are in scope

class Filters(Enum):
    LONGEST = staticmethod(lambda a, b: a if len(a['word']) > len(b['word']) else b)
    SHORTEST = staticmethod(lambda a, b: a if len(a['word']) < len(b['word']) else b)
    MOST_POINTS = staticmethod(lambda a, b: (
        a if calculate_placement_score(a['placement'], a['word'], valeur_obj, standard_multi_grid) >
             calculate_placement_score(b['placement'], b['word'], valeur_obj, standard_multi_grid)
        else b
    ))
    MOST_CONNECTIONS = staticmethod(lambda a, b: (
        a if connections_nb(a) > connections_nb(b) else b
    ))


def filter_words(placements, filter_enum):
    flattened = [dict(placement=placement, word=word)
                 for placement, words in placements
                 for word in words]

    if not flattened:
        return None

    return reduce(filter_enum, flattened)

def find_best_word(dawg, grid, letters, filter):
    all_placements = find_all_placements(grid)
    all_correct_placements = get_all_correct_placements(all_placements, letters, dawg)
    best_word = filter_words(all_correct_placements, Filters.MOST_POINTS)
    best_word['score'] = calculate_placement_score(best_word['placement'], best_word['word'], valeur_obj, standard_multi_grid)
    return best_word