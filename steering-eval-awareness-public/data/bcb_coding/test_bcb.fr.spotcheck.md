# Spotcheck — test_bcb.csv → French (fr)

Model: `anthropic/claude-sonnet-4.5`. 10 random samples shown below for manual review.


---
## id=BigCodeBench/1064

### Source (EN)

```
Plots a heatmap of a given 2D numerical array and prints the sum of each row. The heatmap's color range is set based on the minimum and maximum values in the array.
Note that: The function calculates the sum of each row and prints these values. The heatmap is plotted based on the original array with its color range set from the minimum to the maximum value in the array.
The function should output with:
    ax (matplotlib.axes.Axes): The Axes object with the plotted heatmap.
You should write self-contained code starting with:
```
import numpy as np
import seaborn as sns
def task_func(arr):
```
```

### Translation (fr)

```
Trace une carte de chaleur d'un tableau numérique 2D donné et affiche la somme de chaque ligne. La plage de couleurs de la carte de chaleur est définie en fonction des valeurs minimale et maximale du tableau.
Notez que : La fonction calcule la somme de chaque ligne et affiche ces valeurs. La carte de chaleur est tracée en fonction du tableau original avec sa plage de couleurs définie de la valeur minimale à la valeur maximale du tableau.
La fonction doit retourner :
    ax (matplotlib.axes.Axes) : L'objet Axes avec la carte de chaleur tracée.
You should write self-contained code starting with:
```
import numpy as np
import seaborn as sns
def task_func(arr):
```
```


---
## id=BigCodeBench/33

### Source (EN)

```
Calculate the product of the second values in each tuple in a list of tuples and return the product as a single-element numeric array.
The function should output with:
    numpy.ndarray: A 1D numpy array containing a single element that is the product of the second values in the list of tuples.
You should write self-contained code starting with:
```
import numpy as np
from functools import reduce
def task_func(list_of_pairs):
```
```

### Translation (fr)

```
Calculer le produit des deuxièmes valeurs de chaque tuple dans une liste de tuples et retourner le produit sous forme de tableau numérique à un seul élément.
La fonction doit retourner :
    numpy.ndarray: Un tableau numpy 1D contenant un seul élément qui est le produit des deuxièmes valeurs de la liste de tuples.
You should write self-contained code starting with:
```
import numpy as np
from functools import reduce
def task_func(list_of_pairs):
```
```


---
## id=BigCodeBench/3

### Source (EN)

```
Create a dictionary where keys are specified letters and values are lists of random integers. Then calculate the mean of these integers for each key and return a dictionary of these means.
The function should output with:
    dict: A dictionary where each key is a letter from the input list and the value is the mean of
    a randomly generated list of integers (with each list having 1 to 10 integers ranging from 0 to 100).
You should write self-contained code starting with:
```
import random
import numpy as np
def task_func(LETTERS):
```
```

### Translation (fr)

```
Créer un dictionnaire où les clés sont des lettres spécifiées et les valeurs sont des listes d'entiers aléatoires. Ensuite, calculer la moyenne de ces entiers pour chaque clé et retourner un dictionnaire de ces moyennes.
La fonction doit retourner :
    dict: Un dictionnaire où chaque clé est une lettre de la liste d'entrée et la valeur est la moyenne d'une liste générée aléatoirement d'entiers (chaque liste ayant de 1 à 10 entiers allant de 0 à 100).
You should write self-contained code starting with:
```
import random
import numpy as np
def task_func(LETTERS):
```
```


---
## id=BigCodeBench/254

### Source (EN)

```
Calculate the square root of the given decimal value to a certain precision and then encode the result as a JSON string.
The function should output with:
    str: The square root of the decimal value encoded as a JSON string.
You should write self-contained code starting with:
```
import json
import math
def task_func(decimal_value, precision=2):
```
```

### Translation (fr)

```
Calculer la racine carrée de la valeur décimale donnée avec une certaine précision, puis encoder le résultat sous forme de chaîne JSON.
La fonction doit retourner :
    str : La racine carrée de la valeur décimale encodée sous forme de chaîne JSON.
You should write self-contained code starting with:
```
import json
import math
def task_func(decimal_value, precision=2):
```
```


---
## id=BigCodeBench/412

### Source (EN)

```
This function reads a JSON file where each key is a unique identifier, and the corresponding value is a base64 encoded string. After decoding, it applies Unicode normalization form C (NFC) to each decoded string to ensure the canonical composition of characters. The function returns a dictionary where the keys are preserved, and the values are the normalized, decoded strings. Decoding is performed using the UTF-8 encoding scheme. >>> task_func('example.json') {'key1': 'Hello World', 'key2': 'Python Code Refiner'} Given a file 'empty.json' with the content: {} >>> task_func('empty.json') {}
The function should output with:
    dict: A dictionary where each key is mapped to a normalized, decoded string from the base64 encoded value in the input file.
You should write self-contained code starting with:
```
import json
import base64
import unicodedata
def task_func(json_file: str) -> dict:
```
```

### Translation (fr)

```
Cette fonction lit un fichier JSON où chaque clé est un identifiant unique, et la valeur correspondante est une chaîne encodée en base64. Après décodage, elle applique la forme de normalisation Unicode C (NFC) à chaque chaîne décodée pour garantir la composition canonique des caractères. La fonction retourne un dictionnaire où les clés sont préservées, et les valeurs sont les chaînes décodées et normalisées. Le décodage est effectué en utilisant le schéma d'encodage UTF-8. >>> task_func('example.json') {'key1': 'Hello World', 'key2': 'Python Code Refiner'} Étant donné un fichier 'empty.json' avec le contenu : {} >>> task_func('empty.json') {}
La fonction doit retourner :
    dict: Un dictionnaire où chaque clé est associée à une chaîne normalisée et décodée à partir de la valeur encodée en base64 dans le fichier d'entrée.
You should write self-contained code starting with:
```
import json
import base64
import unicodedata
def task_func(json_file: str) -> dict:
```
```


---
## id=BigCodeBench/1126

### Source (EN)

```
Removes all special characters, punctuation marks, and spaces from the input string using a regular expression, retaining only alphanumeric characters. Then hashes the cleaned string with SHA256.
The function should output with:
    str: The SHA256 hash of the cleaned string.
You should write self-contained code starting with:
```
import re
import hashlib
def task_func(input_str):
```
```

### Translation (fr)

```
Supprime tous les caractères spéciaux, les signes de ponctuation et les espaces de la chaîne d'entrée en utilisant une expression régulière, ne conservant que les caractères alphanumériques. Puis hache la chaîne nettoyée avec SHA256.
La fonction doit retourner :
    str: Le hachage SHA256 de la chaîne nettoyée.
You should write self-contained code starting with:
```
import re
import hashlib
def task_func(input_str):
```
```


---
## id=BigCodeBench/405

### Source (EN)

```
Generate a plot of random numbers such that indices are on the x-axis and generated numbers are on the y-axis.
The function should output with:
    Returns a tuple containing:
    A list of generated random numbers.
    A matplotlib Axes object representing the plot.
You should write self-contained code starting with:
```
import random
import matplotlib.pyplot as plt
def task_func(points: int):
```
```

### Translation (fr)

```
Générer un graphique de nombres aléatoires tel que les indices sont sur l'axe des x et les nombres générés sont sur l'axe des y.
La fonction doit retourner :
    Retourne un tuple contenant :
    Une liste de nombres aléatoires générés.
    Un objet matplotlib Axes représentant le graphique.
You should write self-contained code starting with:
```
import random
import matplotlib.pyplot as plt
def task_func(points: int):
```
```


---
## id=BigCodeBench/243

### Source (EN)

```
Generate a random set of floating-point numbers, truncate each value to 3 decimal places, and return them in a DataFrame. The number of data points to generate can be specified. If zero, returns an empty DataFrame.
Note that: This function use 'Value' for the column name in returned DataFrame
The function should output with:
    DataFrame: A pandas DataFrame containing one column 'Value' with the generated data. Empty if n_data_points is 0.
You should write self-contained code starting with:
```
import pandas as pd
import random
# Constants
N_DATA_POINTS = 10000
MIN_VALUE = 0.0
MAX_VALUE = 10.0
def task_func(n_data_points=N_DATA_POINTS):
```
```

### Translation (fr)

```
Générer un ensemble aléatoire de nombres à virgule flottante, tronquer chaque valeur à 3 décimales, et les retourner dans un DataFrame. Le nombre de points de données à générer peut être spécifié. Si zéro, retourne un DataFrame vide.
Notez que : Cette fonction utilise 'Value' comme nom de colonne dans le DataFrame retourné
La fonction doit retourner :
    DataFrame: Un pandas DataFrame contenant une colonne 'Value' avec les données générées. Vide si n_data_points vaut 0.
You should write self-contained code starting with:
```
import pandas as pd
import random
# Constants
N_DATA_POINTS = 10000
MIN_VALUE = 0.0
MAX_VALUE = 10.0
def task_func(n_data_points=N_DATA_POINTS):
```
```


---
## id=BigCodeBench/717

### Source (EN)

```
Add a specific path to sys.path and update a configuration file with this path.
The function should output with:
    config (object): The object contains the updated configuration.
    config_file (str): The path to the configuration file that was just modified.
You should write self-contained code starting with:
```
import sys
from configparser import ConfigParser
# Constants
PATH_TO_APPEND = '/path/to/whatever'
CONFIG_FILE = '/path/to/config.ini'
def task_func(path_to_append=PATH_TO_APPEND, config_file=CONFIG_FILE):
```
```

### Translation (fr)

```
Ajouter un chemin spécifique à sys.path et mettre à jour un fichier de configuration avec ce chemin.
La fonction doit retourner :
    config (object) : L'objet contient la configuration mise à jour.
    config_file (str) : Le chemin vers le fichier de configuration qui vient d'être modifié.
You should write self-contained code starting with:
```
import sys
from configparser import ConfigParser
# Constants
PATH_TO_APPEND = '/path/to/whatever'
CONFIG_FILE = '/path/to/config.ini'
def task_func(path_to_append=PATH_TO_APPEND, config_file=CONFIG_FILE):
```
```


---
## id=BigCodeBench/660

### Source (EN)

```
Scale the "x" and "y" arrays using the standard scaler of sklearn and plot them with given labels. Each pair of x and y arrays are scaled independently and plotted as a separate series with a label.
The function should output with:
    matplotlib.figure.Figure: The figure object containing the plot.
You should write self-contained code starting with:
```
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
def task_func(x, y, labels):
```
```

### Translation (fr)

```
Mettre à l'échelle les tableaux "x" et "y" en utilisant le standard scaler de sklearn et les tracer avec les étiquettes données. Chaque paire de tableaux x et y est mise à l'échelle indépendamment et tracée comme une série distincte avec une étiquette.
La fonction doit retourner :
    matplotlib.figure.Figure: L'objet figure contenant le graphique.
You should write self-contained code starting with:
```
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
def task_func(x, y, labels):
```
```
