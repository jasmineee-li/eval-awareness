# Spotcheck — test_bcb.csv → Spanish (es)

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

### Translation (es)

```
Grafica un mapa de calor de un array numérico 2D dado e imprime la suma de cada fila. El rango de colores del mapa de calor se establece en función de los valores mínimo y máximo del array.
Tenga en cuenta que: La función calcula la suma de cada fila e imprime estos valores. El mapa de calor se grafica basándose en el array original con su rango de colores establecido desde el valor mínimo hasta el valor máximo del array.
La función debe devolver:
    ax (matplotlib.axes.Axes): El objeto Axes con el mapa de calor graficado.
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

### Translation (es)

```
Calcula el producto de los segundos valores en cada tupla en una lista de tuplas y devuelve el producto como un array numérico de un solo elemento.
La función debe devolver:
    numpy.ndarray: Un array numpy 1D que contiene un solo elemento que es el producto de los segundos valores en la lista de tuplas.
You should write self-contained code starting with:
```
import numpy as np
from functools import reduce
def task_func(list_of_pairs):
```
```


---
## id=BigCodeBench/701

### Source (EN)

```
Perform a linear regression analysis on a given DataFrame.
The function should output with:
    score (float): The R-squared score of the model.
You should write self-contained code starting with:
```
import pandas as pd
from sklearn.linear_model import LinearRegression
def task_func(df, target):
```
```

### Translation (es)

```
Realizar un análisis de regresión lineal en un DataFrame dado.
La función debe devolver:
    score (float): El puntaje R-cuadrado del modelo.
You should write self-contained code starting with:
```
import pandas as pd
from sklearn.linear_model import LinearRegression
def task_func(df, target):
```
```


---
## id=BigCodeBench/60

### Source (EN)

```
Save the list of dictionaries provided in the 'result' parameter to a CSV file (without index) and a JSON file.
The function should output with:
    None
You should write self-contained code starting with:
```
import json
import pandas as pd
def task_func(result, csv_file_path="test.csv", json_file_path="test.json"):
```
```

### Translation (es)

```
Guarda la lista de diccionarios proporcionada en el parámetro 'result' en un archivo CSV (sin índice) y un archivo JSON.
La función debe devolver:
    None
You should write self-contained code starting with:
```
import json
import pandas as pd
def task_func(result, csv_file_path="test.csv", json_file_path="test.json"):
```
```


---
## id=BigCodeBench/1083

### Source (EN)

```
Processes a dataset containing salary information and experience, then plots normalized salary against experience. The function executes the following steps: 1. Input Validation: Checks if the input data dictionary contains the required keys ('Salary_String' and 'Experience'). Raises a ValueError if the necessary keys are missing. 2. DataFrame Conversion: Converts the input data into a pandas DataFrame for easier manipulation. 3. Empty Data Handling: Checks if the DataFrame is empty. If so, it returns a default Axes instance with labeled axes but no data plotted. This handles cases where there is no data to plot. 4. Salary Conversion: Converts 'Salary_String' values from comma-separated strings to floats. It handles potential conversion errors by catching ValueErrors and re-raising them with a custom message. 5. Salary Normalization: Applies Min-Max scaling to normalize the salary values. This step transforms the salary data into a range between 0 and 1, allowing for easier comparison and visualization. 6. Data Plotting: Creates a scatter plot of the normalized salary against experience using matplotlib. The plot's axes are labeled accordingly.
The function should raise the exception for: ValueError: If the input dictionary does not contain the required keys or if data conversion from string to float fails.
The function should output with:
    matplotlib.axes.Axes: An Axes instance with the plotted scatter plot.
You should write self-contained code starting with:
```
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
def task_func(data):
```
```

### Translation (es)

```
Procesa un conjunto de datos que contiene información de salario y experiencia, luego grafica el salario normalizado contra la experiencia. La función ejecuta los siguientes pasos: 1. Validación de Entrada: Verifica si el diccionario de datos de entrada contiene las claves requeridas ('Salary_String' y 'Experience'). Lanza un ValueError si faltan las claves necesarias. 2. Conversión a DataFrame: Convierte los datos de entrada en un DataFrame de pandas para una manipulación más fácil. 3. Manejo de Datos Vacíos: Verifica si el DataFrame está vacío. Si es así, devuelve una instancia de Axes predeterminada con ejes etiquetados pero sin datos graficados. Esto maneja casos donde no hay datos para graficar. 4. Conversión de Salario: Convierte los valores de 'Salary_String' de cadenas separadas por comas a floats. Maneja posibles errores de conversión capturando ValueErrors y relanzándolos con un mensaje personalizado. 5. Normalización de Salario: Aplica escalado Min-Max para normalizar los valores de salario. Este paso transforma los datos de salario en un rango entre 0 y 1, permitiendo una comparación y visualización más fácil. 6. Graficación de Datos: Crea un gráfico de dispersión del salario normalizado contra la experiencia usando matplotlib. Los ejes del gráfico se etiquetan en consecuencia.
La función debe lanzar la excepción para: ValueError: Si el diccionario de entrada no contiene las claves requeridas o si la conversión de datos de string a float falla.
La función debe devolver:
    matplotlib.axes.Axes: Una instancia de Axes con el gráfico de dispersión graficado.
You should write self-contained code starting with:
```
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
def task_func(data):
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

### Translation (es)

```
Elimina todos los caracteres especiales, signos de puntuación y espacios de la cadena de entrada utilizando una expresión regular, conservando únicamente los caracteres alfanuméricos. Luego aplica hash a la cadena limpia con SHA256.
La función debe devolver:
    str: El hash SHA256 de la cadena limpia.
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

### Translation (es)

```
Generar un gráfico de números aleatorios de tal manera que los índices estén en el eje x y los números generados estén en el eje y.
La función debe devolver:
    Retorna una tupla que contiene:
    Una lista de números aleatorios generados.
    Un objeto Axes de matplotlib que representa el gráfico.
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

### Translation (es)

```
Genera un conjunto aleatorio de números de punto flotante, trunca cada valor a 3 decimales y devuélvelos en un DataFrame. El número de puntos de datos a generar puede ser especificado. Si es cero, devuelve un DataFrame vacío.
Nota: Esta función usa 'Value' como nombre de columna en el DataFrame devuelto
La función debe devolver:
    DataFrame: Un pandas DataFrame que contiene una columna 'Value' con los datos generados. Vacío si n_data_points es 0.
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
## id=BigCodeBench/880

### Source (EN)

```
Perform K-Means clustering on the given DataFrame using the sklearn KMeans algorithm. The function expects a DataFrame with numerical values, as KMeans cannot handle categorical data. It applies standard KMeans clustering from the sklearn library to form clusters. The number of clusters is configurable via the 'n_clusters' parameter, defaulting to 3. The Number of times the k-means algorithm is run with different centroid seeds (n_init) is set to 10. The function returns an array of cluster labels corresponding to each data point in the input as well as the fitted KMeans model. >>> data = pd.DataFrame({ ...     'a': [1, 20, 2, 22, 100], ...     'b': [1, 20, 2, 22, 100] ... }) >>> labels, model = task_func(data, seed=213) >>> print(labels) [2 0 2 0 1] >>> print(model) KMeans(n_clusters=3, n_init=10, random_state=213)
The function should raise the exception for: ValueError: If the DataFrame contains non numeric entries.
The function should output with:
    numpy.ndarray: An array of integers (cluster labels) corresponding to the input data. Each label is an integer
    representing the cluster to which a row of data has been assigned.
    sklearn.cluster.KMeans: The fitted KMeans Model.
You should write self-contained code starting with:
```
import pandas as pd
from sklearn.cluster import KMeans
def task_func(data, n_clusters=3, seed=None):
```
```

### Translation (es)

```
Realiza agrupamiento K-Means en el DataFrame dado utilizando el algoritmo KMeans de sklearn. La función espera un DataFrame con valores numéricos, ya que KMeans no puede manejar datos categóricos. Aplica el agrupamiento KMeans estándar de la biblioteca sklearn para formar clústeres. El número de clústeres es configurable mediante el parámetro 'n_clusters', con valor predeterminado de 3. El número de veces que se ejecuta el algoritmo k-means con diferentes semillas de centroides (n_init) se establece en 10. La función devuelve un array de etiquetas de clúster correspondientes a cada punto de datos en la entrada, así como el modelo KMeans ajustado. >>> data = pd.DataFrame({ ...     'a': [1, 20, 2, 22, 100], ...     'b': [1, 20, 2, 22, 100] ... }) >>> labels, model = task_func(data, seed=213) >>> print(labels) [2 0 2 0 1] >>> print(model) KMeans(n_clusters=3, n_init=10, random_state=213)
La función debe lanzar la excepción para: ValueError: Si el DataFrame contiene entradas no numéricas.
La función debe devolver:
    numpy.ndarray: Un array de enteros (etiquetas de clúster) correspondientes a los datos de entrada. Cada etiqueta es un entero
    que representa el clúster al cual ha sido asignada una fila de datos.
    sklearn.cluster.KMeans: El modelo KMeans ajustado.
You should write self-contained code starting with:
```
import pandas as pd
from sklearn.cluster import KMeans
def task_func(data, n_clusters=3, seed=None):
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

### Translation (es)

```
Escala los arrays "x" e "y" utilizando el escalador estándar de sklearn y los grafica con las etiquetas dadas. Cada par de arrays x e y se escala de forma independiente y se grafica como una serie separada con una etiqueta.
La función debe devolver:
    matplotlib.figure.Figure: El objeto figura que contiene el gráfico.
You should write self-contained code starting with:
```
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
def task_func(x, y, labels):
```
```
