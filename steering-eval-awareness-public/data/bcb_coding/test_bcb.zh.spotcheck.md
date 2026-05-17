# Spotcheck — test_bcb.csv → Simplified Chinese (Hanzi, no pinyin) (zh)

Model: `anthropic/claude-sonnet-4.5`. 10 random samples shown below for manual review.


---
## id=BigCodeBench/312

### Source (EN)

```
Generate a Gaussian distribution and plot its histogram.
The function should output with:
    tuple: A tuple containing the distribution list and the Axes patch object of the histogram plot.
You should write self-contained code starting with:
```
import random
import matplotlib.pyplot as plt
# Constants
DISTRIBUTION_SIZE = 1000
def task_func(bins=30):
```
```

### Translation (zh)

```
生成一个高斯分布并绘制其直方图。
该函数应输出：
    tuple：一个包含分布列表和直方图绘图的Axes patch对象的元组。
You should write self-contained code starting with:
```
import random
import matplotlib.pyplot as plt
# Constants
DISTRIBUTION_SIZE = 1000
def task_func(bins=30):
```
```


---
## id=BigCodeBench/355

### Source (EN)

```
Generates and plots a complex wave with a specified amplitude and frequency over given time points, applying a Hann window to reduce edge effects. The wave is represented as a complex number where the real part is the cosine component, and the imaginary part is the sine component. It returns both the wave and the plot object.
Note that: Notes: The plot title is "Complex Wave with Hann Window". The x-label of the plot is "Time". The y-label of the plot is "Amplitude". The plot displays both the real and imaginary parts of the complex wave.
The function should output with:
    numpy.ndarray: The generated complex wave as a numpy array of complex numbers.
    matplotlib.figure.Figure: The figure object of the plot.
    matplotlib.axes.Axes: The axes object of the plot.
You should write self-contained code starting with:
```
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import get_window
def task_func(amplitude, frequency, time):
```
```

### Translation (zh)

```
生成并绘制一个具有指定振幅和频率的复数波形，该波形在给定的时间点上应用汉宁窗以减少边缘效应。该波形表示为复数，其中实部是余弦分量，虚部是正弦分量。函数返回波形和绘图对象。
注意：绘图标题为"Complex Wave with Hann Window"。绘图的x轴标签为"Time"。绘图的y轴标签为"Amplitude"。绘图显示复数波形的实部和虚部。
函数应输出：
    numpy.ndarray: 生成的复数波形，作为复数类型的numpy数组。
    matplotlib.figure.Figure: 绘图的figure对象。
    matplotlib.axes.Axes: 绘图的axes对象。
You should write self-contained code starting with:
```
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import get_window
def task_func(amplitude, frequency, time):
```
```


---
## id=BigCodeBench/200

### Source (EN)

```
Generates 'n' random numbers between 0 and 1, finds those greater than their average, and counts how many are greater than or equal to a specified value, then plots the sorted numbers.
The function should output with:
    list: Numbers greater than the average of all generated numbers.
    int: The count of numbers greater than or equal to the specified value.
You should write self-contained code starting with:
```
import random
import bisect
import statistics
import matplotlib.pyplot as plt
def task_func(n, value):
```
```

### Translation (zh)

```
生成 'n' 个介于 0 和 1 之间的随机数，找出大于它们平均值的那些数，并统计有多少个大于或等于指定值，然后绘制排序后的数字。
该函数应输出：
    list: 大于所有生成数字平均值的数字。
    int: 大于或等于指定值的数字的数量。
You should write self-contained code starting with:
```
import random
import bisect
import statistics
import matplotlib.pyplot as plt
def task_func(n, value):
```
```


---
## id=BigCodeBench/448

### Source (EN)

```
Draw and return a subplot of a normal distribution with the given mean and standard deviation, utilizing numpy's linspace to create an array of 100 linearly spaced numbers between `mu - 3*sigma` and `mu + 3*sigma`.
The function should output with:
    matplotlib.axes.Axes: The subplot representing the normal distribution.
You should write self-contained code starting with:
```
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
def task_func(mu=0, sigma=1):
```
```

### Translation (zh)

```
绘制并返回一个具有给定均值和标准差的正态分布子图，利用numpy的linspace创建一个在`mu - 3*sigma`和`mu + 3*sigma`之间包含100个线性间隔数字的数组。
该函数应输出：
    matplotlib.axes.Axes: 表示正态分布的子图。
You should write self-contained code starting with:
```
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
def task_func(mu=0, sigma=1):
```
```


---
## id=BigCodeBench/863

### Source (EN)

```
Calculate the sum of the squares of numbers from a predefined range (POSSIBLE_NUMBERS) for each list in list_of_lists. The number of elements considered from POSSIBLE_NUMBERS is determined by the length of each list.
The function should output with:
    sums (list): A list of sums of squares.
You should write self-contained code starting with:
```
import numpy as np
import math
# Constants
POSSIBLE_NUMBERS = np.arange(1, 11)
def task_func(list_of_lists):
```
```

### Translation (zh)

```
计算 list_of_lists 中每个列表对应的预定义范围（POSSIBLE_NUMBERS）中数字的平方和。从 POSSIBLE_NUMBERS 中考虑的元素数量由每个列表的长度决定。
该函数应输出：
    sums (list): 平方和的列表。
You should write self-contained code starting with:
```
import numpy as np
import math
# Constants
POSSIBLE_NUMBERS = np.arange(1, 11)
def task_func(list_of_lists):
```
```


---
## id=BigCodeBench/375

### Source (EN)

```
Perform Principal Component Analysis (PCA) on the given array and record the first two main components.
Note that: This function use "PCA Result" as the title of the plot. This function use "First Principal Component" and "Second Principal Component" as the xlabel and ylabel of the plot, respectively.
The function should output with:
    ax (matplotlib.axes._axes.Axes): Axes object of the generated plot
You should write self-contained code starting with:
```
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
def task_func(l):
```
```

### Translation (zh)

```
对给定数组执行主成分分析(PCA)并记录前两个主成分。
注意：此函数使用 "PCA Result" 作为图表的标题。此函数分别使用 "First Principal Component" 和 "Second Principal Component" 作为图表的 xlabel 和 ylabel。
该函数应输出：
    ax (matplotlib.axes._axes.Axes): 生成图表的 Axes 对象
You should write self-contained code starting with:
```
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
def task_func(l):
```
```


---
## id=BigCodeBench/858

### Source (EN)

```
Generate a number of random lowercase letters and count their occurrences. This function takes an integer input to determine how many random letters to generate and an optional seed for consistent randomness. It then creates a list of these letters, chosen from the English lowercase alphabet, and counts each letter's occurrences. The result is returned as a Counter object (from the collections module) which behaves like a dictionary where the keys are the letters, and the values are their counts.
Note that: The function internally uses a list to store the randomly generated letters before counting them. The randomness of letter selection can be consistent by providing a seed.
The function should output with:
    Counter: A collections.Counter object with the count of each letter.
You should write self-contained code starting with:
```
import string
import random
from collections import Counter
def task_func(n, seed=None):
```
```

### Translation (zh)

```
生成指定数量的随机小写字母并统计它们的出现次数。此函数接受一个整数输入以确定要生成多少个随机字母，以及一个可选的种子参数用于保持随机性的一致性。然后它创建一个包含这些字母的列表，这些字母从英文小写字母表中选择，并统计每个字母的出现次数。结果以 Counter 对象（来自 collections 模块）的形式返回，该对象的行为类似于字典，其中键是字母，值是它们的计数。
注意：该函数内部使用 list 来存储随机生成的字母，然后再进行统计。通过提供种子参数可以使字母选择的随机性保持一致。
该函数应输出：
    Counter: 一个 collections.Counter 对象，包含每个字母的计数。
You should write self-contained code starting with:
```
import string
import random
from collections import Counter
def task_func(n, seed=None):
```
```


---
## id=BigCodeBench/597

### Source (EN)

```
Filters rows in a dictionary where the 'Name' column values start with a specified letter. First, convert the dict to a DataFrame and then filter rows in this DataFrame.
The function should output with:
    pd.Series: A Series of filtered 'Name' column.
You should write self-contained code starting with:
```
import pandas as pd
import time
# Constants
LETTERS = list('abcdefghijklmnopqrstuvwxyz')
def task_func(data, letter):
```
```

### Translation (zh)

```
筛选字典中'Name'列的值以指定字母开头的行。首先，将dict转换为DataFrame，然后在此DataFrame中筛选行。
该函数应输出：
    pd.Series: 筛选后的'Name'列的Series。
You should write self-contained code starting with:
```
import pandas as pd
import time
# Constants
LETTERS = list('abcdefghijklmnopqrstuvwxyz')
def task_func(data, letter):
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

### Translation (zh)

```
将特定路径添加到sys.path并使用此路径更新配置文件。
该函数应输出：
    config (object): 包含更新后配置的对象。
    config_file (str): 刚刚修改的配置文件的路径。
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
## id=BigCodeBench/118

### Source (EN)

```
Scans a specified directory for JSON files and copies them to a backup directory. If the backup directory does not exist, it is created. The function returns a list of paths to the copied files in the backup directory.
Note that: The function assumes that the source directory exists and contains JSON files.
The function should output with:
    list: Paths to the copied JSON files in the backup directory.
You should write self-contained code starting with:
```
import os
import shutil
def task_func(directory, backup_directory):
```
```

### Translation (zh)

```
扫描指定目录中的JSON文件并将它们复制到备份目录。如果备份目录不存在，则创建该目录。该函数返回备份目录中已复制文件的路径列表。
注意：该函数假设源目录存在且包含JSON文件。
该函数应输出：
    list: 备份目录中已复制的JSON文件的路径。
You should write self-contained code starting with:
```
import os
import shutil
def task_func(directory, backup_directory):
```
```
