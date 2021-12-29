import pandas as pd
from os import listdir
from os.path import isfile, join

path = "/edward-slow-vol/cartoonset40k/"

files = [join(path,f) for f in listdir(path) if isfile(join(path,f))]
df = pd.DataFrame(files)
df.to_csv('list_cartoon.csv')
