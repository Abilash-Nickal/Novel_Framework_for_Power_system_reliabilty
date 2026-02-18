import random
import pandas as pd
import numpy as np
import os
gen_data = pd.read_csv("../data/CEB_GEN_FOR_for_each_unit.csv")
gen_data.columns = gen_data.columns.str.strip()   # remove unwanted spaces
Gen = gen_data["unit capacity"].values

 print(f"gen: {Gen}")