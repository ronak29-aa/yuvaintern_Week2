from sklearn.datasets import load_iris
import pandas as pd
from pathlib import Path
OUT=Path("data/iris_storytelling.csv"); OUT.parent.mkdir(parents=True,exist_ok=True)
iris=load_iris(as_frame=True)
df=iris.frame.rename(columns={"sepal length (cm)":"sepal_length_cm","sepal width (cm)":"sepal_width_cm","petal length (cm)":"petal_length_cm","petal width (cm)":"petal_width_cm","target":"species_id"})
df["species"]=df.pop("species_id").map(dict(enumerate(iris.target_names)))
df.insert(0,"record_id",range(1,len(df)+1))
df["sepal_area_cm2"]=df.sepal_length_cm*df.sepal_width_cm
df["petal_area_cm2"]=df.petal_length_cm*df.petal_width_cm
df.to_csv(OUT,index=False)
print(f"Saved {len(df)} records to {OUT}")
