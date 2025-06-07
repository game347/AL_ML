import pandas as pd
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', None)

csv_file = pd.read_csv("muffin_vs_cupcake_integer_dataset.csv")
print(csv_file.info())
# sns.jointplot(x="smokes_per_day",y="drinks_per_week",data=csv_file,alpha=0.5) show some
# sns.pairplot(csv_file, kind="scatter", plot_kws={'alpha': 0.4})  #use to show every thing in graph
# sns.lmplot(x='flour',
#            y='sugar',
#            data=csv_file,
#            hue='label',
#            palette='Set1',
#            fit_reg=False,
#            scatter_kws={'alpha': 1})
types_label = np.where(csv_file['label']=='muffin',0,1)
classifile = csv_file.columns.values[0:8].tolist()
print(classifile)
ingredients = csv_file[['flour','sugar']].values
print(ingredients)
model = svm.SVC(kernel='linear')  # C-Support Vector Classification.
model.fit(ingredients,types_label)
print(model.fit(ingredients,types_label))
# get the hyper plane
w = model.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(0,100,500) # make the rang of line
yy = a * xx - (model.intercept_[0] / w[1])
# plot the paralles to seperate hyperplane that pass
b = model.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
b = model.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])
print(yy)
sns.scatterplot(x='flour', y='sugar', hue='label', data=csv_file, palette='Set1', alpha=1)
plt.plot(xx, yy, linewidth=2, color='black')
plt.show()