import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', None)

csv_file = pd.read_csv("muffin_vs_cupcake_integer_dataset.csv")
print(csv_file.info())
# sns.jointplot(x="smokes_per_day",y="drinks_per_week",data=csv_file,alpha=0.5) show some
# sns.pairplot(csv_file, kind="scatter", plot_kws={'alpha': 0.4})  #use to show every thing in graph
sns.lmplot(x='flour',
           y='sugar',
           data=csv_file,
           hue='label',
           palette='Set1',
           fit_reg=False,
           scatter_kws={'alpha': 0.3})
plt.show()