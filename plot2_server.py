import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.ticker import StrMethodFormatter
color1 = (66/255 ,122/255 ,178/255 )
color2 = (240/255 ,145/255 ,72/255 )
color3 = (255/255 ,152/255 ,150/255 )
color4 = (219/255 ,219/255 ,141/255 )
color5 = (197/255 ,157/255 ,148/255 )
color6 = (175/255 ,199/255 ,232/255 )

data = {
    'x': list(range(1, 10)),
    '1/2^n': list(100*np.array([0.792387543253,0.824826989619,0.835207612457,0.835207612457,0.840830449827,0.852941176471,0.858996539792,0.86937716263,0.875432525952])),
    '1/n^2': list(100*np.array([0.792387543253,0.820934256055,0.825692041522,0.830017301038,0.841262975779,0.859861591696,0.867647058824,0.869809688581,0.837802768166])),
    'pure': list(100*np.array([0.792387543253,0.82785467128,0.775519031142,0.726643598616,0.684688581315,0.568771626298,0.459342560554,0.424740484429,0.376730103806])),
    'Ja':list(100*np.array([0.0099481,0.0099481,0.0142734,0.0895329,0.266003,0.440744,0.541955,0.645329,0.749567])),
    'Ma':list(100*np.array([0,0,0.7413494809688581,0.6561418685121108,0.5856401384083045,0.5743944636678201,0.6816608996539792,0.7115051903114187,0.6371107266435986])),
    'JS':list(100*np.array([0.430363,0.846886,0.87846,0.812716,0.670848,0.531142,0.5359,0.566609,0.635813])),
}


df = pd.DataFrame(data)

sns.set(style="white")

fig, ax = plt.subplots(figsize=(8, 6))

sns.lineplot(data=df, x="x", y="1/2^n", label="NV1",color=color3)
sns.lineplot(data=df, x="x", y="1/n^2", label="NV2",color=color6)
sns.lineplot(data=df, x="x", y="pure", label="NV3",color=color5)
sns.lineplot(data=df, x="x", y="Ma", label="Markov",color=color2)
sns.lineplot(data=df, x="x", y="JS", label="Jensen",color=color1)
sns.lineplot(data=df, x="x", y="Ja", label="Jaccard",color=color4)


ax.axhline(y=92.73, color='black', linestyle='--', label='Optimal metric')


plt.yticks([10,30,50,70,90]) 
plt.xlabel("K", fontsize=15)
plt.ylabel("Testing accuracy (%)", fontsize=15)
#plt.title("Comparison with other alignment-free methods")
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend(fontsize=12,loc="lower right")
plt.savefig("fig_Compare.eps", format='eps', bbox_inches='tight',dpi=300)
plt.show()
