import pandas as pd, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
DATA=Path("data/iris_storytelling.csv"); OUT=Path("visualizations"); OUT.mkdir(exist_ok=True)
df=pd.read_csv(DATA); n=["sepal_length_cm","sepal_width_cm","petal_length_cm","petal_width_cm"]; plt.rcParams.update({"figure.dpi":130,"axes.titleweight":"bold"})
# 1 distributions
fig,ax=plt.subplots(2,2,figsize=(11,8))
for a,c in zip(ax.ravel(),n):
    for s in sorted(df.species.unique()): a.hist(df.loc[df.species==s,c],bins=12,alpha=.45,label=s)
    a.set_title(c.replace("_"," ").title()); a.set_xlabel("Centimetres"); a.set_ylabel("Count")
ax[0,0].legend(frameon=False); fig.suptitle("How Do the Three Species Differ Across Their Measurements?",fontsize=16); fig.tight_layout(rect=[0,0,1,.96]); fig.savefig(OUT/"01_feature_distributions_by_species.png",bbox_inches="tight"); plt.close(fig)
# 2 petal box + points
fig,a=plt.subplots(figsize=(9,6)); gs=[df.loc[df.species==s,"petal_length_cm"] for s in sorted(df.species.unique())]; a.boxplot(gs,tick_labels=sorted(df.species.unique()))
for i,s in enumerate(sorted(df.species.unique()),1):
    v=df.loc[df.species==s,"petal_length_cm"]; a.scatter(np.random.default_rng(i).normal(i,.045,len(v)),v,alpha=.45,s=18)
a.set_title("Petal Length Reveals a Strong Species Difference"); a.set_xlabel("Species"); a.set_ylabel("Petal length (cm)"); fig.tight_layout(); fig.savefig(OUT/"02_petal_length_distribution.png",bbox_inches="tight"); plt.close(fig)
# 3 relationship
fig,a=plt.subplots(figsize=(9,6))
for s in sorted(df.species.unique()):
    q=df[df.species==s]; a.scatter(q.petal_length_cm,q.petal_width_cm,label=s,alpha=.7); z=np.polyfit(q.petal_length_cm,q.petal_width_cm,1); x=np.linspace(q.petal_length_cm.min(),q.petal_length_cm.max(),50); a.plot(x,np.polyval(z,x))
a.set_title("Petal Length and Width Move Together"); a.set_xlabel("Petal length (cm)"); a.set_ylabel("Petal width (cm)"); a.legend(frameon=False); fig.tight_layout(); fig.savefig(OUT/"03_petal_relationship_regression.png",bbox_inches="tight"); plt.close(fig)
# 4 correlation
corr=df[n].corr(); fig,a=plt.subplots(figsize=(9,7)); im=a.imshow(corr,cmap="viridis",vmin=-1,vmax=1); fig.colorbar(im,ax=a,label="Correlation"); lab=[x.replace("_"," ").title() for x in n]; a.set_xticks(range(4),lab,rotation=30,ha="right"); a.set_yticks(range(4),lab); a.set_title("Which Measurements Move Together?")
for i in range(4):
    for j in range(4): a.text(j,i,f"{corr.iloc[i,j]:.2f}",ha="center",va="center")
fig.tight_layout(); fig.savefig(OUT/"04_correlation_heatmap.png",bbox_inches="tight"); plt.close(fig)
# 5 radar
means=df.groupby("species")[n].mean(); z=(means-means.mean())/means.std(); ang=np.linspace(0,2*np.pi,4,endpoint=False).tolist(); ang+=ang[:1]; fig=plt.figure(figsize=(9,8)); a=fig.add_subplot(111,polar=True)
for s in means.index:
    v=z.loc[s].tolist()+[z.loc[s].iloc[0]]; a.plot(ang,v,linewidth=2,label=s); a.fill(ang,v,alpha=.08)
a.set_xticks(ang[:-1],lab); a.set_title("Species Profiles Relative to the Overall Average",pad=25); a.legend(frameon=False,bbox_to_anchor=(1.25,1.12)); fig.tight_layout(); fig.savefig(OUT/"05_species_radar_profile.png",bbox_inches="tight"); plt.close(fig)
# 6 PCA
X=StandardScaler().fit_transform(df[n]); p=PCA(2); pcs=p.fit_transform(X); fig,a=plt.subplots(figsize=(9,6))
for s in sorted(df.species.unique()):
    q=pcs[df.species==s]; a.scatter(q[:,0],q[:,1],label=s,alpha=.75)
a.axhline(0,linewidth=.8); a.axvline(0,linewidth=.8); a.set_title(f"PCA Species Map ({p.explained_variance_ratio_.sum()*100:.1f}% of variance)"); a.set_xlabel("Principal Component 1"); a.set_ylabel("Principal Component 2"); a.legend(frameon=False); fig.tight_layout(); fig.savefig(OUT/"06_pca_species_map.png",bbox_inches="tight"); plt.close(fig)
print("Six visualizations generated.")
