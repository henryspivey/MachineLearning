import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("data/web_traffic.tsv", delimiter="\t")
print(data[:10])
print(data.shape)

x = data[:,0]
y = data[:,1]

print(np.sum(np.isnan(y)))

x = x[~np.isnan(y)]
y = y[~np.isnan(y)]


def plot_web_traffic(x,y, models=None):
    plt.figure(figsize=(12,6))
    plt.scatter(x,y,s=10)
    plt.title("Web Traffic over the last month")

    plt.xlabel("Time")
    plt.ylabel("Hits/hour")
    plt.xticks([w*7*24 for w in range(5)], ['week %i' %(w+1) for w in range(5)])

    if models:
        colors = ['g', 'k', 'b', 'm', 'r']
        linestyles = ['-', '-', '--', ':', '-']

        mx = sp.linspace(0, x[-1], 1000)
        for model, style, color in zip(models, linestyles, colors):
            plt.plot(mx, model(mx), linestyle = style, linewidth=2, c=color)
        plt.legend(["d=%i" % m.order for m in models], loc="upper left")

    plt.autoscale(tight=True)
    plt.grid()
    plt.show()


plot_web_traffic(x,y)
