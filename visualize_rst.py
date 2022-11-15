import pickle
with open('data/performance.pkl', 'rb') as f:
    data = pickle.load(f)

for k,v in data.items():
    print("\n",k)
    for i in v:
        print(f"{i:.2f}", end=" ")
    print()
