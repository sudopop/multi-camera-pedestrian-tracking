import pickle

# data =[[1 1 1],[1 1 1],[1 1 1]]
#
# with open('data.pickle', 'wb') as f:
#     pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
#
with open(f"data/homography/data_v3.pickle", 'rb') as f:
    data = pickle.load(f)
    print(data)
    data = pickle.load(f)
    print(data)
    data = pickle.load(f)
    print(data)


# with open(f"data/homography/matrices-B.pickle", "rb") as f:
#     data = pickle.load(f)
#     print(data)
#     data = pickle.load(f)
#     print(data)
#     data = pickle.load(f)
#     print(data)