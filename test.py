import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt



with open("data/train_list.pickle", "rb") as f:
    test_pairs = pickle.load(f)

with open("data/idx_to_name.pickle", 'rb') as f:
    idx_to_name = pickle.load(f)

new_list = []
for i in test_pairs:
    temp = [i[0], idx_to_name[i[0]], [idx_to_name[j] for j in i[2]]]
    new_list.append(temp)

with open("data/test_list.pickle", 'wb') as f:
    pickle.dump(new_list, f)

for i in new_list:
    print(i)
    exit()

# # same name author
# sum_author_num = 0
# author_count = {}
# for j in tqdm(test_pairs, total=len(test_pairs)):
#     sum_author_num = sum_author_num + len(j[1])
#     if len(j[1]) not in author_count:
#         author_count[len(j[1])] = 0
#     author_count[len(j[1])] += 1

# print(sum_author_num / len(test_pairs))

# plt.bar(author_count.keys(), author_count.values())
# plt.show()

# # co-author
# sum_author_num = 0
# author_count = {}
# for j in tqdm(test_pairs, total=len(test_pairs)):
#     sum_author_num = sum_author_num + len(j[2])
#     if len(j[2]) not in author_count:
#         author_count[len(j[2])] = 0
#     author_count[len(j[2])] += 1

# print(sum_author_num / len(test_pairs))

# plt.bar(author_count.keys(), author_count.values())
# plt.show()