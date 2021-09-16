from sentence_transformers import SentenceTransformer, util
sentences = [
            "Stephen Sanwo cash transfer",
            "cash payment to Adeolu Taiwo",
            "Funds transfer to Hugging Face",
            "Procurement of data from MTN",
            "Professional service fee KPMG",
               "Staff Advance Stephen Sanwo"
               ]

# sentences = {
#         "source_sentence": "Cash transfer to Stephen Sanwo",
#         "sentence compares to": [
#             "Stephen Sanwo cash transfer",
#             "cash payment to Adeolu Taiwo",
#             "Funds transfer to Hugging Face",
#             "Procurement of data from MTN",
#             "Professional service fee KPMG",
#                "Staff Advance Stephen Sanwo"
#                ]
#     },

model = SentenceTransformer('./sentence_transformers_model')
# embeddings = model.encode(sentences)

# print(embeddings)

#Encode all sentences
embeddings = model.encode(sentences)

#Compute cosine similarity between all pairs
cos_sim = util.cos_sim(embeddings, embeddings)

#Add all pairs to a list with their cosine similarity score
all_sentence_combinations = []
for i in range(len(cos_sim)-1):
    for j in range(i+1, len(cos_sim)):
        all_sentence_combinations.append([cos_sim[i][j], i, j])

#Sort list by the highest cosine similarity score
all_sentence_combinations = sorted(all_sentence_combinations, key=lambda x: x[0], reverse=True)

for score, i, j in all_sentence_combinations:
    print("{} \t {} \t {:.4f}".format(sentences[i], sentences[j], cos_sim[i][j]))