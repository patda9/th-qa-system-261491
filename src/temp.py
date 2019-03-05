# get doc id
# group sentences
# find original token index => [0, 2, 4, 5, 8, 13, ...]
# find original token lenghts => [9, 16, 24, 26, 32, 40, ...]
# find all sentence ranges => [(0, 20), (10, 30), (20, 40), ...]
# vectorize all questions
# calculate similarity of question[i] and candidates[i]
# choose 15 candidate sentences with highest similarity
# candidate = {
##     "question_id": question_idx + 1,
##     "sentence": candidate_documents[question_idx][i][begin_index:end_index],
##      TODO change sentence to sentenge range instead, add document id
##     "answer_begin_position ": begin_position,
##     "answer_end_position": end_position,
##     "article_id": candidate_document_ids[question_idx][i],
##     "similarity_score": float(score)
## }
# write candidate sentences => .json
