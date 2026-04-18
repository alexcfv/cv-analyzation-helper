This is a system that processes resumes into vector embeddings and retrieves the most relevant candidates based on a job query. It uses an LLM (Mistral) to rank results and generate explanations, enabling automated and intelligent candidate screening.

Candidate Ranking Algorithm
Group results by resume
We take all found chunks and group them by their source (each source = one resume).
Analyze each resume
For every resume, we look at all its chunks.

Calculate score
For each resume:

find the best chunk (smallest distance)
calculate the average distance
combine them:
score = 0.7 * best + 0.3 * average
Rank candidates
We sort resumes by score:
lower score = better match